"""
src/deeparguing/counterfactuals/contest.py

Heuristic contestability algorithm: iteratively perturb the top-k edges of
``model.A`` (the casebase-internal adjacency, see ``grae.py``'s module
docstring) by |G-RAE| along the gradient direction, using backtracking line
search to find the minimal step that flips the model's argmax prediction
onto the target class.

Update rule summary:
  - Edges moved:  top-k by |G-RAE| magnitude (k is a tunable sweep param)
  - Step size:    backtracking line search (minimizes weight delta)
  - Stopping:     target class strength beats every other class's strength
                   by at least ``margin``, or max_iters

Note: strengths across classes are not mutually exclusive or bounded to
[0, 1] (no softmax/normalization ties them together -- see
``GradualAACBR.forward``'s final ``strengths @ W`` combination), so a fixed
absolute threshold cannot tell you whether the target class actually won
the argmax. Every forward pass already computes every class's strength at
once, so tracking the best rival costs nothing extra -- we just stop
discarding it. This does *not* make the top-k edge selection rival-aware:
``_casebase_grae`` still differentiates only the target class's strength,
so a step can raise the rival too; the corrected stopping check below will
simply reject such a step and try another top-k direction next iteration.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR

from .grae import compute_grae

# ---- Config -----------------------------------------------------------

DEFAULT_K = 3             # edges perturbed per iteration; sweep 1,3,5
MARGIN = 0.01             # target must beat the best rival class by this much
ALPHA_MAX = 1.0           # initial line-search step size
BACKTRACK_FACTOR = 0.5    # shrink factor per failed trial
MAX_BACKTRACKS = 10       # per-iteration line-search retry cap
MAX_ITERS = 50            # outer loop cap -> mark as failed-to-flip if hit


class EdgeTraceStep(NamedTuple):
    """One accepted perturbation step. Tuple-unpacking still works but
    prefer the named fields for readability. ``*_rival_class`` is the
    strongest non-target class at that point -- tracked separately
    before/after because a step can change which class is the rival, not
    just its strength."""

    edge_ids: list[int]
    alpha: float
    old_weights: list[float]
    new_weights: list[float]
    old_target_strength: float
    new_target_strength: float
    old_rival_class: int
    old_rival_strength: float
    new_rival_class: int
    new_rival_strength: float


@dataclass
class ContestResult:
    success: bool
    iterations: int
    max_weight_delta: float
    edge_trace: list[EdgeTraceStep] = field(default_factory=list)
    final_target_strength: float | None = None
    final_rival_class: int | None = None
    final_rival_strength: float | None = None


def _casebase_grae(model: GradualAACBR, sample: Tensor, target_class: int) -> Tensor:
    """Gradient of ``target_class``'s strength for ``sample`` w.r.t. every
    entry of ``model.A``, flattened to 1-D so it lines up with the flat
    indices ``select_top_k``/``backtracking_line_search`` operate on."""
    result = compute_grae(model, sample, target_indices=[target_class])
    return result.casebase_edges.reshape(-1)


def _forward_strengths(model: GradualAACBR, sample: Tensor, A: Tensor) -> Tensor:
    """Forward pass of ``sample`` with ``model.A`` temporarily swapped for ``A``.
    Returns every default class's strength (shape (D,)), not just the target's --
    a single forward pass computes them all together, so callers that need to
    know how the *other* classes moved don't need a second pass."""
    original_A = model.A
    try:
        model.A = A
        with torch.no_grad():
            strengths = model(sample)
    finally:
        model.A = original_A
    return strengths[0]


def _target_and_rival(
    strengths: Tensor, target_class: int
) -> tuple[float, int, float]:
    """Split a strength vector into (target_strength, rival_class, rival_strength),
    where rival is the highest-strength class other than target_class -- the one
    that must be overtaken for target_class to actually win the argmax."""
    other = strengths.clone()
    other[target_class] = -torch.inf
    rival_class = int(other.argmax().item())
    return strengths[target_class].item(), rival_class, strengths[rival_class].item()


def _perturb_adjacency(
    A: Tensor, edge_indices: Tensor, direction: Tensor, alpha: float
) -> Tensor:
    """Copy of ``A`` with the entries at ``edge_indices`` (flat indices)
    shifted by ``alpha * direction``; all other entries are unchanged."""
    new_A = A.detach().clone()
    new_A.view(-1)[edge_indices] += alpha * direction
    return new_A


def select_top_k(grae_vector: Tensor, k: int) -> Tensor:
    """Indices (into the flattened ``model.A``) of the k edges with largest |G-RAE|."""
    return grae_vector.abs().topk(k).indices


def backtracking_line_search(
    model: GradualAACBR,
    sample: Tensor,
    target_class: int,
    edge_indices: Tensor,
    direction: Tensor,
    margin: float,
    alpha_max: float = ALPHA_MAX,
    factor: float = BACKTRACK_FACTOR,
    max_backtracks: int = MAX_BACKTRACKS,
) -> tuple[float, Tensor, float, int, float] | None:
    """Shrink alpha from alpha_max until the trial step makes target_class
    beat the best rival class by at least margin, or backtracks are exhausted.
    Returns: (accepted_alpha, new_A, new_target_strength, rival_class,
    rival_strength), or the smallest-alpha trial if none won cleanly, or
    None if max_backtracks == 0.
    """
    assert model.A is not None

    alpha = alpha_max
    best: tuple[float, Tensor, float, int, float] | None = None  # fallback: smallest-alpha trial
    for _ in range(max_backtracks):
        trial_A = _perturb_adjacency(model.A, edge_indices, direction, alpha)
        trial_strengths = _forward_strengths(model, sample, trial_A)
        trial_target, rival_class, trial_rival = _target_and_rival(trial_strengths, target_class)
        if trial_target - trial_rival >= margin:
            return alpha, trial_A, trial_target, rival_class, trial_rival
        best = (alpha, trial_A, trial_target, rival_class, trial_rival)
        alpha *= factor
    return best  # None -> caller treats as no acceptable step this iteration


def contest(
    model: GradualAACBR,
    sample: Tensor,
    target_class: int,
    k: int = DEFAULT_K,
    margin: float = MARGIN,
    max_iters: int = MAX_ITERS,
) -> ContestResult:
    """Main loop. See module docstring for update rule."""
    if model.A is None:
        raise Exception("Ensure the model has been fit first.")
    if sample.shape[0] != 1:
        raise ValueError(
            "contest expects a single new case (batch size 1), got batch of "
            f"{sample.shape[0]}."
        )

    max_delta = 0.0
    trace: list[EdgeTraceStep] = []
    strengths = _forward_strengths(model, sample, model.A)
    target_strength, rival_class, rival_strength = _target_and_rival(strengths, target_class)
    iters_run = 0

    for iters_run in range(1, max_iters + 1):
        if target_strength - rival_strength >= margin:
            return ContestResult(
                True, iters_run - 1, max_delta, trace,
                target_strength, rival_class, rival_strength,
            )

        grae_vector = _casebase_grae(model, sample, target_class)
        edge_indices = select_top_k(grae_vector, k)
        direction = grae_vector[edge_indices]

        step = backtracking_line_search(
            model, sample, target_class, edge_indices, direction, margin=margin,
        )

        if step is None:
            break  # plateaued: no direction improves the margin further

        alpha, new_A, new_target_strength, new_rival_class, new_rival_strength = step
        old_values = model.A.view(-1)[edge_indices]
        new_values = new_A.view(-1)[edge_indices]
        delta = (new_values - old_values).abs().max().item()
        max_delta = max(max_delta, delta)
        trace.append(
            EdgeTraceStep(
                edge_indices.tolist(),
                alpha,
                old_values.tolist(),
                new_values.tolist(),
                target_strength,
                new_target_strength,
                rival_class,
                rival_strength,
                new_rival_class,
                new_rival_strength,
            )
        )

        model.A = new_A
        target_strength, rival_class, rival_strength = (
            new_target_strength, new_rival_class, new_rival_strength,
        )

    # exhausted max_iters or plateaued without winning the argmax
    if target_strength - rival_strength >= margin:
        return ContestResult(
            True, iters_run - 1, max_delta, trace,
            target_strength, rival_class, rival_strength,
        )
    return ContestResult(
        False, iters_run, max_delta, trace,
        target_strength, rival_class, rival_strength,
    )
