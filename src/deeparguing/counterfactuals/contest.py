"""
src/deeparguing/counterfactuals/contest.py

Heuristic contestability algorithm: iteratively perturb the top-k edges of
``model.A`` (the casebase-internal adjacency, see ``grae.py``'s module
docstring) by |G-RAE| along the gradient direction, using backtracking line
search to find the minimal step that crosses the target class threshold.

Update rule summary:
  - Edges moved:  top-k by |G-RAE| magnitude (k is a tunable sweep param)
  - Step size:    backtracking line search (minimizes weight delta)
  - Stopping:     target class strength >= threshold + margin, or max_iters
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR

from .grae import compute_grae

# ---- Config -----------------------------------------------------------

DEFAULT_K = 3             # edges perturbed per iteration; sweep 1,3,5
THRESHOLD = 0.5           # classification boundary on target semantics score
MARGIN = 0.01             # cross threshold + margin, not exactly at it
ALPHA_MAX = 1.0           # initial line-search step size
BACKTRACK_FACTOR = 0.5    # shrink factor per failed trial
MAX_BACKTRACKS = 10       # per-iteration line-search retry cap
MAX_ITERS = 50            # outer loop cap -> mark as failed-to-flip if hit


class EdgeTraceStep(NamedTuple):
    """One accepted perturbation step. Tuple-unpacking still works
    (edge_ids, alpha, old_weights, new_weights, old_strength, new_strength)
    but prefer the named fields for readability."""

    edge_ids: list[int]
    alpha: float
    old_weights: list[float]
    new_weights: list[float]
    old_strength: float
    new_strength: float


@dataclass
class ContestResult:
    success: bool
    iterations: int
    max_weight_delta: float
    edge_trace: list[EdgeTraceStep] = field(default_factory=list)
    final_strength: float | None = None


def _casebase_grae(model: GradualAACBR, sample: Tensor, target_class: int) -> Tensor:
    """Gradient of ``target_class``'s strength for ``sample`` w.r.t. every
    entry of ``model.A``, flattened to 1-D so it lines up with the flat
    indices ``select_top_k``/``backtracking_line_search`` operate on."""
    result = compute_grae(model, sample, target_indices=[target_class])
    return result.casebase_edges.reshape(-1)


def _target_strength(
    model: GradualAACBR, sample: Tensor, target_class: int, A: Tensor
) -> float:
    """Forward pass of ``sample`` with ``model.A`` temporarily swapped for ``A``."""
    original_A = model.A
    try:
        model.A = A
        with torch.no_grad():
            strengths = model(sample)
    finally:
        model.A = original_A
    return strengths[0, target_class].item()


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
    threshold: float,
    margin: float,
    alpha_max: float = ALPHA_MAX,
    factor: float = BACKTRACK_FACTOR,
    max_backtracks: int = MAX_BACKTRACKS,
) -> tuple[float, Tensor, float] | None:
    """Shrink alpha from alpha_max until the trial step crosses
    threshold + margin, or backtracks are exhausted.
    Returns: (accepted_alpha, new_A, new_strength), or the smallest-alpha
    trial if none crossed cleanly, or None if max_backtracks == 0.
    """
    assert model.A is not None

    alpha = alpha_max
    best: tuple[float, Tensor, float] | None = None  # fallback: smallest-alpha trial
    for _ in range(max_backtracks):
        trial_A = _perturb_adjacency(model.A, edge_indices, direction, alpha)
        trial_strength = _target_strength(model, sample, target_class, trial_A)
        if trial_strength >= threshold + margin:
            return alpha, trial_A, trial_strength
        best = (alpha, trial_A, trial_strength)
        alpha *= factor
    return best  # None -> caller treats as no acceptable step this iteration


def contest(
    model: GradualAACBR,
    sample: Tensor,
    target_class: int,
    k: int = DEFAULT_K,
    threshold: float = THRESHOLD,
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
    strength = _target_strength(model, sample, target_class, model.A)
    iters_run = 0

    for iters_run in range(1, max_iters + 1):
        if strength >= threshold + margin:
            return ContestResult(True, iters_run - 1, max_delta, trace, strength)

        grae_vector = _casebase_grae(model, sample, target_class)
        edge_indices = select_top_k(grae_vector, k)
        direction = grae_vector[edge_indices]

        step = backtracking_line_search(
            model, sample, target_class, edge_indices, direction,
            threshold=threshold, margin=margin,
        )

        if step is None:
            break  # plateaued: no direction improves strength further

        alpha, new_A, new_strength = step
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
                strength,
                new_strength,
            )
        )

        model.A = new_A
        strength = new_strength

    # exhausted max_iters or plateaued without crossing threshold
    if strength >= threshold + margin:
        return ContestResult(True, iters_run - 1, max_delta, trace, strength)
    return ContestResult(False, iters_run, max_delta, trace, strength)
