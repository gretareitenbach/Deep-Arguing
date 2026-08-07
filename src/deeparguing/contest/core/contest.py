"""
src/deeparguing/contest/core/contest.py

Single-sample contestability search: iteratively perturbs the top-k edges of
``model.A`` along the G-RAE gradient direction, using a bracket-and-bisect
line search to find the minimal step that makes the target class beat the
best rival class by at least ``margin``. Falls back to
``bottleneck.find_and_escape_bottleneck`` when the gradient is uniformly ~0
(a saturated ReLU node).
"""

from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR

from .grae import compute_grae

# ---- Config -----------------------------------------------------------

DEFAULT_K = 3               # edges perturbed per iteration; sweep 1,3,5
THRESHOLD = 0.5              # virtual rival strength when target_class has no real rival
MARGIN = 0.01                # target must beat the best rival class by this much
LIVE_GRAD_THRESHOLD = 1e-9   # max|grae_vector| at or below this counts as "dead"
ALPHA_MAX = 1.0               # initial line-search step size
BACKTRACK_FACTOR = 0.5       # shrink factor per failed bracketing trial
MAX_BACKTRACKS = 10          # bracketing-phase retry cap (per iteration)
MAX_BISECTIONS = 30          # refinement-phase retry cap, once a bracket is found
BISECT_TOL = 1e-6            # stop bisecting once the bracket is this narrow
MAX_ITERS = 50               # outer loop cap -> mark as failed-to-flip if hit


class EdgeTraceStep(NamedTuple):
    """One accepted perturbation step.

    Fields: edge_ids, alpha, old_weights, new_weights, old_target_strength,
    new_target_strength, old_rival_class, old_rival_strength,
    new_rival_class, new_rival_strength. ``*_rival_class`` is ``None`` if
    target_class has no real rival.
    """

    edge_ids: list[int]
    alpha: float
    old_weights: list[float]
    new_weights: list[float]
    old_target_strength: float
    new_target_strength: float
    old_rival_class: int | None
    old_rival_strength: float
    new_rival_class: int | None
    new_rival_strength: float


@dataclass
class ContestResult:
    """Outcome of ``contest()``: whether the target class flipped, how many
    iterations it took, the largest single edge-weight change made, the
    full step-by-step trace, and the final target/rival strengths."""

    success: bool
    iterations: int
    max_weight_delta: float
    edge_trace: list[EdgeTraceStep] = field(default_factory=list)
    final_target_strength: float | None = None
    final_rival_class: int | None = None
    final_rival_strength: float | None = None


def _casebase_grae(model: GradualAACBR, sample: Tensor, target_class: int) -> Tensor:
    """Gradient of ``target_class``'s strength for ``sample`` w.r.t. every
    entry of ``model.A``.

    Returns a flattened (n*n*d,) tensor matching ``select_top_k``'s index
    convention.
    """
    result = compute_grae(model, sample, target_indices=[target_class])
    return result.casebase_edges.reshape(-1)


def _forward_strengths(model: GradualAACBR, sample: Tensor, A: Tensor) -> Tensor:
    """Forward pass of ``sample`` with ``model.A`` temporarily swapped for ``A``.

    Returns every default class's strength, shape (D,).
    """
    original_A = model.A
    try:
        model.A = A
        with torch.no_grad():
            strengths = model(sample)
    finally:
        model.A = original_A
    return strengths[0]


def _target_and_rival(
    strengths: Tensor, target_class: int, threshold: float
) -> tuple[float, int | None, float]:
    """Split a strength vector into (target_strength, rival_class,
    rival_strength), where rival is the highest-strength class other than
    ``target_class``.

    If ``target_class`` is the only default argument, ``rival_class`` is
    ``None`` and ``rival_strength`` falls back to ``threshold``.
    """
    if strengths.numel() == 1:
        return strengths[target_class].item(), None, threshold
    other = strengths.clone()
    other[target_class] = -torch.inf
    rival_class = int(other.argmax().item())
    return strengths[target_class].item(), rival_class, strengths[rival_class].item()


def _forward_strengths_batch(model: GradualAACBR, samples: Tensor, A: Tensor) -> Tensor:
    """Batched analogue of ``_forward_strengths``.

    Returns every default class's strength for every sample, shape (B, D).
    """
    original_A = model.A
    try:
        model.A = A
        with torch.no_grad():
            strengths = model(samples)
    finally:
        model.A = original_A
    return strengths


def _target_and_rival_batch(
    strengths: Tensor, target_classes: Sequence[int], threshold: float
) -> tuple[Tensor, list[int | None], Tensor]:
    """Batched analogue of ``_target_and_rival``.

    Returns (target_strengths, rival_classes, rival_strengths), each of
    length B (shape (B,) for the tensors).
    """
    batch_size, num_classes = strengths.shape
    idx = torch.arange(batch_size, device=strengths.device)
    target_t = torch.as_tensor(list(target_classes), dtype=torch.long, device=strengths.device)
    target_strengths = strengths[idx, target_t]

    if num_classes == 1:
        rival_classes: list[int | None] = [None] * batch_size
        rival_strengths = torch.full_like(target_strengths, threshold)
        return target_strengths, rival_classes, rival_strengths

    other = strengths.clone()
    other[idx, target_t] = -torch.inf
    rival_t = other.argmax(dim=-1)
    rival_strengths = strengths[idx, rival_t]
    return target_strengths, rival_t.tolist(), rival_strengths


def _perturb_adjacency(
    A: Tensor, edge_indices: Tensor, direction: Tensor, alpha: float
) -> Tensor:
    """Copy of ``A`` with the entries at ``edge_indices`` (flat indices)
    shifted by ``alpha * direction`` and clamped to [-1, 1]; all other
    entries are unchanged.
    """
    new_A = A.detach().clone()
    flat = new_A.view(-1)
    flat[edge_indices] = torch.clamp(flat[edge_indices] + alpha * direction, min=-1.0, max=1.0)
    return new_A


def select_top_k(grae_vector: Tensor, k: int) -> Tensor:
    """Indices (into the flattened ``model.A``) of the k edges with largest |G-RAE|."""
    return grae_vector.abs().topk(k).indices


def _default_source_mask(default_indexes: Tensor, shape: torch.Size) -> Tensor:
    """Boolean mask, flattened to match a ``(n, n, d)``-shaped ``model.A``,
    marking every entry whose source node (the first axis) is one of
    ``default_indexes``.
    """
    n, m, d = shape
    mask = torch.zeros(n, m, d, dtype=torch.bool, device=default_indexes.device)
    mask[default_indexes] = True
    return mask.reshape(-1)


def _mask_default_sources(model: GradualAACBR, vector: Tensor) -> Tensor:
    """Zero out every entry of ``vector`` (flat, same ``n*n*d`` layout as
    ``model.A``) whose source is one of ``model.default_indexes``.

    A no-op when ``model.defaults_not_attack`` is False.
    """
    if not model.defaults_not_attack:
        return vector
    assert model.A is not None
    return vector.masked_fill(_default_source_mask(model.default_indexes, model.A.shape), 0.0)


def bisection_line_search(
    model: GradualAACBR,
    sample: Tensor,
    target_class: int,
    edge_indices: Tensor,
    direction: Tensor,
    margin: float,
    threshold: float = THRESHOLD,
    alpha_max: float = ALPHA_MAX,
    factor: float = BACKTRACK_FACTOR,
    max_backtracks: int = MAX_BACKTRACKS,
    max_bisections: int = MAX_BISECTIONS,
    bisect_tol: float = BISECT_TOL,
) -> tuple[float, Tensor, float, int | None, float] | None:
    """Two-phase search for close to the smallest alpha (along ``direction``)
    that makes target_class beat the best rival class by at least margin.

    Phase 1 (bracket): shrink alpha geometrically from ``alpha_max`` until a
    trial crosses the margin. Phase 2 (bisect): binary search inside that
    bracket to converge toward the minimal crossing point.

    Returns ``(accepted_alpha, new_A, new_target_strength, rival_class,
    rival_strength)`` for the smallest known-crossing alpha found, or the
    smallest-alpha trial if none crossed within ``max_backtracks``, or
    ``None`` if ``max_backtracks == 0``.
    """
    assert model.A is not None

    def trial(alpha: float) -> tuple[Tensor, float, int | None, float]:
        trial_A = _perturb_adjacency(model.A, edge_indices, direction, alpha)
        trial_strengths = _forward_strengths(model, sample, trial_A)
        trial_target, rival_class, trial_rival = _target_and_rival(
            trial_strengths, target_class, threshold
        )
        return trial_A, trial_target, rival_class, trial_rival

    def crossed(target: float, rival: float) -> bool:
        return target - rival >= margin

    # ---- Phase 1: bracket ----
    alpha_lo, alpha_hi = 0.0, alpha_max
    best: tuple[float, Tensor, float, int | None, float] | None = None
    hi_result: tuple[Tensor, float, int | None, float] | None = None

    alpha = alpha_max
    for _ in range(max_backtracks):
        trial_A, trial_target, rival_class, trial_rival = trial(alpha)
        if crossed(trial_target, trial_rival):
            alpha_hi, hi_result = alpha, (trial_A, trial_target, rival_class, trial_rival)
            break
        best = (alpha, trial_A, trial_target, rival_class, trial_rival)
        alpha_lo = alpha
        alpha *= factor
    else:
        return best  # never crossed within budget

    # ---- Phase 2: bisect within [alpha_lo, alpha_hi] ----
    for _ in range(max_bisections):
        if alpha_hi - alpha_lo < bisect_tol:
            break
        alpha_mid = (alpha_lo + alpha_hi) / 2
        trial_A, trial_target, rival_class, trial_rival = trial(alpha_mid)
        if crossed(trial_target, trial_rival):
            alpha_hi, hi_result = alpha_mid, (trial_A, trial_target, rival_class, trial_rival)
        else:
            alpha_lo = alpha_mid

    trial_A, trial_target, rival_class, trial_rival = hi_result
    return alpha_hi, trial_A, trial_target, rival_class, trial_rival


def contest(
    model: GradualAACBR,
    sample: Tensor,
    target_class: int,
    k: int = DEFAULT_K,
    threshold: float = THRESHOLD,
    margin: float = MARGIN,
    max_iters: int = MAX_ITERS,
    max_edits: int | None = None,
) -> ContestResult:
    """Contest a single sample's prediction towards ``target_class``.

    Parameters
    ----------
    model : GradualAACBR
        A fitted model (``model.A`` populated).
    sample : Tensor
        A single new case, shape (1, x1, ..., xn).
    target_class : int
        Which entry of ``model.default_indexes`` to push the sample's
        strength towards.
    k, threshold, margin, max_iters
        See module-level defaults above.
    max_edits : int | None
        Stop once this many distinct edges (flattened indices into
        ``model.A``) have been touched -- same semantics as
        ``batch_contest``'s ``max_edits``. ``None`` (default) is unbounded.

    Returns
    -------
    ContestResult
    """
    # Deferred import: bottleneck.py imports several private helpers back
    # from this module, so importing it at module load time would be circular.
    from .bottleneck import find_and_escape_bottleneck

    if model.A is None:
        raise Exception("Ensure the model has been fit first.")
    if sample.shape[0] != 1:
        raise ValueError(
            "contest expects a single new case (batch size 1), got batch of "
            f"{sample.shape[0]}."
        )

    max_delta = 0.0
    trace: list[EdgeTraceStep] = []
    touched_edges: set[int] = set()
    strengths = _forward_strengths(model, sample, model.A)
    target_strength, rival_class, rival_strength = _target_and_rival(
        strengths, target_class, threshold
    )
    iters_run = 0

    for iters_run in range(1, max_iters + 1):
        if target_strength - rival_strength >= margin:
            return ContestResult(
                True, iters_run - 1, max_delta, trace,
                target_strength, rival_class, rival_strength,
            )

        grae_vector = _mask_default_sources(model, _casebase_grae(model, sample, target_class))

        if grae_vector.abs().max().item() <= LIVE_GRAD_THRESHOLD:
            step = find_and_escape_bottleneck(
                model, sample, target_class, grae_vector, k=k, threshold=threshold
            )
        else:
            edge_indices = select_top_k(grae_vector, k)
            direction = grae_vector[edge_indices]
            bisection_step = bisection_line_search(
                model, sample, target_class, edge_indices, direction,
                margin=margin, threshold=threshold,
            )
            step = None if bisection_step is None else (edge_indices, *bisection_step)

        if step is None:
            break  # plateaued, or no edge can escape the bottleneck

        edge_indices, alpha, new_A, new_target_strength, new_rival_class, new_rival_strength = step
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

        touched_edges.update(edge_indices.tolist())
        if max_edits is not None and len(touched_edges) >= max_edits:
            break  # edit budget hit -- stop even if the margin isn't met yet

    if target_strength - rival_strength >= margin:
        return ContestResult(
            True, iters_run - 1, max_delta, trace,
            target_strength, rival_class, rival_strength,
        )
    return ContestResult(
        False, iters_run, max_delta, trace,
        target_strength, rival_class, rival_strength,
    )
