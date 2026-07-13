"""
src/deeparguing/counterfactuals/contest.py

Heuristic contestability algorithm: iteratively perturb the top-k edges of
``model.A`` (the casebase-internal adjacency, see ``grae.py``'s module
docstring) by |G-RAE| along the gradient direction, using a bracket-and-bisect
line search to find close to the minimal step that flips the model's argmax
prediction onto the target class.

Update rule summary:
  - Edges moved:  top-k by |G-RAE| magnitude (k is a tunable sweep param)
  - Step size:    bracket (geometric backtracking) then bisect within it to
                   converge toward the minimal crossing alpha
  - Stopping:     target class strength beats the best rival class's
                   strength by at least ``margin``, or max_iters

Note: strengths across classes are not mutually exclusive or bounded to
[0, 1] (no softmax/normalization ties them together -- see
``GradualAACBR.forward``'s final ``strengths @ W`` combination), so a fixed
absolute threshold cannot by itself tell you whether the target class
actually won the argmax. Every forward pass already computes every class's
strength at once, so tracking the best rival costs nothing extra -- we just
stop discarding it. If ``target_class`` is the *only* default argument
(the single-topic-argument setting from the original Contestability paper,
with no classification competition to speak of), there is no rival to
compare against; ``threshold`` is then used as a fixed virtual competitor,
recovering the original absolute-threshold criterion for that case.

This does *not* make the top-k edge selection rival-aware: ``_casebase_grae``
still differentiates only the target class's strength, so a step can raise
the rival too; the stopping check below will simply reject such a step and
try another top-k direction next iteration.

Dead gradients: ``_casebase_grae`` can come back uniformly ~0 for a sample
because ``target_class``'s own final strength is pinned at exactly 0 by its
outer ReLU -- since the aggregation
step is dense (every entry of ``model.A`` has a well-defined one-hop
gradient contribution equal to its source node's strength, whether or not
that entry happens to be nonzero right now), that outer saturation is what
zeroes the derivative *uniformly*, not merely some upstream node being
saturated. Fixing an upstream node that feeds the target (rather than the
target's own incoming edges directly) is still often the more useful lever,
since it can cascade forward and revive the target as a side effect. Each
iteration checks ``grae_vector``'s magnitude and routes accordingly: "live"
takes the ordinary top-k/bisection path below; "dead" instead calls
``bottleneck.find_and_escape_bottleneck``, which walks toward the target to
find a saturated node and grows a step that un-sticks it (see
``bottleneck.py``'s module docstring). Both paths feed the same
trace-recording/commit logic, so a dead-then-escaped sample continues
through ordinary gradient steps on the next iteration once
``_casebase_grae`` is recomputed and live again.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR

from .grae import compute_grae

# ---- Config -----------------------------------------------------------

DEFAULT_K = 3             # edges perturbed per iteration; sweep 1,3,5
THRESHOLD = 0.5           # virtual rival strength when target_class has no real rival
MARGIN = 0.01             # target must beat the best rival class by this much
LIVE_GRAD_THRESHOLD = 1e-9  # max|grae_vector| at or below this counts as "dead" -- routes
                            # to bottleneck.find_and_escape_bottleneck instead of the
                            # ordinary line search. Not re-validated against a real max|grad|
                            # distribution (bimodal: exact 0.0s vs. a comfortable spread
                            # above) -- check that assumption before relying on this value
                            # on a new checkpoint/dataset.
ALPHA_MAX = 1.0           # initial line-search step size
BACKTRACK_FACTOR = 0.5    # shrink factor per failed bracketing trial
MAX_BACKTRACKS = 10       # bracketing-phase retry cap (per iteration)
MAX_BISECTIONS = 30       # refinement-phase retry cap, once a bracket is found
                          # (each step is one cheap no_grad forward pass, and
                          # halves the bracket -- 30 gets ~1e-9 relative
                          # resolution on alpha_max=1.0 for ~20 extra passes
                          # over the old cap of 10)
BISECT_TOL = 1e-6         # stop bisecting once the bracket is this narrow
MAX_ITERS = 50            # outer loop cap -> mark as failed-to-flip if hit


class EdgeTraceStep(NamedTuple):
    """One accepted perturbation step. Tuple-unpacking still works but
    prefer the named fields for readability. ``*_rival_class`` is the
    strongest non-target class at that point (``None`` if target_class has
    no real rival -- see module docstring) -- tracked separately
    before/after because a step can change which class is the rival, not
    just its strength."""

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
    indices ``select_top_k``/``bisection_line_search`` operate on."""
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
    strengths: Tensor, target_class: int, threshold: float
) -> tuple[float, int | None, float]:
    """Split a strength vector into (target_strength, rival_class, rival_strength),
    where rival is the highest-strength class other than target_class -- the one
    that must be overtaken for target_class to actually win the argmax.

    If target_class is the only default argument, there are no other classes
    to compare against: rival_class is None and rival_strength falls back to
    the fixed threshold, so the caller's ``target - rival >= margin`` check
    still recovers the original single-topic-argument criterion."""
    if strengths.numel() == 1:
        return strengths[target_class].item(), None, threshold
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

    Phase 1 (bracket): shrink alpha geometrically from alpha_max, same as
    plain backtracking, until a trial crosses the margin -- this handles
    "the direction is weak, a bigger step is needed." This gives a bracket
    [last failing alpha (or 0), first crossing alpha].

    Phase 2 (bisect): binary search inside that bracket to converge toward
    the minimal crossing point, instead of accepting the first crossing
    trial found. Plain backtracking stops at the first success, which can
    accept a wildly oversized step whenever alpha_max itself overshoots the
    margin on the very first trial (no shrinking ever happens in that case).

    Returns: (accepted_alpha, new_A, new_target_strength, rival_class,
    rival_strength) for the smallest known-crossing alpha found, or the
    smallest-alpha trial if none crossed within max_backtracks, or None if
    max_backtracks == 0.
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
    best: tuple[float, Tensor, float, int | None, float] | None = None  # fallback: smallest-alpha trial tried
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
        return best  # never crossed within budget -> caller treats as no acceptable step

    # ---- Phase 2: bisect within [alpha_lo, alpha_hi] to refine toward the
    # minimal crossing point ----
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
) -> ContestResult:
    """Main loop. See module docstring for update rule."""
    # Deferred import: bottleneck.py imports several private helpers back
    # from this module, so importing it at module load time (before those
    # names exist yet) would be circular.
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

        grae_vector = _casebase_grae(model, sample, target_class)

        if grae_vector.abs().max().item() <= LIVE_GRAD_THRESHOLD:
            # Dead gradient: a saturated ReLU node is blocking flow to
            # target_class's default argument. Escape it instead of
            # searching along a direction that's uniformly ~0.
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
            break  # structurally hopeless: plateaued, or no edge can escape the bottleneck

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
