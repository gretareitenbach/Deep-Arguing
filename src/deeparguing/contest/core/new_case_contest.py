"""
src/deeparguing/contest/core/new_case_contest.py

Single-sample contestability search restricted to a new case's own edges
into the casebase (``model.new_cases_attacks_adjacency``), instead of
``contest.py``'s edits to the shared ``model.A``. Same bracket-and-bisect
line search as ``contest.py``, retargeted from ``A`` to a sample's own
irrelevance row ``E`` (``E = -irrelevance_edge_weights(sample, casebase)``).

Unlike ``contest()``, nothing is ever written back onto the model: ``E`` is
recomputed from scratch by the network on every real forward pass (see
``GradualAACBR.__new_case_influence``), so there is nothing meaningful to
persist there -- an edit committed to ``model.new_cases_attacks_adjacency``
would just be overwritten the next time anything calls ``model(...)``. This
also means ``_new_case_grae`` cannot simply reread ``model``'s live state
across outer iterations the way ``contest.py``'s ``_casebase_grae`` rereads
``model.A`` (which ``contest()`` *does* mutate in place each iteration) --
it differentiates at an explicitly-passed ``E`` instead, so the gradient
direction stays correct as the search moves away from the network's
original output. Callers collect the returned ``final_E`` (and the trace's
old/new weights) for downstream use, e.g. Tuesday's fine-tuning dataset of
``(sample, casebase_item, old_E, corrected_E)`` triples.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR

from .contest import (ALPHA_MAX, BACKTRACK_FACTOR, BISECT_TOL, DEFAULT_K,
                       LIVE_GRAD_THRESHOLD, MARGIN, MAX_BACKTRACKS,
                       MAX_BISECTIONS, MAX_ITERS, THRESHOLD, _target_and_rival,
                       select_top_k)
from .grae import _batched_casebase_base_scores, _replay_default_strengths

# ---- Config -------------------------------------------------------------

# E = -irrelevance_edge_weights(...), and irrelevance is valued in [0, 1]
# (RegularIrrelevance = 1 - partial_order, partial_order in [0, 1);
# FeatureWeightedIrrelevance = sigmoid(...) in (0, 1)) -- so E's valid range
# is [-1, 0], unlike _perturb_adjacency's [-1, 1] for A.
E_MIN = -1.0
E_MAX = 0.0


class NewCaseEdgeTraceStep(NamedTuple):
    """One accepted perturbation step, mirroring ``contest.py``'s
    ``EdgeTraceStep`` but for ``E`` instead of ``A``.

    Fields: edge_ids (flat indices into ``E``'s (n*d,) layout), alpha,
    old_weights, new_weights, old_target_strength, new_target_strength,
    old_rival_class, old_rival_strength, new_rival_class, new_rival_strength.
    ``*_rival_class`` is ``None`` if target_class has no real rival.
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
class NewCaseContestResult:
    """Outcome of ``new_case_contest()``: whether the target class flipped,
    how many iterations it took, the largest single ``E``-entry change
    made, the full step-by-step trace, the final target/rival strengths,
    and both the starting and final ``E``.

    ``initial_E``/``final_E`` are included (unlike ``ContestResult``, which
    has no analogue) because nothing is persisted onto the model -- callers
    must collect ``final_E`` themselves.
    """

    success: bool
    iterations: int
    max_weight_delta: float
    edge_trace: list[NewCaseEdgeTraceStep] = field(default_factory=list)
    final_target_strength: float | None = None
    final_rival_class: int | None = None
    final_rival_strength: float | None = None
    initial_E: Tensor | None = None
    final_E: Tensor | None = None


def _perturb_new_case_edges(
    E: Tensor, edge_indices: Tensor, direction: Tensor, alpha: float
) -> Tensor:
    """Copy of ``E`` (a single sample's (n, d) row of
    ``model.new_cases_attacks_adjacency``) with the entries at
    ``edge_indices`` (flat indices) shifted by ``alpha * direction`` and
    clamped to ``[E_MIN, E_MAX]``; all other entries are unchanged.
    """
    new_E = E.detach().clone()
    flat = new_E.view(-1)
    flat[edge_indices] = torch.clamp(flat[edge_indices] + alpha * direction, min=E_MIN, max=E_MAX)
    return new_E


def _new_case_grae(
    model: GradualAACBR,
    E: Tensor,
    target_class: int,
    casebase_base_scores: Tensor,
    new_cases_base_scores: Tensor,
) -> Tensor:
    """Gradient of ``target_class``'s strength w.r.t. every entry of ``E``
    (a single sample's own (n, d) row into the casebase), evaluated *at*
    the given ``E`` -- which may already be a search-perturbed value, not
    the network's live output.

    Reuses ``grae.py``'s ``_replay_default_strengths`` for the aggregation/
    influence/semantics replay (``model.A`` is detached first: it may carry
    a live graph back to the trainable partial-order network from ``fit()``,
    and we only want the gradient w.r.t. ``E``, not a wasted -- or
    potentially graph-already-freed -- backward through ``A`` too).

    Returns a flattened (n*d,) tensor matching ``select_top_k``'s index
    convention.
    """
    assert model.A is not None
    A = model.A.detach()
    E_leaf = E.detach().clone().unsqueeze(0).requires_grad_(True)
    target_strength = _replay_default_strengths(
        model, A, E_leaf, casebase_base_scores, new_cases_base_scores,
    )[0, target_class]
    target_strength.backward()
    assert E_leaf.grad is not None
    return E_leaf.grad.detach().reshape(-1)


def _forward_new_case_strengths(
    model: GradualAACBR,
    E: Tensor,
    casebase_base_scores: Tensor,
    new_cases_base_scores: Tensor,
) -> Tensor:
    """Trial forward pass with ``model.A`` fixed and the sample's own new-
    case row set to ``E`` (shape (1, n, d)), by replaying ``grae.py``'s
    ``_replay_default_strengths`` -- so neither ``model.A`` nor any live
    model state is touched, and no repeated ``compute_base_scores`` calls
    are needed once the caller has hoisted ``casebase_base_scores``/
    ``new_cases_base_scores`` (both invariant across the whole search: they
    depend only on ``model.A``/``model.X_train``/the sample, none of which
    ``new_case_contest`` ever changes).

    Returns every default class's strength, shape (D,).
    """
    assert model.A is not None
    with torch.no_grad():
        strengths = _replay_default_strengths(
            model, model.A, E, casebase_base_scores, new_cases_base_scores,
        )
    return strengths[0]


def bisection_line_search(
    model: GradualAACBR,
    target_class: int,
    E: Tensor,
    edge_indices: Tensor,
    direction: Tensor,
    casebase_base_scores: Tensor,
    new_cases_base_scores: Tensor,
    margin: float,
    threshold: float = THRESHOLD,
    alpha_max: float = ALPHA_MAX,
    factor: float = BACKTRACK_FACTOR,
    max_backtracks: int = MAX_BACKTRACKS,
    max_bisections: int = MAX_BISECTIONS,
    bisect_tol: float = BISECT_TOL,
) -> tuple[float, Tensor, float, int | None, float] | None:
    """Two-phase search for close to the smallest alpha (along ``direction``)
    that makes ``target_class`` beat the best rival class by at least
    ``margin`` -- same two-phase (bracket, then bisect) structure as
    ``contest.py``'s ``bisection_line_search``, retargeted to perturb ``E``
    instead of ``model.A``.

    Phase 1 (bracket): shrink alpha geometrically from ``alpha_max`` until a
    trial crosses the margin. Phase 2 (bisect): binary search inside that
    bracket to converge toward the minimal crossing point.

    Returns ``(accepted_alpha, new_E, new_target_strength, rival_class,
    rival_strength)`` for the smallest known-crossing alpha found, or the
    smallest-alpha trial if none crossed within ``max_backtracks``, or
    ``None`` if ``max_backtracks == 0``.
    """
    assert model.A is not None

    def trial(alpha: float) -> tuple[Tensor, float, int | None, float]:
        trial_E = _perturb_new_case_edges(E, edge_indices, direction, alpha)
        trial_strengths = _forward_new_case_strengths(
            model, trial_E.unsqueeze(0), casebase_base_scores, new_cases_base_scores
        )
        trial_target, rival_class, trial_rival = _target_and_rival(
            trial_strengths, target_class, threshold
        )
        return trial_E, trial_target, rival_class, trial_rival

    def crossed(target: float, rival: float) -> bool:
        return target - rival >= margin

    # ---- Phase 1: bracket ----
    alpha_lo, alpha_hi = 0.0, alpha_max
    best: tuple[float, Tensor, float, int | None, float] | None = None
    hi_result: tuple[Tensor, float, int | None, float] | None = None

    alpha = alpha_max
    for _ in range(max_backtracks):
        trial_E, trial_target, rival_class, trial_rival = trial(alpha)
        if crossed(trial_target, trial_rival):
            alpha_hi, hi_result = alpha, (trial_E, trial_target, rival_class, trial_rival)
            break
        best = (alpha, trial_E, trial_target, rival_class, trial_rival)
        alpha_lo = alpha
        alpha *= factor
    else:
        return best  # never crossed within budget

    # ---- Phase 2: bisect within [alpha_lo, alpha_hi] ----
    for _ in range(max_bisections):
        if alpha_hi - alpha_lo < bisect_tol:
            break
        alpha_mid = (alpha_lo + alpha_hi) / 2
        trial_E, trial_target, rival_class, trial_rival = trial(alpha_mid)
        if crossed(trial_target, trial_rival):
            alpha_hi, hi_result = alpha_mid, (trial_E, trial_target, rival_class, trial_rival)
        else:
            alpha_lo = alpha_mid

    trial_E, trial_target, rival_class, trial_rival = hi_result
    return alpha_hi, trial_E, trial_target, rival_class, trial_rival


def new_case_contest(
    model: GradualAACBR,
    sample: Tensor,
    target_class: int,
    k: int = DEFAULT_K,
    threshold: float = THRESHOLD,
    margin: float = MARGIN,
    max_iters: int = MAX_ITERS,
    max_edits: int | None = None,
) -> NewCaseContestResult:
    """Contest a single sample's prediction towards ``target_class`` by
    editing only its own edges into the casebase (``E``), never
    ``model.A`` and never any live model state.

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
        See ``contest.py``'s module-level defaults (reused here).
    max_edits : int | None
        Stop once this many distinct entries of ``E`` (flat indices into
        its (n*d,) layout) have been touched -- same semantics as
        ``contest()``'s ``max_edits``. ``None`` (default) is unbounded.

    Returns
    -------
    NewCaseContestResult
    """
    if model.A is None:
        raise Exception("Ensure the model has been fit first.")
    if sample.shape[0] != 1:
        raise ValueError(
            "new_case_contest expects a single new case (batch size 1), got "
            f"batch of {sample.shape[0]}."
        )

    with torch.no_grad():
        model(sample)
    initial_E = model.new_cases_attacks_adjacency.detach().clone()[0]
    casebase_base_scores = _batched_casebase_base_scores(model, batch_size=1)
    new_cases_base_scores = model.new_cases_base_scores.detach()

    E = initial_E
    max_delta = 0.0
    trace: list[NewCaseEdgeTraceStep] = []
    touched_edges: set[int] = set()

    strengths = _forward_new_case_strengths(
        model, E.unsqueeze(0), casebase_base_scores, new_cases_base_scores
    )
    target_strength, rival_class, rival_strength = _target_and_rival(
        strengths, target_class, threshold
    )
    iters_run = 0

    for iters_run in range(1, max_iters + 1):
        if target_strength - rival_strength >= margin:
            return NewCaseContestResult(
                True, iters_run - 1, max_delta, trace,
                target_strength, rival_class, rival_strength, initial_E, E,
            )

        grae_vector = _new_case_grae(
            model, E, target_class, casebase_base_scores, new_cases_base_scores
        )

        if grae_vector.abs().max().item() <= LIVE_GRAD_THRESHOLD:
            break  # dead gradient -- E-only edits can't move this sample

        edge_indices = select_top_k(grae_vector, k)
        direction = grae_vector[edge_indices]
        step = bisection_line_search(
            model, target_class, E, edge_indices, direction,
            casebase_base_scores, new_cases_base_scores,
            margin=margin, threshold=threshold,
        )

        if step is None:
            break  # plateaued -- no accepted step within budget

        alpha, new_E, new_target_strength, new_rival_class, new_rival_strength = step
        if torch.equal(new_E, E):
            # Direction points straight into the E_MIN/E_MAX clamp boundary
            # (e.g. weakening an already-zero attack further) -- every trial
            # collapses to a no-op, so further iterations would just repeat
            # this same stalemate.
            break

        old_values = E.view(-1)[edge_indices]
        new_values = new_E.view(-1)[edge_indices]
        delta = (new_values - old_values).abs().max().item()
        max_delta = max(max_delta, delta)
        trace.append(
            NewCaseEdgeTraceStep(
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

        E = new_E
        target_strength, rival_class, rival_strength = (
            new_target_strength, new_rival_class, new_rival_strength,
        )

        touched_edges.update(edge_indices.tolist())
        if max_edits is not None and len(touched_edges) >= max_edits:
            break  # edit budget hit -- stop even if the margin isn't met yet

    if target_strength - rival_strength >= margin:
        return NewCaseContestResult(
            True, iters_run - 1, max_delta, trace,
            target_strength, rival_class, rival_strength, initial_E, E,
        )
    return NewCaseContestResult(
        False, iters_run, max_delta, trace,
        target_strength, rival_class, rival_strength, initial_E, E,
    )
