"""Single-sample contestability search restricted to a new case's irrelevance edges
into the casebase, instead of contest.py's edits to the shared model
adjacency. (Same bracket-and-bisect line search as contest.py)
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR

from .contest import (ALPHA_MAX, BACKTRACK_FACTOR, BISECT_TOL, DEFAULT_K,
                       LIVE_GRAD_THRESHOLD, MARGIN, MAX_BACKTRACKS,
                       MAX_BISECTIONS, MAX_ITERS, THRESHOLD, _bisection_search,
                       _target_and_rival, select_top_k)
from .grae import _batched_casebase_base_scores, _replay_default_strengths

E_MIN = -1.0
E_MAX = 0.0


class NewCaseEdgeTraceStep(NamedTuple):

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
    """Outcome of new_case_contest(): whether the target class flipped, how
    many iterations it took, the largest single E-entry change made, the
    full step-by-step trace, the final target/rival strengths, and both the
    starting and final E.
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
    """Copy of E (a single sample's (n, d) row into the casebase) with the
    entries at edge_indices shifted by alpha * direction and clamped to
    [E_MIN, E_MAX]; all other entries are unchanged.
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
    """Gradient of target_class's strength w.r.t. every entry of E, evaluated
    at the given E (which may already be a search-perturbed value, not the
    network's live output). model.A is detached first since only the
    gradient w.r.t. E is wanted. Returns a flattened (n*d,) tensor matching
    select_top_k's index convention.
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
    """Trial forward pass with model.A fixed and the sample's own new-case
    row set to E, without touching any live model state. Returns every
    default class's strength, shape (D,).
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
    """Two-phase search for close to the smallest alpha (along direction)
    that makes target_class beat the best rival class by at least margin.
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

    return _bisection_search(
        trial, margin, alpha_max, factor, max_backtracks, max_bisections, bisect_tol
    )


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
    """Contest a single sample's prediction towards target_class by editing
    only its own edges into the casebase (E).
    max_edits stops the search once that many distinct
    entries of E have been touched (None is unbounded).
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
            break  # dead gradient

        edge_indices = select_top_k(grae_vector, k)
        direction = grae_vector[edge_indices]
        step = bisection_line_search(
            model, target_class, E, edge_indices, direction,
            casebase_base_scores, new_cases_base_scores,
            margin=margin, threshold=threshold,
        )

        if step is None:
            break  # plateaued

        alpha, new_E, new_target_strength, new_rival_class, new_rival_strength = step
        if torch.equal(new_E, E):
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
            break  # edit budget hit

    if target_strength - rival_strength >= margin:
        return NewCaseContestResult(
            True, iters_run - 1, max_delta, trace,
            target_strength, rival_class, rival_strength, initial_E, E,
        )
    return NewCaseContestResult(
        False, iters_run, max_delta, trace,
        target_strength, rival_class, rival_strength, initial_E, E,
    )
