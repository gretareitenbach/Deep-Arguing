import pytest
import torch

from deeparguing import GradualAACBR
from deeparguing.counterfactuals.contest import (
    MARGIN,
    THRESHOLD,
    ContestResult,
    bisection_line_search,
    contest,
    select_top_k,
)
from deeparguing.counterfactuals.grae import compute_grae
from deeparguing.semantics.sigmoid_semantics import SigmoidSemantics
from qbaf_fixtures import (
    TARGET_INDEX,
    base_score_fn as _base_score_fn,
    edge_weights_fn as _edge_weights_fn,
    irrelevance_fn as _irrelevance_fn,
    make_fitted_model as _make_qbaf_model,
)

# ---------------------------------------------------------------------------
# Shared small synthetic EW-QBAF -- see ``tests/qbaf_fixtures.py`` (5
# casebase arguments, 1 default argument, 2 new cases) so the
# strengths/G-RAEs behind ``contest``'s decisions are already known-good.
# ---------------------------------------------------------------------------


def _make_fitted_model(max_iters: int) -> GradualAACBR:
    return _make_qbaf_model(SigmoidSemantics(max_iters=max_iters, epsilon=0))


# ---------------------------------------------------------------------------
# select_top_k
# ---------------------------------------------------------------------------


def test_select_top_k_orders_by_absolute_magnitude():
    grae_vector = torch.tensor([1.0, -5.0, 2.0, 0.5])
    indices = select_top_k(grae_vector, 2)
    assert indices.tolist() == [1, 2]


def test_select_top_k_respects_k():
    grae_vector = torch.tensor([1.0, -5.0, 2.0, 0.5])
    indices = select_top_k(grae_vector, 1)
    assert indices.tolist() == [1]


# ---------------------------------------------------------------------------
# bisection_line_search
# ---------------------------------------------------------------------------


def _top_k_direction(model, new_case, target_class, k=2):
    grae = compute_grae(model, new_case, [target_class])
    grae_vector = grae.casebase_edges.reshape(-1)
    edge_indices = select_top_k(grae_vector, k)
    return edge_indices, grae_vector[edge_indices]


def test_bisection_line_search_finds_a_crossing_alpha():
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)
    edge_indices, direction = _top_k_direction(model, new_case, TARGET_INDEX)

    step = bisection_line_search(
        model, new_case, TARGET_INDEX, edge_indices, direction,
        threshold=THRESHOLD, margin=MARGIN,
    )

    assert step is not None
    alpha, new_A, new_target_strength, rival_class, rival_strength = step
    # single-default-argument casebase -> no real rival, threshold is the
    # fixed virtual competitor
    assert rival_class is None
    assert rival_strength == THRESHOLD
    assert new_target_strength >= THRESHOLD + MARGIN
    assert new_A.shape == model.A.shape


def test_bisection_line_search_does_not_mutate_model_A():
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)
    edge_indices, direction = _top_k_direction(model, new_case, TARGET_INDEX)
    original_A = model.A.detach().clone()

    bisection_line_search(
        model, new_case, TARGET_INDEX, edge_indices, direction,
        threshold=THRESHOLD, margin=MARGIN,
    )

    assert torch.equal(model.A, original_A)


def test_bisection_line_search_returns_none_when_max_backtracks_is_zero():
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)
    edge_indices, direction = _top_k_direction(model, new_case, TARGET_INDEX)

    step = bisection_line_search(
        model, new_case, TARGET_INDEX, edge_indices, direction,
        threshold=THRESHOLD, margin=MARGIN, max_backtracks=0,
    )

    assert step is None


def test_bisection_line_search_refines_below_alpha_max_when_it_overshoots():
    """If alpha_max itself crosses the margin on the first trial, bisection
    should keep narrowing instead of accepting alpha_max outright -- unlike
    plain backtracking, which would return alpha_max unrefined."""
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)
    edge_indices, direction = _top_k_direction(model, new_case, TARGET_INDEX)

    # A very generous margin/threshold gap forces alpha_max to overshoot
    # substantially, giving bisection real room to refine downward.
    step = bisection_line_search(
        model, new_case, TARGET_INDEX, edge_indices, direction,
        threshold=0.0, margin=0.01, alpha_max=1.0,
    )

    assert step is not None
    alpha, new_A, new_target_strength, rival_class, rival_strength = step
    assert new_target_strength - rival_strength >= 0.01
    assert alpha <= 1.0


# ---------------------------------------------------------------------------
# contest
# ---------------------------------------------------------------------------


def test_contest_finds_a_crossing_step_and_commits_it_to_model_A():
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)

    before = model(new_case)[0, TARGET_INDEX].item()
    assert before < THRESHOLD + MARGIN  # sanity: starts below the boundary

    result = contest(model, new_case, TARGET_INDEX, k=2, max_iters=10)

    assert isinstance(result, ContestResult)
    assert result.success
    assert result.final_target_strength is not None
    assert result.final_target_strength >= THRESHOLD + MARGIN
    # single-default-argument casebase -> no real rival, threshold stands in
    assert result.final_rival_class is None
    assert result.final_rival_strength == THRESHOLD
    assert result.iterations >= 1
    assert result.max_weight_delta > 0
    assert len(result.edge_trace) == result.iterations

    # the accepted step must persist on the model, not just a trial copy
    after = model(new_case)[0, TARGET_INDEX].item()
    assert after == pytest.approx(result.final_target_strength)


def test_contest_only_changes_the_traced_edges_of_model_A():
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)
    original_A = model.A.detach().clone()

    result = contest(model, new_case, TARGET_INDEX, k=2, max_iters=10)

    diff = (model.A.reshape(-1) - original_A.reshape(-1)).abs() > 0
    changed_indices = set(diff.nonzero().flatten().tolist())

    traced_indices = set()
    for step in result.edge_trace:
        traced_indices.update(step.edge_ids)

    assert changed_indices == traced_indices


def test_contest_reports_failure_when_threshold_is_unreachable_in_time():
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)

    result = contest(
        model, new_case, TARGET_INDEX, k=2, threshold=0.999, margin=0.0, max_iters=3
    )

    assert not result.success
    assert result.iterations == 3
    assert result.final_target_strength is not None
    assert result.final_target_strength < 0.999


def test_contest_raises_if_model_not_fitted():
    model = GradualAACBR(
        SigmoidSemantics(max_iters=1, epsilon=0),
        _base_score_fn,
        _irrelevance_fn,
        _edge_weights_fn,
    )
    with pytest.raises(Exception, match="fit"):
        contest(model, torch.tensor([[6.0]]), TARGET_INDEX)


def test_contest_raises_on_batch_size_other_than_one():
    model = _make_fitted_model(max_iters=3)
    new_cases = torch.tensor([[6.0], [7.0]])
    with pytest.raises(ValueError, match="batch"):
        contest(model, new_cases, TARGET_INDEX)
