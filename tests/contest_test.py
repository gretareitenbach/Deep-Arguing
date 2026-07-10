import pytest
import torch

from deeparguing import GradualAACBR
from deeparguing.counterfactuals.contest import (
    MARGIN,
    THRESHOLD,
    ContestResult,
    backtracking_line_search,
    contest,
    select_top_k,
)
from deeparguing.counterfactuals.grae import compute_grae
from deeparguing.semantics.sigmoid_semantics import SigmoidSemantics

# ---------------------------------------------------------------------------
# Same small synthetic EW-QBAF as ``tests/grae_test.py`` (5 casebase
# arguments, 1 default argument, 2 new cases) so the strengths/G-RAEs behind
# ``contest``'s decisions are already known-good.
# ---------------------------------------------------------------------------

_EDGE_WEIGHTS = torch.tensor(
    [
        # 0, 1, 2, 3, 4, 5, 6, 7
        [0, 1, 0, 0, 0, 0, 0, 0],  # 0
        [1, 0, 0, 0, 0, 0, 0, 0],  # 1
        [1, 0, 0, 1, 0, 0, 0, 0],  # 2
        [0, 1, 0, 0, 0, 0, 0, 0],  # 3
        [1, 0, 0, 1, 0, 0, 0, 0],  # 4
        [0, 0, 0, 0, 0, 0, 0, 0],  # 5
        [0, 0, 0, 0, 0, 0, 0, 0],  # 6
        [0, 0, 0, 0, 1, 0, 0, 0],  # 7
    ],
    dtype=torch.float32,
)

_BASE_SCORES = torch.tensor([0.5, 0.5, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5])


def _base_score_fn(case: torch.Tensor) -> torch.Tensor:
    case = case.to(dtype=torch.int).squeeze(-1)
    return _BASE_SCORES[case].unsqueeze(-1)


def _edge_weights_fn(attacker: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    attacker = attacker.to(dtype=torch.int).squeeze(-1)
    target = target.to(dtype=torch.int).squeeze(-1)
    return _EDGE_WEIGHTS[attacker.unsqueeze(1), target.unsqueeze(0)]


def _irrelevance_fn(new_cases: torch.Tensor, casebase: torch.Tensor) -> torch.Tensor:
    new_cases = new_cases.unsqueeze(1).to(dtype=torch.int)
    casebase = casebase.unsqueeze(0).to(dtype=torch.int)
    r = _EDGE_WEIGHTS[new_cases, casebase].squeeze(-1)
    return r.unsqueeze(-1)


def _make_fitted_model(max_iters: int) -> GradualAACBR:
    semantics = SigmoidSemantics(max_iters=max_iters, epsilon=0)
    model = GradualAACBR(
        semantics,
        _base_score_fn,
        _irrelevance_fn,
        _edge_weights_fn,
    )
    model.use_symmetric_attacks = True
    model.use_supports = True
    model.use_blockers = False

    X_train = torch.tensor([[0], [1], [2], [3], [4]])
    y_train = torch.tensor([[0], [1], [0], [1], [0]])
    X_default = torch.tensor([[5]], dtype=torch.float32)
    y_default = torch.tensor([[5]], dtype=torch.float32)
    model.fit(X_train, y_train, X_default, y_default)
    return model


# Only one default argument exists in this casebase, so it is always the
# (only) topic/target argument.
TARGET_INDEX = 0


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
# backtracking_line_search
# ---------------------------------------------------------------------------


def _top_k_direction(model, new_case, target_class, k=2):
    grae = compute_grae(model, new_case, [target_class])
    grae_vector = grae.casebase_edges.reshape(-1)
    edge_indices = select_top_k(grae_vector, k)
    return edge_indices, grae_vector[edge_indices]


def test_backtracking_line_search_finds_a_crossing_alpha():
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)
    edge_indices, direction = _top_k_direction(model, new_case, TARGET_INDEX)

    step = backtracking_line_search(
        model, new_case, TARGET_INDEX, edge_indices, direction,
        threshold=THRESHOLD, margin=MARGIN,
    )

    assert step is not None
    alpha, new_A, new_strength = step
    assert new_strength >= THRESHOLD + MARGIN
    assert new_A.shape == model.A.shape


def test_backtracking_line_search_does_not_mutate_model_A():
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)
    edge_indices, direction = _top_k_direction(model, new_case, TARGET_INDEX)
    original_A = model.A.detach().clone()

    backtracking_line_search(
        model, new_case, TARGET_INDEX, edge_indices, direction,
        threshold=THRESHOLD, margin=MARGIN,
    )

    assert torch.equal(model.A, original_A)


def test_backtracking_line_search_returns_none_when_max_backtracks_is_zero():
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)
    edge_indices, direction = _top_k_direction(model, new_case, TARGET_INDEX)

    step = backtracking_line_search(
        model, new_case, TARGET_INDEX, edge_indices, direction,
        threshold=THRESHOLD, margin=MARGIN, max_backtracks=0,
    )

    assert step is None


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
    assert result.final_strength is not None
    assert result.final_strength >= THRESHOLD + MARGIN
    assert result.iterations >= 1
    assert result.max_weight_delta > 0
    assert len(result.edge_trace) == result.iterations

    # the accepted step must persist on the model, not just a trial copy
    after = model(new_case)[0, TARGET_INDEX].item()
    assert after == pytest.approx(result.final_strength)


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
    assert result.final_strength is not None
    assert result.final_strength < 0.999


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
