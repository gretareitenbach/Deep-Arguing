import pytest
import torch

from deeparguing import GradualAACBR
from deeparguing.contest.core.batch_contest import batch_contest
from deeparguing.contest.global_optimize import global_optimize
from deeparguing.semantics.relu_semantics import ReluSemantics

_N = 14
_EDGE_WEIGHTS = torch.zeros((_N, _N), dtype=torch.float32)
_EDGE_WEIGHTS[0, 5] = 1
_EDGE_WEIGHTS[2, 5] = 1
_EDGE_WEIGHTS[4, 5] = 1
_EDGE_WEIGHTS[1, 5] = 1
_EDGE_WEIGHTS[3, 5] = 1
_EDGE_WEIGHTS[1, 6] = 1
_EDGE_WEIGHTS[3, 6] = 1
_EDGE_WEIGHTS[0, 6] = 1
_EDGE_WEIGHTS[2, 6] = 1
_EDGE_WEIGHTS[4, 6] = 1
for _i in (7, 9, 11):
    _EDGE_WEIGHTS[_i, 0] = 1
    _EDGE_WEIGHTS[_i, 2] = 1
    _EDGE_WEIGHTS[_i, 4] = 1
for _i in (8, 10, 12):
    _EDGE_WEIGHTS[_i, 1] = 1
    _EDGE_WEIGHTS[_i, 3] = 1
_FLIP_INDEX = 13
_EDGE_WEIGHTS[_FLIP_INDEX, 1] = 1
_EDGE_WEIGHTS[_FLIP_INDEX, 3] = 1

_BASE_SCORES = torch.full((_N,), 0.5, dtype=torch.float32)
_BASE_SCORES[2] = 0.7
_BASE_SCORES[3] = 0.8
_BASE_SCORES[4] = 0.9


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


def _make_fitted_model(max_iters: int = 5) -> GradualAACBR:
    semantics = ReluSemantics(max_iters=max_iters, epsilon=0)
    model = GradualAACBR(semantics, _base_score_fn, _irrelevance_fn, _edge_weights_fn)
    model.use_symmetric_attacks = True
    model.use_supports = True
    model.use_blockers = False

    X_train = torch.tensor([[0], [1], [2], [3], [4]])
    y_train = torch.tensor([[0], [1], [0], [1], [0]])
    X_default = torch.tensor([[5], [6]], dtype=torch.float32)
    y_default = torch.tensor([[0], [1]], dtype=torch.float32)
    model.fit(X_train, y_train, X_default, y_default)
    model.eval()
    return model


X_EVAL = torch.tensor([[7], [8], [9], [10], [11], [12]], dtype=torch.float32)
Y_EVAL = torch.tensor(
    [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
)

FLIP_SAMPLE = torch.tensor([[13]], dtype=torch.float32)
FLIP_TRUE_CLASSES = [1]

_K = 3
_COMMON_KWARGS = dict(k=_K, margin=0.01, divergence_bound=100.0)


def _run(protect_lambda: float, max_acc_drop: float, eval_every: int, max_iters: int = 20, **kwargs):
    torch.manual_seed(0)
    model = _make_fitted_model()
    result = global_optimize(
        model, FLIP_SAMPLE, FLIP_TRUE_CLASSES, X_EVAL, Y_EVAL,
        max_iters=max_iters, protect_lambda=protect_lambda,
        max_acc_drop=max_acc_drop, eval_every=eval_every,
        **_COMMON_KWARGS, **kwargs,
    )
    return model, result


def test_global_optimize_rollback_restores_last_good_A_when_acc_drop_exceeds_budget():
    model, result = _run(protect_lambda=0.0, max_acc_drop=0.0, eval_every=1)

    assert result.rolled_back
    assert result.stopped_reason == "acc_drop"
    assert len(result.rounds) == 2
    assert result.rounds[0]["rolled_back"] is False
    assert result.rounds[1]["rolled_back"] is True

    assert result.batch_result.num_cleared == 0
    assert not bool(result.batch_result.cleared[0])
    assert result.acc_drop == pytest.approx(0.0)
    assert result.final_metrics.accuracy == pytest.approx(1.0)

    torch.manual_seed(0)
    reference_model = _make_fitted_model()
    reference_result = batch_contest(
        reference_model, FLIP_SAMPLE, FLIP_TRUE_CLASSES, max_iters=1, **_COMMON_KWARGS
    )
    assert torch.allclose(model.A, reference_model.A)
    assert result.batch_result.touched_edge_indices == reference_result.touched_edge_indices


def test_global_optimize_converged_stops_early_without_using_full_max_iters():
    model, result = _run(protect_lambda=0.0, max_acc_drop=1.0, eval_every=5, max_iters=20)

    assert result.stopped_reason == "converged"
    assert not result.rolled_back
    assert len(result.rounds) == 1
    assert result.rounds[0]["iterations"] == 3
    assert bool(result.batch_result.cleared[0])


def test_global_optimize_eval_every_cadence():
    model, result = _run(protect_lambda=0.0, max_acc_drop=1.0, eval_every=2, max_iters=20)

    assert result.stopped_reason == "converged"
    assert len(result.rounds) == 2
    assert sum(r["iterations"] for r in result.rounds) == 3


def test_global_optimize_matches_single_batch_contest_call_when_guardrail_never_trips():
    model, result = _run(protect_lambda=0.0, max_acc_drop=1.0, eval_every=1, max_iters=20)

    torch.manual_seed(0)
    reference_model = _make_fitted_model()
    reference_result = batch_contest(
        reference_model, FLIP_SAMPLE, FLIP_TRUE_CLASSES, max_iters=20, **_COMMON_KWARGS
    )

    assert not result.rolled_back
    assert torch.allclose(model.A, reference_model.A)
    assert result.batch_result.touched_edge_indices == reference_result.touched_edge_indices
    assert result.batch_result.num_cleared == reference_result.num_cleared


def test_global_optimize_global_edit_budget_enforced_across_rounds():
    model, result = _run(
        protect_lambda=0.0, max_acc_drop=1.0, eval_every=1, max_iters=20, max_edits=3
    )

    assert result.stopped_reason == "max_edits"
    assert len(result.rounds) == 1
    assert result.batch_result.num_edges_changed == 3
    assert len(result.batch_result.touched_edge_indices) <= 3
    assert result.batch_result.num_cleared == 0


def test_global_optimize_result_fields_and_round_schema():
    model, result = _run(protect_lambda=0.0, max_acc_drop=1.0, eval_every=5)

    assert result.baseline.accuracy == pytest.approx(1.0)
    assert isinstance(result.final_metrics.accuracy, float)
    assert isinstance(result.acc_drop, float)
    assert isinstance(result.rolled_back, bool)
    assert result.stopped_reason in {"converged", "acc_drop", "max_iters", "max_edits"}
    assert result.protect_sample_size >= 0
    assert len(result.rounds) >= 1
    for round_entry in result.rounds:
        assert set(round_entry.keys()) == {
            "iterations", "num_cleared", "global_acc", "acc_drop", "rolled_back",
        }


def test_global_optimize_raises_if_model_not_fitted():
    model = GradualAACBR(
        ReluSemantics(max_iters=1, epsilon=0), _base_score_fn, _irrelevance_fn, _edge_weights_fn
    )
    with pytest.raises(Exception, match="fit"):
        global_optimize(model, FLIP_SAMPLE, FLIP_TRUE_CLASSES, X_EVAL, Y_EVAL)
