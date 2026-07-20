import numpy as np
import pytest
import torch

from deeparguing import GradualAACBR
from deeparguing.evals.global_contest_eval import (
    compute_baseline_metrics,
    evaluate_contested_model,
)
from deeparguing.semantics.relu_semantics import ReluSemantics

# ---------------------------------------------------------------------------
# Synthetic EW-QBAF fixture, extending the one shared by
# ``tests/grae_test.py``/``tests/contest_test.py``/``tests/batch_contest_test.py``:
# same 5-argument casebase (indices 0-4; 0/2/4 are class 0, 1/3 are class 1),
# but with *two* default/topic arguments (indices 5 and 6, one per class --
# same pattern as ``batch_contest_test.py``'s ``test_rival_indices_*`` --
# instead of the single-default setup) so that argmax over the two default
# strengths gives an actual 2-class prediction to compute global test-set
# metrics over. Two more nodes (7, 8) act as the "new case" test set: 7
# attacks every class-0 casebase argument (pushing the classification
# towards class 1), 8 attacks every class-1 casebase argument (pushing
# towards class 0). Verified empirically (see PR description / commit) that
# with ``ReluSemantics`` this fixture classifies 7 -> class 1 and 8 -> class
# 0, both correctly, at baseline -- giving a non-trivial (100%, not
# vacuous-1-class) starting point for the delta tests below.
# ---------------------------------------------------------------------------

_N = 9
_EDGE_WEIGHTS = torch.zeros((_N, _N), dtype=torch.float32)
# casebase -> default 5 (class 0): same-label cases support, diff-label attack
_EDGE_WEIGHTS[0, 5] = 1
_EDGE_WEIGHTS[2, 5] = 1
_EDGE_WEIGHTS[4, 5] = 1
_EDGE_WEIGHTS[1, 5] = 1
_EDGE_WEIGHTS[3, 5] = 1
# casebase -> default 6 (class 1): same-label cases support, diff-label attack
_EDGE_WEIGHTS[1, 6] = 1
_EDGE_WEIGHTS[3, 6] = 1
_EDGE_WEIGHTS[0, 6] = 1
_EDGE_WEIGHTS[2, 6] = 1
_EDGE_WEIGHTS[4, 6] = 1
# new/test case 7: attacks every class-0 casebase argument
_EDGE_WEIGHTS[7, 0] = 1
_EDGE_WEIGHTS[7, 2] = 1
_EDGE_WEIGHTS[7, 4] = 1
# new/test case 8: attacks every class-1 casebase argument
_EDGE_WEIGHTS[8, 1] = 1
_EDGE_WEIGHTS[8, 3] = 1

_BASE_SCORES = torch.tensor([0.5, 0.5, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5, 0.5])


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


# Full "test set": case 7 is truly class 1, case 8 is truly class 0 -- both
# classified correctly by the unedited model (see fixture docstring above).
X_TEST = torch.tensor([[7], [8]], dtype=torch.float32)
Y_TEST = torch.tensor([[0.0, 1.0], [1.0, 0.0]])


# ---------------------------------------------------------------------------
# (1) zero-edit model.A reproduces baseline exactly
# ---------------------------------------------------------------------------


def test_zero_edit_reproduces_baseline_exactly():
    model = _make_fitted_model()
    baseline = compute_baseline_metrics(model, X_TEST, Y_TEST)

    # Sanity: baseline is non-trivial (not vacuously 0% or driven by a
    # degenerate single-class prediction).
    assert baseline.accuracy == pytest.approx(1.0)

    zero_edit_A = model.A.detach().clone()
    result = evaluate_contested_model(model, zero_edit_A, X_TEST, Y_TEST, baseline)

    assert result.metrics.accuracy == pytest.approx(baseline.accuracy)
    assert result.metrics.precision == pytest.approx(baseline.precision)
    assert result.metrics.recall == pytest.approx(baseline.recall)
    assert result.metrics.f1 == pytest.approx(baseline.f1)
    assert np.array_equal(result.metrics.confusion_matrix, baseline.confusion_matrix)

    assert result.delta_accuracy == pytest.approx(0.0)
    assert result.delta_precision == pytest.approx(0.0)
    assert result.delta_recall == pytest.approx(0.0)
    assert result.delta_f1 == pytest.approx(0.0)


def test_evaluate_contested_model_restores_model_A():
    model = _make_fitted_model()
    baseline = compute_baseline_metrics(model, X_TEST, Y_TEST)
    original_A = model.A.detach().clone()

    contested_A = -model.A.detach().clone()
    evaluate_contested_model(model, contested_A, X_TEST, Y_TEST, baseline)

    assert torch.equal(model.A, original_A)


# ---------------------------------------------------------------------------
# (2) a hand-crafted edit moves accuracy in the predicted direction
# ---------------------------------------------------------------------------


def test_hand_crafted_edit_moves_accuracy_in_predicted_direction():
    """Negating every entry of ``model.A`` swaps the role of every
    attack/support edge (class-0-supporting edges become class-0-attacking
    and vice versa), which should flip both test cases' predicted class and
    therefore turn a 100%-accuracy baseline into a 0%-accuracy contested
    model -- a strictly negative, fully predictable accuracy delta."""
    model = _make_fitted_model()
    baseline = compute_baseline_metrics(model, X_TEST, Y_TEST)
    assert baseline.accuracy == pytest.approx(1.0)

    contested_A = -model.A.detach().clone()
    result = evaluate_contested_model(model, contested_A, X_TEST, Y_TEST, baseline)

    assert result.metrics.accuracy == pytest.approx(0.0)
    assert result.delta_accuracy == pytest.approx(-1.0)
    assert result.delta_accuracy < 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_compute_baseline_metrics_raises_if_model_not_fitted():
    model = GradualAACBR(
        ReluSemantics(max_iters=1, epsilon=0),
        _base_score_fn,
        _irrelevance_fn,
        _edge_weights_fn,
    )
    with pytest.raises(Exception, match="fit"):
        compute_baseline_metrics(model, X_TEST, Y_TEST)


def test_evaluate_contested_model_raises_if_model_not_fitted():
    model = GradualAACBR(
        ReluSemantics(max_iters=1, epsilon=0),
        _base_score_fn,
        _irrelevance_fn,
        _edge_weights_fn,
    )
    dummy_A = torch.zeros((1, 1, 1))
    with pytest.raises(Exception, match="fit"):
        evaluate_contested_model(model, dummy_A, X_TEST, Y_TEST, baseline=None)
