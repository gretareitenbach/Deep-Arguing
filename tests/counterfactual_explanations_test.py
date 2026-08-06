import pytest
import torch

from deeparguing.contest.scripts.counterfactual_explanations import (
    _case_label,
    _decode_edge,
    explain_sample,
)
from deeparguing.semantics.sigmoid_semantics import SigmoidSemantics
from qbaf_fixtures import TARGET_INDEX, make_fitted_model as _make_qbaf_model

# ---------------------------------------------------------------------------
# Shared small synthetic EW-QBAF -- see tests/qbaf_fixtures.py.
# ---------------------------------------------------------------------------


def _make_fitted_model(max_iters: int):
    return _make_qbaf_model(SigmoidSemantics(max_iters=max_iters, epsilon=0))


# ---------------------------------------------------------------------------
# _decode_edge
# ---------------------------------------------------------------------------


def test_decode_edge_unravels_flat_index_to_source_target_head():
    # n2=3 cases, d=2 heads: edge_id = ((source * n2) + target) * d + head
    assert _decode_edge(edge_id=0, n2=3, d=2) == (0, 0, 0)
    assert _decode_edge(edge_id=1, n2=3, d=2) == (0, 0, 1)
    assert _decode_edge(edge_id=8, n2=3, d=2) == (1, 1, 0)


# ---------------------------------------------------------------------------
# _case_label
# ---------------------------------------------------------------------------


def test_case_label_uses_item_for_single_column_labels():
    # qbaf_fixtures' y_train is a single scalar column (Y==1), not one-hot --
    # case 1's row is literally [1.0] (see EDGE_WEIGHTS/y_train comments).
    model = _make_fitted_model(max_iters=1)
    assert _case_label(model, default_index_set=set(), case_index=1) == "case #1 (label 1)"


def test_case_label_flags_default_cases():
    model = _make_fitted_model(max_iters=1)
    default_index_set = set(model.default_indexes.tolist())
    label = _case_label(model, default_index_set, case_index=next(iter(default_index_set)))
    assert ", default" in label


# ---------------------------------------------------------------------------
# explain_sample
# ---------------------------------------------------------------------------


def test_explain_sample_finds_a_counterfactual_and_restores_model_A():
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)
    original_A = model.A.detach().clone()

    explanation = explain_sample(
        model, new_case, sample_index=0, true_class=1, target_class=TARGET_INDEX,
        k=2, max_iters=10,
    )

    assert explanation.success
    assert explanation.edges  # at least one edge reported
    assert torch.equal(model.A, original_A)  # undone -- no persisted side effect


def test_explain_sample_edges_report_original_and_final_weights():
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)
    original_flat = model.A.detach().clone().reshape(-1)

    explanation = explain_sample(
        model, new_case, sample_index=0, true_class=1, target_class=TARGET_INDEX,
        k=2, max_iters=10,
    )

    for edge in explanation.edges:
        assert edge.old_weight == pytest.approx(original_flat[edge.edge_id].item())
        assert edge.new_weight != edge.old_weight  # every reported edge actually moved
        assert edge.delta == pytest.approx(edge.new_weight - edge.old_weight)


def test_explain_sample_reports_failure_without_mutating_model_A():
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)
    original_A = model.A.detach().clone()

    explanation = explain_sample(
        model, new_case, sample_index=0, true_class=1, target_class=TARGET_INDEX,
        k=2, threshold=0.999, margin=0.0, max_iters=3,
    )

    assert not explanation.success
    assert torch.equal(model.A, original_A)


def test_explain_sample_touches_only_unique_edges_despite_repeated_visits():
    """If contest() revisits the same edge across iterations, explain_sample
    must report it once, with the true original->final net change -- not one
    row per visit."""
    model = _make_fitted_model(max_iters=5)
    new_case = torch.tensor([[6]], dtype=torch.float32)

    explanation = explain_sample(
        model, new_case, sample_index=0, true_class=1, target_class=TARGET_INDEX,
        k=2, max_iters=10,
    )

    edge_ids = [e.edge_id for e in explanation.edges]
    assert len(edge_ids) == len(set(edge_ids))
