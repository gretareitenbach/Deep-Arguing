import numpy as np
import pytest
import torch

from deeparguing import GradualAACBR
from deeparguing.evals import evaluate_model
from deeparguing.evals.compute_graph import make_dot
from deeparguing.semantics.sigmoid_semantics import SigmoidSemantics

mock_gradual_semantics = SigmoidSemantics(max_iters=0, epsilon=0)
mock_casebase_edge_weights = lambda a, b: torch.where(
    torch.all(a.unsqueeze(1) >= b.unsqueeze(0), dim=-1),
    1.0,
    0.0,
).unsqueeze(-1)
mock_irrelevance_edge_weights = lambda a, b: 1 - mock_casebase_edge_weights(a, b)


class LearnableConstantBaseScore(torch.nn.Module):
    """A base score with one learnable parameter, so that forward() output
    stays attached to a real autograd graph for the compute-graph tests."""

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, nodes):
        return torch.full((nodes.shape[0], 1), 1.0) * self.bias


def make_model():
    return GradualAACBR(
        mock_gradual_semantics,
        LearnableConstantBaseScore(),
        mock_irrelevance_edge_weights,
        mock_casebase_edge_weights,
    )


X_casebase = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
y_casebase = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
X_default = torch.zeros((2, 4))
y_default = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
X_new_cases = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
y_new_cases = torch.tensor([[1.0, 0.0], [0.0, 1.0]])


def test_evaluate_model_returns_metrics():
    model = make_model()
    accuracy, precision, recall, f1, cm = evaluate_model(
        model, X_casebase, y_casebase, X_default, y_default, X_new_cases, y_new_cases
    )
    assert isinstance(accuracy, float)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)
    assert cm.shape == (2, 2)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_evaluate_model_batching_matches_unbatched(batch_size):
    model = make_model()
    batched = evaluate_model(
        model,
        X_casebase,
        y_casebase,
        X_default,
        y_default,
        X_new_cases,
        y_new_cases,
        batch_size=batch_size,
    )

    model_single = make_model()
    model_single.load_state_dict(model.state_dict())
    unbatched = evaluate_model(
        model_single, X_casebase, y_casebase, X_default, y_default, X_new_cases, y_new_cases
    )

    for batched_metric, unbatched_metric in zip(batched[:4], unbatched[:4]):
        assert batched_metric == pytest.approx(unbatched_metric)
    assert np.array_equal(batched[4], unbatched[4])


def test_evaluate_model_print_compute_graph(monkeypatch):
    """
    Regression test for two bugs fixed in evals.py's print_compute_graph
    block:
      1. `final_strengths` was undefined (NameError) after the eval loop was
         batched and its output variable renamed to `predictions`.
      2. After renaming it back to `final_strengths`, it was being read after
         `.cpu().detach().numpy()` had already stripped its autograd graph,
         so make_dot had nothing to draw.
    make_dot is monkeypatched here so this test doesn't depend on the
    Graphviz `dot` binary being installed.
    """
    captured = {}

    def fake_make_dot(var, **kwargs):
        captured["var"] = var

        class _FakeDot:
            def render(self, *args, **kwargs):
                return None

        return _FakeDot()

    monkeypatch.setattr("deeparguing.evals.evals.make_dot", fake_make_dot)

    model = make_model()
    evaluate_model(
        model,
        X_casebase,
        y_casebase,
        X_default,
        y_default,
        X_new_cases,
        y_new_cases,
        print_compute_graph=True,
    )

    assert "var" in captured
    assert isinstance(captured["var"], torch.Tensor)
    assert captured["var"].grad_fn is not None, (
        "compute graph tensor must retain its autograd history"
    )


def test_compute_graph_structure_is_well_formed():
    """
    Confirms make_dot renders a clean, connected autograd graph (not just
    a single disconnected node) for final_strengths.sum() on a toy example.
    """
    model = make_model()
    model.fit(X_casebase, y_casebase, X_default, y_default)

    final_strengths = model(X_new_cases)
    single_value = final_strengths.sum()

    dot = make_dot(single_value, params=dict(model.named_parameters()))
    source = dot.source

    assert single_value.grad_fn is not None
    assert "SumBackward" in source
    assert str(id(single_value)) in source
    assert "compute_base_scores.bias" in source
