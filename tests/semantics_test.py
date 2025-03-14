import pytest
import torch

from deeparguing.semantics.relu_semantics import ReluSemantics
from deeparguing.semantics.sigmoid_semantics import SigmoidSemantics

A = torch.tensor(
    [
        #    0  1  2  3  4
        [0, -1, 0, 0, 0],  # 0
        [-1, 0, 0, 0, 0],  # 1
        [1, 0, 0, -1, 0],  # 2
        [0, 1, 0, 0, 0],  # 3
        [1, 0, 0, -1, 0],  # 4
    ],
    dtype=torch.float32,
)


base_scores = torch.tensor([0.5, 0.5, 0.7, 0.8, 0.9])

expected_map = {
    # Relu Tests
    ("Relu Base Case", 0, 0, ReluSemantics): base_scores,
    ("Relu one iteration", 1, 0, ReluSemantics): torch.tensor(
        [1.6000, 0.8000, 0.7000, 0.000, 0.9000]
    ),
    ("Relu large epsilon = one iteration", 5, 10, ReluSemantics): torch.tensor(
        [1.6000, 0.8000, 0.7000, 0.000, 0.9000]
    ),
    ("Relu five iterations", 5, 0, ReluSemantics): torch.tensor(
        [2.1000, 0.000, 0.7000, 0.000, 0.9000]
    ),
    # Sigmoid Tests
    ("Sigmoid Base Case", 0, 0, SigmoidSemantics): base_scores,
    ("Sigmoid one iteration", 1, 0, SigmoidSemantics): torch.tensor(
        [0.7503, 0.5744, 0.7000, 0.4468, 0.9000]
    ),
    ("Sigmoid large epsilon = one iteration", 5, 10, SigmoidSemantics): torch.tensor(
        [0.7503, 0.5744, 0.7000, 0.4468, 0.9000]
    ),
    ("Sigmoid five iterations", 5, 0, SigmoidSemantics): torch.tensor(
        [0.7647, 0.4215, 0.7000, 0.4468, 0.9000]
    ),
}


@pytest.mark.parametrize("params", list(expected_map.keys()))
def test_semantics(params):
    test_name, max_iters, epsilon, semantic_class = params
    semantics = semantic_class(max_iters=max_iters, epsilon=epsilon)
    strengths = semantics(A, base_scores)
    expected = expected_map[params]
    print(expected)
    print(strengths)
    assert torch.allclose(
        strengths, expected, atol=1e-4
    ), f"{test_name} Failed: Expected {expected}. Actual {strengths}."
