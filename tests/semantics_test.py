import pytest
import torch
from deeparguing.semantics.relu_semantics import ReluSemantics
from deeparguing.semantics.sigmoid_semantics import SigmoidSemantics



A = torch.tensor([
                #    0  1  2  3  4
                    [0, -1, 0, 0, 0], # 0
                    [-1, 0, 0, 0, 0], # 1
                    [1, 0, 0, -1, 0], # 2
                    [0, 1, 0, 0, 0], # 3
                    [1, 0, 0, -1, 0], # 4
                ], dtype=torch.float32)


base_scores = torch.tensor([0.5, 0.5, 0.7, 0.8, 0.9])

expected_map = {
    (0, 0, ReluSemantics): base_scores,
    (0, 0, SigmoidSemantics): base_scores, 
    (5, 0, SigmoidSemantics): torch.tensor([0.7647, 0.4215, 0.7000, 0.4468, 0.9000]),
}


@pytest.mark.parametrize("params", list(expected_map.keys()))
def test_semantics(params):
    max_iters, epsilon, semantic_class = params
    semantics = semantic_class(max_iters=max_iters, epsilon=epsilon)
    strengths = semantics(A, base_scores)
    expected = expected_map[(max_iters, epsilon, semantic_class)]
    print(expected)
    print(strengths)
    assert torch.allclose(strengths, expected, atol=1e-4)




