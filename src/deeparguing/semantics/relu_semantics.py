from typing import override

import torch
from torch import Tensor

from deeparguing.semantics.gradual_semantics import GradualSemantics


class ReluSemantics(GradualSemantics):

    def __init__(self, max_iters: int, epsilon: float | None = None) -> None:
        super().__init__(max_iters, epsilon)

    @override
    def aggregation_func(self, A: Tensor, strengths: Tensor):
        return torch.matmul(torch.transpose(A, -2, -1), strengths)

    @override
    def influence_func(self, base_scores: Tensor, aggregations: Tensor):
        return torch.relu(torch.relu(base_scores) + aggregations)
