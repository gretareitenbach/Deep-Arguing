from typing import override

import torch
from torch import Tensor

from deeparguing.semantics.gradual_semantics import GradualSemantics


class SigmoidSemantics(GradualSemantics):

    def __init__(self, max_iters: int, epsilon: float | None = None) -> None:
        super().__init__(max_iters, epsilon)

    @override
    def aggregation_func(self, A: Tensor, strengths: Tensor):
        if A.ndim == 3:
            # A has shape (n, n, d)
            # Strengths has shape (b, n, d)
            return torch.einsum("jid,bjd->bid", A, strengths)
        elif A.ndim == 4:
            # A has shape (b, 1, n, d)
            # Strengths has shape (b, d, 1)
            result = torch.einsum("bjid,bdj->bid", A, strengths)
            return result

        raise ValueError(f"Adjacency Matrix A of shape {A.shape} is incorrect.")

    @override
    def influence_func(self, base_scores: Tensor, aggregations: Tensor) -> Tensor:
        return torch.sigmoid(torch.log(base_scores / (1 - base_scores)) + aggregations)
