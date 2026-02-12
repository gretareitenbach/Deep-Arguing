from typing import override

import torch
from torch import Tensor

from deeparguing.semantics.gradual_semantics import GradualSemantics


class QuadraticEnergySemantics(GradualSemantics):

    def __init__(self, max_iters: int, epsilon: float | None = None, conservativeness: float = 1) -> None:
        super().__init__(max_iters, epsilon)
        self.conservativeness = conservativeness

    @override
    def aggregation_func(self, A: Tensor, strengths: Tensor):
        if A.ndim == 2:
            # Original d=1 case: A has shape (n, n), strengths has shape (n,) or (b, n)
            if strengths.ndim == 1:
                return torch.matmul(A.T, strengths)
            else:
                return torch.einsum("ji,bi->bi", A, strengths)
        elif A.ndim == 3:
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
    def influence_func(self, base_scores: Tensor, aggregations: Tensor):
        pos_mask = aggregations > 0

        scaled_aggregate = aggregations / self.conservativeness
        h = scaled_aggregate**2 / (1 + scaled_aggregate**2)

        positive_update = h * (1 - base_scores)
        negative_update = -h * base_scores

        update = torch.where(pos_mask, positive_update, negative_update)

        result = base_scores + update

        return result

