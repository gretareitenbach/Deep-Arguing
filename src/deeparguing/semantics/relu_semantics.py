from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor

from deeparguing.semantics.gradual_semantics import GradualSemantics


class ReluSemantics(GradualSemantics):

    def __init__(
        self,
        max_iters: int,
        epsilon: float | None = None,
        damping: float = 1,
        use_soft_relu: bool = False,
    ) -> None:
        super().__init__(max_iters, epsilon, damping)
        self.use_soft_relu = use_soft_relu

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
        f = F.softplus if self.use_soft_relu else torch.relu
        return f(f(base_scores) + aggregations)
