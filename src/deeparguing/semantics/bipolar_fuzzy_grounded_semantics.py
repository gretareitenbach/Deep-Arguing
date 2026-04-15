from typing import override

import torch
from torch import Tensor

from deeparguing.semantics.gradual_semantics import GradualSemantics
from deeparguing.t_norm import TNorm


class BipolarFuzzyGroundedSemantics(GradualSemantics):

    def __init__(
        self,
        t_norm: TNorm,
        max_iters: int,
        epsilon: float | None = None,
        damping: float = 1,
    ) -> None:
        super().__init__(max_iters, epsilon, damping)
        self.t_norm: TNorm = t_norm

    @override
    def aggregation_func(self, A: Tensor, strengths: Tensor):

        A_attacks = torch.where(A < 0, -A, 0)
        A_support = torch.where(A > 0, A, 0)

        att_agg = self._agg(A_attacks, strengths)
        support_agg = self._agg(A_support, strengths)

        return torch.maximum(1 - att_agg, support_agg)


    def _agg(self, A: Tensor, strengths: Tensor):

        assert torch.all(A >= 0)

        tnorm = self.t_norm

        if A.ndim == 2:
            # A: (n, n)
            # strengths: (n,) OR (b, n)

            if strengths.ndim == 1:
                # (n,) → treat as (1, n)
                agg = tnorm.and_op(strengths[:, None], A)  # (n, 1)  # (n, n)
                return tnorm.or_aggregate(agg, dim=0)

            else:
                # strengths: (b, n)
                agg = tnorm.and_op(
                    strengths[:, :, None], A[None, :, :]  # (b, n, 1)  # (1, n, n)
                )

                return tnorm.or_aggregate(agg, dim=1)

        elif A.ndim == 3:
            # A: (n, n, d)
            # strengths: (b, n, d)

            agg = tnorm.and_op(
                strengths[:, :, None, :],  # (b, n, 1, d)
                A[None, :, :, :],  # (1, n, n, d)
            )

            return tnorm.or_aggregate(agg, dim=1)

        elif A.ndim == 4:
            # A: (b, 1, n, d)
            # strengths: (b, d, 1)

            agg = tnorm.and_op(
                A, strengths[:, :, None, :]  # (b, 1, n, d)  # (b, 1, 1, d)
            )

            return tnorm.or_aggregate(agg, dim=1)  # b, n, d

        else:
            raise ValueError("Unsupported tensor rank for A")

    @override
    def influence_func(self, base_scores: Tensor, aggregations: Tensor):

        # base_scores (B, n)
        # aggregations (B, n)
        return self.t_norm.and_op(aggregations, base_scores)


