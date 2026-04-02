from typing import override
import torch

from torch import Tensor

from deeparguing.semantics.gradual_semantics import GradualSemantics
from deeparguing.t_norm import TNorm


class FuzzyGroundedSemantics(GradualSemantics):

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

        A = -A
        assert(torch.all(A >= 0))

        tnorm = self.t_norm

        if A.ndim == 2:
            # A: (n, n)
            # strengths: (n,) OR (b, n)

            A_t = A.T  # (n, n)

            if strengths.ndim == 1:
                # (n,) → treat as (1, n)
                agg = tnorm.and_op(strengths[:, None], A_t)  # (n, 1)  # (n, n)
                return tnorm.or_aggregate(agg, dim=0)

            else:
                # strengths: (b, n)
                agg = tnorm.and_op(
                    strengths[:, :, None], A_t[None, :, :]  # (b, n, 1)  # (1, n, n)
                )

                return tnorm.or_aggregate(agg, dim=1)

        elif A.ndim == 3:
            # A: (n, n, d)
            # strengths: (b, n, d)

            A_t = A.permute(1, 0, 2)  # (n, n, d)

            agg = tnorm.and_op(
                strengths[:, :, None, :],  # (b, n, 1, d)
                A_t[None, :, :, :],  # (1, n, n, d)
            )

            return tnorm.or_aggregate(agg, dim=1)

        elif A.ndim == 4:
            # A: (b, 1, n, d)
            # strengths: (b, d, 1)

            strengths = strengths.transpose(1, 2)  # (b, 1, d)

            agg = tnorm.and_op(
                A, strengths[:, :, None, :]  # (b, 1, n, d)  # (b, 1, 1, d)
            )

            return tnorm.or_aggregate(agg, dim=2)

        else:
            raise ValueError("Unsupported tensor rank for A")

    # @override
    # def aggregation_func(self, A: Tensor, strengths: Tensor):
    #
    #     A_t = A.T
    #
    #     agg = self.t_norm.and_op(
    #         strengths[:, :, None],
    #         A_t[None, :, :]
    #     )
    #
    #     return self.t_norm.or_aggregate(agg, dim=1)

    @override
    def influence_func(self, base_scores: Tensor, aggregations: Tensor):

        # base_scores (B, n)
        # aggregations (B, n)
        return self.t_norm.or_op(1 - aggregations, base_scores)
