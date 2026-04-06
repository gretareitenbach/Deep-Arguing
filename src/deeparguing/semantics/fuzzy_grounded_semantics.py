from typing import override

import torch
from torch import Tensor

from deeparguing.semantics.gradual_semantics import GradualSemantics
from deeparguing.t_norm import GodelTNorm, TNorm


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
        return self.t_norm.and_op(1 - aggregations, base_scores)


# if __name__ == "__main__":
#
#     """
#
#     c -> b -> a
#
#     """
#
#     # A = torch.tensor(
#     #     [
#     #         [0, 0, 0],
#     #         [-1, 0, 0],
#     #         [0, -1, 0],
#     #     ],
#     #     dtype=torch.float32,
#     # ).unsqueeze(-1)
#     #
#     # base_scores = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32).unsqueeze(
#     #     -1
#     # )
#
#     # semantics = FuzzyGroundedSemantics(GodelTNorm(), max_iters=5)
#     # aggregations = semantics.aggregation_func(A, base_scores)
#     # influence = semantics.influence_func(base_scores, aggregations)
#     # print("Agg", aggregations)
#     # print("Inf", influence)
#     #
#     # print("final", semantics.forward_till_convergence(A, base_scores))
#
#     """
#     1) N -> a; N-> b; N -> c
#     2) N-> b;
#     3) N -> a;  N -> c
#     """
#
#     A = (
#         torch.tensor(
#             [
#                 [-1, -1, -1],
#                 [0, -1, 0],
#                 [-1, 0, -1],
#             ],
#             dtype=torch.float32,
#         )
#         .unsqueeze(1)
#         .unsqueeze(-1)
#     )
#
#     print("A shape", A.shape)
#
#     new_case_strengths = (
#         torch.tensor([1, 1, 1], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
#     )
#
#     base_scores = torch.tensor(
#         [
#             [1, 1, 1],
#             [1, 1, 1],
#             [1, 1, 1],
#         ],
#         dtype=torch.float32,
#     ).unsqueeze(-1)
#
#     print("new case strengths shape", new_case_strengths.shape)
#     print("base scores shape", base_scores.shape)
#
#     semantics = FuzzyGroundedSemantics(GodelTNorm(), max_iters=5)
#     aggregations = semantics.aggregation_func(A, new_case_strengths)
#     influence = semantics.influence_func(base_scores, aggregations)
#     print("Agg", aggregations)
#     print("Inf", influence)

# final = semantics.forward_till_convergence(A, base_scores)
# print("final", final)
# print("final shape", final.shape)
