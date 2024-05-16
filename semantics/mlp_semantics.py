from .gradual_semantics import GradualSemantics
import torch


class MLPBasedSemantics(GradualSemantics):

    def aggregation_func(self, A, strengths):
        return torch.matmul(A.T, strengths)

    def influence_func(self, base_scores, aggregations):
        return torch.sigmoid(
            torch.log(base_scores/(1 - base_scores)) + aggregations
        )
