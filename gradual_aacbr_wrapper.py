import torch
from semantics.gradual_semantics import GradualSemantics
from new_case_influence.new_case_attacks import NewCasesAttacks
from edge_weights.compute_edge_weights import ComputeEdgeWeights
from typing import Callable


class GradualAACBRWrapper(torch.nn.Module):

    def __init__(self, compute_base_score: Callable[[torch.Tensor], torch.Tensor], gradual_semantics: GradualSemantics, new_case_influence: NewCasesAttacks, compute_edge_weights: ComputeEdgeWeights):
        super(GradualAACBRWrapper, self).__init__()
        # n = no.nodes
        # f = no.features
        self.gradual_semantics = gradual_semantics
        self.compute_base_scores = compute_base_score
        self.new_case_influence = new_case_influence
        self.compute_edge_weights = compute_edge_weights

    def batch_base_scores(self, base_scores: torch.Tensor, batch_size: int) -> torch.Tensor:
        base_scores = torch.tile(base_scores.unsqueeze(
            dim=0), (batch_size, 1))  # (B x n)
        base_scores = base_scores.unsqueeze(2)  # (B x n x 1)
        return base_scores

    def forward(self, nodes: torch.Tensor, A: torch.Tensor, new_cases: torch.Tensor, new_cases_attacks: torch.Tensor):


        if new_cases_attacks.dim() == 1:
            new_cases_attacks = new_cases_attacks.unsqueeze(dim=0)
        batch_size = new_cases_attacks.shape[0]

        base_scores = self.compute_base_scores(nodes)  # (n)
        base_scores = self.batch_base_scores(base_scores, batch_size)

        A = self.compute_edge_weights(A, nodes)

        base_scores, A = self.new_case_influence(
            base_scores, A, new_cases, new_cases_attacks, self.compute_base_scores, self.gradual_semantics)

        strengths = self.gradual_semantics(A, base_scores)

        # TODO: Check if this is necessary:
        final_strengths = strengths[-1].squeeze()
        if final_strengths.dim() == 1:
            final_strengths = final_strengths.unsqueeze(0)

        return final_strengths
