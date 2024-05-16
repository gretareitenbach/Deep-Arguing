import torch
from semantics.gradual_semantics import GradualSemantics
from typing import Callable, Sequence


class GradualAACBR(torch.nn.Module):

    def __init__(self, compute_base_score: Callable[..., Sequence], gradual_semantics: GradualSemantics):
        super(GradualAACBR, self).__init__()
        # n = no.nodes
        # f = no.features
        self.gradual_semantics = gradual_semantics
        self.compute_base_scores = compute_base_score

    def batch_base_scores(self, nodes, new_cases_attacks):
        batch_size = new_cases_attacks.shape[0]
        base_scores = self.compute_base_scores(nodes)  # (n)
        base_scores = torch.tile(base_scores.unsqueeze(
            dim=0), (batch_size, 1))  # (B x n)
        # TODO: new_case should do one of:
        #   CURRENT: set base scores of any that it attacks to 0 -> i.e. irrelevant arguments have no strength
        #   POSSIBLE: compute it's base strength and use that + attacks to influence what it attacks
        # TODO: Change how this handled
        base_scores = torch.where(new_cases_attacks, 1e-6, base_scores)
        base_scores = base_scores.unsqueeze(2)  # (B x n x 1)
        return base_scores

    def forward(self, nodes, A, new_cases_attacks, max_iters=5):

        if new_cases_attacks.dim() == 1:
            new_cases_attacks = new_cases_attacks.unsqueeze(dim=0)

        base_scores = self.batch_base_scores(nodes, new_cases_attacks)

        strengths = self.gradual_semantics(
            A, base_scores, max_iters, epsilon=0)

        # TODO: Check if this is necessary:
        final_strengths = strengths[-1].squeeze()
        if final_strengths.dim() == 1:
            final_strengths = final_strengths.unsqueeze(0)

        return final_strengths
