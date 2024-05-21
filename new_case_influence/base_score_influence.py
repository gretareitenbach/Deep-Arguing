from .new_case_attacks import NewCasesAttacks
import torch


class BaseScoreInfluence(NewCasesAttacks):

    def forward(self, base_scores, A, new_cases, new_cases_attacks, compute_base_score, gradual_semantics):
        """
        Utilises the base score of the new cases to alter the strengths of the irrelevant cases.

        """

        new_cases_base_scores = compute_base_score(
            new_cases).unsqueeze(-1).unsqueeze(-1)  # (B x 1)

        new_cases_attacks_adjacency = torch.where(
            new_cases_attacks, -1, 0).to(dtype=torch.float32)  # B x n
        # B x 1 x n
        new_cases_attacks_adjacency = new_cases_attacks_adjacency.unsqueeze(-2)

        # We compute the aggregations *only* for the attacks by the new cases.
        # As new cases are unattacked, this can be computed in a single pass of aggregation/influence function
        aggregations = gradual_semantics.aggregation_func(
            new_cases_attacks_adjacency, new_cases_base_scores)
        strengths = gradual_semantics.influence_func(base_scores, aggregations)

        return strengths, A
