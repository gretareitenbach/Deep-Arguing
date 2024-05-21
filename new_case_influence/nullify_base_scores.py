from .new_case_attacks import NewCasesAttacks
import torch


class NullifyBaseScores(NewCasesAttacks):


    def forward(self, base_scores, A, new_cases, new_cases_attacks, compute_base_score, gradual_semantics):
        """

        Parameters:
            base_scores : Torch Tensor
                N x 1 vector representing base scores.
            
            A : Torch Tensor
                N x N adjacency matrix representing connections between nodes.
            
            new_cases_attacks: Torch Tensor
                N vector - the value at new_cases_attacks[i] is true if the new case 
                attacks case i and false otherwise

        Returns:
            base_scores : Array-like
                N x 1 vector representing base scores.
            
            A : Array-like
                N x N adjacency matrix representing connections between nodes.
        """
        # TODO: new_case should do one of:
        #   CURRENT: set base scores of any that it attacks to 0 -> i.e. irrelevant arguments have no strength
        #   POSSIBLE: compute it's base strength and use that + attacks to influence what it attacks
        # TODO: Change how this handled
        new_cases_attacks = new_cases_attacks.unsqueeze(2)
        base_scores = torch.where(new_cases_attacks, 1e-6, base_scores)
        return base_scores, A