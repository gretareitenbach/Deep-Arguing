from abc import ABC, abstractmethod
from typing import Any


class GradualSemantics(ABC):

    @abstractmethod
    def aggregation_func(self, A, strengths):
        """
        A = N x N adjacency matrix 
        strengths =  N x 1  - strength vector

        return N x 1 - B aggregation vector
        """
        pass

    @abstractmethod
    def influence_func(self, base_scores, aggregations):
        """
        base_scores = N x 1 - B base_score vectors
        aggregations =  N x 1 - B aggregation vectors

        return N x 1 - strength vector
        """
        pass

    def forward(self, A, base_scores, strengths):
        aggregations = self.aggregation_func(A, strengths)
        return self.influence_func(base_scores, aggregations)

    def forward_till_convergence(self, A, base_scores, max_iters, epsilon):
        strengths = [base_scores]
        # TODO: change to use one of the following stop conditions:
        #   (convergence under some epsilon or max iters reached) OR
        #   sort the nodes topologically and figure out how to do a single pass with matrix operations -> Only works for ACYCLIC graphs
        for i in range(max_iters):
            # TODO: consider epsilon in forward pass
            strengths.append(self.forward(A, base_scores, strengths[i]))

        return strengths

    def __call__(self, A, base_scores, max_iters, epsilon) -> Any:
        return self.forward_till_convergence(A, base_scores, max_iters, epsilon)
