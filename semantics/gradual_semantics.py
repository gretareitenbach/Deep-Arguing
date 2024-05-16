from abc import ABC, abstractmethod
from typing import Any


class GradualSemantics(ABC):

    @abstractmethod
    def aggregation_func(self, A, strengths):
        """
        Computes the aggregation vector based on the given adjacency matrix and strength vector.

        Parameters:
            A : Array-like
                N x N adjacency matrix representing connections between nodes.
            strengths : Array-like
                N x 1 vector representing strengths of nodes.

        Returns:
            Array-like
                N x 1 aggregation vector.

        Examples:
            # Example usage:
            A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
            strengths = np.array([0.5, 0.8, 0.6])
            result = aggregation_func(A, strengths)
        """
        pass

    @abstractmethod
    def influence_func(self, base_scores, aggregations):
        """
        Computes the strength vector based on the given base score vectors and aggregation vectors.

        Parameters:
            base_scores : Array-like
                N x 1 vector representing base scores.
            aggregations : Array-like
                N x 1 vector representing aggregation scores.

        Returns:
            Array-like
                N x 1 strength vector.

        Examples:
            # Example usage:
            base_scores = np.array([0.5, 0.8, 0.6])
            aggregations = np.array([0.2, 0.4, 0.3])
            result = influence_func(base_scores, aggregations)
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
