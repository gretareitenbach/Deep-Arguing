from typing import override

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.regularisers.regulariser import Regulariser
from deeparguing.regularisers.utils import FilterFunc


class ClassBalanceRegulariser(Regulariser):
    """
    Penalize imbalance in edge weights across class-specific subgraphs.

    This regulariser encourages the adjacency matrix to have balanced
    edge weight distributions across all classes, preventing the model
    from favoring certain classes in its graph structure.
    """

    def __init__(
        self,
        filter_func: FilterFunc = lambda A: A,
        epsilon: float = 1e-8,
    ):
        """
        Parameters
        ----------
        filter_func : FilterFunc
            Optional filter to apply to adjacency matrix before computing.
        epsilon : float
            Small constant for numerical stability.
        """
        super().__init__()
        self.filter_func = filter_func
        self.epsilon = epsilon

    @override
    def forward(self, model: GradualAACBR) -> Tensor:
        assert model.A is not None
        assert model.y_train is not None
        y_train = model.y_train
        self.num_classes = y_train.shape[1]
        self.class_indices = torch.argmax(y_train, dim=1)
        A = self.filter_func(model.A)
        A = torch.abs(A)

        class_edge_sums: list[Tensor] = []

        for c in range(self.num_classes):

            class_mask = self.class_indices == c
            class_node_indices = torch.where(class_mask)[0]

            if len(class_node_indices) == 0:
                class_edge_sums.append(torch.zeros(1, device=A.device))
                continue

            edges_to_class = A[class_node_indices, :, :].sum()
            edges_from_class = A[:, class_node_indices, :].sum()
            total_edges = edges_to_class + edges_from_class

            avg_edges = total_edges / (len(class_node_indices) + self.epsilon)
            class_edge_sums.append(avg_edges.unsqueeze(0))

        class_edge_tensor = torch.cat(class_edge_sums)
        variance = torch.var(class_edge_tensor)

        return variance

    @override
    def step(self, model: GradualAACBR) -> bool:
        return True

    @override
    def reset(self) -> None:
        pass
