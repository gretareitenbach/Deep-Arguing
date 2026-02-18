from typing import override

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.regularisers.regulariser import Regulariser
from deeparguing.regularisers.utils import FilterFunc


class ClassConnectivityRegulariser(Regulariser):
    """
    Penalize classes with insufficient incoming edges in the adjacency matrix.

    This regulariser helps prevent class collapse by ensuring each class
    maintains a minimum level of connectivity in the argumentation graph.
    """

    def __init__(
        self,
        y_train: Tensor,
        min_edges: float = 1.0,
        filter_func: FilterFunc = lambda A: A,
        epsilon: float = 1e-8,
    ):
        """
        Parameters
        ----------
        y_train : Tensor
            One-hot encoded training labels, shape (N, C) where C is number of classes.
        min_edges : float
            Minimum expected incoming edge weight sum per class node.
        filter_func : FilterFunc
            Optional filter to apply to adjacency matrix before computing.
        epsilon : float
            Small constant for numerical stability.
        """
        super().__init__()
        self.y_train = y_train
        self.min_edges = min_edges
        self.filter_func = filter_func
        self.epsilon = epsilon
        self.class_indices = torch.argmax(y_train, dim=1)
        self.num_classes = y_train.shape[1]

    @override
    def forward(self, model: GradualAACBR) -> Tensor:
        assert model.A is not None
        A = self.filter_func(model.A)
        A = torch.abs(A)

        _n = A.shape[0]
        _num_casebase = _n - self.num_classes

        penalty = torch.zeros(1, device=A.device).squeeze()

        for c in range(self.num_classes):
            class_mask = self.class_indices == c
            class_node_indices = torch.where(class_mask)[0]

            if len(class_node_indices) == 0:
                continue

            incoming_edges = A[class_node_indices, :, :].sum()
            avg_incoming = incoming_edges / (len(class_node_indices) + self.epsilon)

            shortfall = torch.relu(self.min_edges - avg_incoming)
            penalty = penalty + shortfall

        return penalty / self.num_classes

    @override
    def step(self, model: GradualAACBR) -> bool:
        return True

    @override
    def reset(self) -> None:
        pass
