from typing import override

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.criterion.criterion import Criterion
from deeparguing.criterion.regularisers.utils import FilterFunc


class LaplacianRegulariser(Criterion):
    """
    Encourage nodes of the same class to have similar connection patterns.

    This regulariser computes a smoothness penalty based on the graph Laplacian,
    penalizing cases where nodes of the same class have very different
    incoming/outgoing edge patterns.
    """

    def __init__(
        self,
        y_train: Tensor,
        filter_func: FilterFunc = lambda A: A,
        epsilon: float = 1e-8,
    ):
        """
        Parameters
        ----------
        y_train : Tensor
            One-hot encoded training labels, shape (N, C) where C is number of classes.
        filter_func : FilterFunc
            Optional filter to apply to adjacency matrix before computing.
        epsilon : float
            Small constant for numerical stability.
        """
        super().__init__()
        self.y_train = y_train
        self.filter_func = filter_func
        self.epsilon = epsilon
        self.class_indices = torch.argmax(y_train, dim=1)
        self.num_classes = y_train.shape[1]

    @override
    def forward(self, model: GradualAACBR, predictions: Tensor, targets: Tensor) -> Tensor:
        assert model.A is not None
        A = self.filter_func(model.A)
        A = torch.abs(A)

        _n = A.shape[0]
        _d = A.shape[2]

        A_summed = A.sum(dim=2)

        smoothness_penalty = torch.zeros(1, device=A.device).squeeze()

        for c in range(self.num_classes):
            class_mask = self.class_indices == c
            class_node_indices = torch.where(class_mask)[0]

            if len(class_node_indices) < 2:
                continue

            class_rows = A_summed[class_node_indices, :]

            mean_pattern = class_rows.mean(dim=0, keepdim=True)

            deviations = class_rows - mean_pattern
            class_penalty = (deviations ** 2).sum()

            smoothness_penalty = smoothness_penalty + class_penalty / (len(class_node_indices) + self.epsilon)

        return smoothness_penalty / self.num_classes

    @override
    def step(self, model: GradualAACBR) -> bool:
        return True

    @override
    def reset(self) -> None:
        pass
