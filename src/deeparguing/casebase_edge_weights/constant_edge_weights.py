from typing import override
import torch
from torch import Tensor

from deeparguing.casebase_edge_weights.compute_partial_order import \
    ComputePartialOrder


class ConstantPartialOrder(ComputePartialOrder):

    def __init__(self, edge_weights: int, dim: int = 1) -> None:
        super(ConstantPartialOrder, self).__init__()
        self.edge_weights = edge_weights
        self.dim = dim

    @override
    def forward(self, attacker: Tensor, target: Tensor) -> Tensor:
        return torch.ones((attacker.shape[0],  target.shape[0], self.dim), device=attacker.device) * self.edge_weights

    @override
    def plot_parameters(self):
        print(self.edge_weights)
