from typing import override

import torch
from torch import Tensor

from deeparguing.base_scores.compute_base_scores import ComputeBaseScores


class ConstantBaseScore(ComputeBaseScores):

    def __init__(self, constant: float, dim: int = 1):
        super(ConstantBaseScore, self).__init__()
        self.constant = constant
        self.dim = dim

    @override
    def forward(self, nodes: Tensor) -> Tensor:
        batch_size = nodes.shape[0]
        return torch.ones((batch_size, self.dim), device=nodes.device) * self.constant

    @override
    def plot_parameters(self):
        print("Constant:", self.constant)
