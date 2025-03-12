import torch

from deeparguing.casebase_edge_weights.compute_partial_order import \
    ComputePartialOrder


class ConstantPartialOrder(ComputePartialOrder):

    def __init__(self, edge_weights: int) -> None:
        super(ConstantPartialOrder, self).__init__()
        self.edge_weights = edge_weights

    def forward(self, attacker: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.ones(attacker.shape[0] * target.shape[0]) * self.edge_weights

    def plot_parameters(self):
        print(self.edge_weights)
