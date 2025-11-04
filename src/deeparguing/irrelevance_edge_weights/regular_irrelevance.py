from typing import override

from torch import Tensor

from deeparguing.casebase_edge_weights.compute_partial_order import \
    ComputePartialOrder
from deeparguing.irrelevance_edge_weights.compute_irrelevance import \
    ComputeIrrelevance


class RegularIrrelevance(ComputeIrrelevance):

    def __init__(self, compute_partial_order: ComputePartialOrder):
        super(RegularIrrelevance, self).__init__()

        self.compute_partial_order = compute_partial_order

    @override
    def forward(self, new_cases: Tensor, X_train: Tensor) -> Tensor:

        return 1 - self.compute_partial_order(new_cases, X_train)

    @override
    def plot_parameters(self):
        self.compute_partial_order.plot_parameters()
