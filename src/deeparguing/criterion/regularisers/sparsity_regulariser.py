from typing import override

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.criterion.criterion import Criterion
from deeparguing.criterion.regularisers.utils import FilterFunc


class SparsityRegulariser(Criterion):

    def __init__(self, filter_func: FilterFunc = lambda A: A):
        super().__init__()
        self.filter_func = filter_func

    @override
    def forward(self, model: GradualAACBR, predictions: Tensor, targets: Tensor) -> Tensor:
        assert model.A is not None
        A = self.filter_func(model.A)
        result = torch.sum(torch.abs(A))
        result = result / len(model.A)
        return result

    @override
    def step(self, model: GradualAACBR) -> bool:
        return True

    @override
    def reset(self) -> None:
        pass

class IrrelevanceSparsityRegulariser(Criterion):

    def __init__(self, filter_func: FilterFunc = lambda A: A):
        super().__init__()
        self.filter_func = filter_func

    @override
    def forward(self, model: GradualAACBR, predictions: Tensor, targets: Tensor) -> Tensor:
        A = -model.new_cases_attacks_adjacency
        A = self.filter_func(A)
        result = torch.sum(torch.abs(A))
        result = result / len(A)
        return result

    @override
    def step(self, model: GradualAACBR) -> bool:
        return True

    @override
    def reset(self) -> None:
        pass
