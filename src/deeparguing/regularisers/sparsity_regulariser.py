from typing import override

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.regularisers.regulariser import Regulariser
from deeparguing.regularisers.utils import FilterFunc


class SparsityRegulariser(Regulariser):

    def __init__(self, filter_func: FilterFunc = lambda A: A):
        super().__init__()
        self.filter_func = filter_func

    @override
    def forward(self, model: GradualAACBR) -> Tensor:
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
