from typing import override

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.criterion.criterion import Criterion
from deeparguing.criterion.regularisers.utils import FilterFunc


class DAGRegulariser(Criterion):
    """
    This is a modification of NOTEARS (https://proceedings.neurips.cc/paper_files/paper/2018/file/e347c51419ffb23ca3fd5050202f9c3d-Paper.pdf)
    which introduces a regulariser that forces the learned graph to be a DAG.
    """

    def __init__(
        self,
        filter_func: FilterFunc = lambda A: A,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.filter_func = filter_func
        self.alpha = alpha

    @override
    def forward(self, model: GradualAACBR, predictions: Tensor, targets: Tensor) -> Tensor:
        assert model.A is not None
        A = self.filter_func(model.A)
        d = A.shape[0]
        A = A.permute(2, 0, 1)
        m = torch.matrix_exp(A * A)
        h = torch.diagonal(m, dim1=1, dim2=2).sum(1) - d
        result = h + self.alpha * h**2
        result = result.sum()

        return result

    @override
    def step(self, model: GradualAACBR) -> bool:
        return True

    @override
    def reset(self) -> None:
        pass
