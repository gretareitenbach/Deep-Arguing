from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Tuple, Union, override

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR

type FilterFunc = Callable[[Tensor], Tensor]
type RegulariserType = Union[Callable[[Any], float], Regulariser]


class Regulariser(metaclass=ABCMeta):

    @abstractmethod
    def forward(self, model: GradualAACBR) -> Tensor:
        pass

    def __call__(self, model: GradualAACBR) -> Tensor:
        return self.forward(model)


class SparsityRegulariser(Regulariser):

    def __init__(self, filter_func: FilterFunc = lambda A: A):
        super().__init__()
        self.filter_func = filter_func

    @override
    def forward(self, model: GradualAACBR):
        assert model.A != None
        A = self.filter_func(model.A)
        result = torch.sum(torch.abs(A))
        result = result / len(model.A)
        return result


class CommunityPreservationRegulariser(Regulariser):

    def __init__(self, filter_func: FilterFunc = lambda A: A, method: str = "svd"):
        super().__init__()
        self.filter_func = filter_func
        methods = ["svd", "nuc", "fro"]
        if method not in methods:
            return ValueError(
                f"Unknown community preservation method: {method}. Please select from {methods}."
            )
        if method == "svd":
            self.method: FilterFunc = lambda A: torch.sum(torch.svd(A).S)
        if method == "nuc":
            self.method: FilterFunc = lambda A: torch.linalg.norm(A, ord="nuc")
        if method == "fro":
            self.method: FilterFunc = lambda A: torch.norm(A, p="fro")

    @override
    def forward(self, model: GradualAACBR):
        assert model.A != None
        A = self.filter_func(model.A)
        A = torch.abs(A)  # This regulariser expects values between 0 and 1
        return self.method(A)


class ConnectivityRegulariser(Regulariser):

    def __init__(self, filter_func: FilterFunc = lambda A: A, epsilon: float = 1e-8):
        super().__init__()
        self.filter_func = filter_func
        self.epsilon = epsilon

    @override
    def forward(self, model: GradualAACBR):
        assert model.A != None
        A = self.filter_func(model.A)
        A = torch.abs(A)  # This regulariser expects values between 0 and 1
        A = torch.sum(A, dim=1) + self.epsilon
        result = -torch.sum(torch.log(A))
        result = result / len(model.A)
        return result


class DAGRegulariser(Regulariser):
    """
    This is an modification of NOTEARS (https://proceedings.neurips.cc/paper_files/paper/2018/file/e347c51419ffb23ca3fd5050202f9c3d-Paper.pdf)
    which introduces a regulariser that forces the learned graph to be a DAG
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
    def forward(self, model: GradualAACBR) -> Tensor:
        assert model.A != None
        A = self.filter_func(model.A)
        d = A.shape[0]
        m = torch.matrix_exp(A * A)
        h = torch.trace(m) - d
        result = h + self.alpha * h ** 2
        # result = 0.5 * self.rho * (h**2) + self.alpha * h
        # self.alpha = self.alpha * 1.005

        return result


class RegulariserList(Regulariser):

    def __init__(self, regularisers: list[Tuple[Regulariser, float]]):
        super().__init__()
        self.regularisers = regularisers

    @override
    def forward(self, model: GradualAACBR):
        assert model.A != None

        total = torch.zeros(1, device=model.device).squeeze()

        for reg_func, weight in self.regularisers:
            total += weight * reg_func(model)

        return total


filter_to_attacks: FilterFunc = lambda A: torch.where(A < 0, A, 0)
filter_to_supports: FilterFunc = lambda A: torch.where(A > 0, A, 0)
