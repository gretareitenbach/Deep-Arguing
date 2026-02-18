from typing import override

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.regularisers.regulariser import Regulariser
from deeparguing.regularisers.utils import FilterFunc


class NOTEARSRegulariser(Regulariser):
    """
    NOTEARS regulariser using augmented Lagrangian method.

    Computes: lambda * h(A) + (rho/2) * h(A)^2

    Where h(A) = trace(e^{A*A}) - d is the acyclicity constraint.

    After each outer iteration, call step() to update lambda and rho:
        lambda <- lambda + rho * h
        rho <- rho * rho_factor  (if h didn't improve by factor gamma)

    Reference: https://proceedings.neurips.cc/paper_files/paper/2018/file/e347c51419ffb23ca3fd5050202f9c3d-Paper.pdf
    """

    def __init__(
        self,
        filter_func: FilterFunc = lambda A: A,
        lambda_init: float = 0.0,
        rho_init: float = 1.0,
        rho_max: float = 1e16,
        rho_factor: float = 10.0,
        gamma: float = 0.25,
        h_tolerance: float = 1e-8,
    ):
        super().__init__()
        self.filter_func = filter_func

        self.lambda_init = lambda_init
        self.rho_init = rho_init

        self.rho_max = rho_max
        self.rho_factor = rho_factor
        self.gamma = gamma
        self.h_tolerance = h_tolerance

        self.reset()

    def compute_h(self, model: GradualAACBR) -> Tensor:
        """
        Compute h(A) = trace(e^{A*A}) - d

        This equals 0 if and only if A is a DAG.
        """
        assert model.A is not None
        A = self.filter_func(model.A)
        d = A.shape[0]
        A = A.permute(2, 0, 1)
        m = torch.matrix_exp(A * A)
        h = torch.diagonal(m, dim1=1, dim2=2).sum(1) - d
        return h.sum()

    @override
    def forward(self, model: GradualAACBR) -> Tensor:
        """Compute lambda * h + (rho/2) * h^2"""
        h = self.compute_h(model)
        return self.lambda_ * h + (self.rho / 2) * h**2

    @override
    def step(self, model: GradualAACBR) -> bool:
        """
        Update lambda and rho after an outer iteration.
        Returns True if converged (h < tolerance).
        """
        h = self.compute_h(model).detach().item()

        self.lambda_ = self.lambda_ + self.rho * h

        if self.h_old is not None and h > self.gamma * self.h_old:
            self.rho = min(self.rho * self.rho_factor, self.rho_max)

        self.h_old = h
        self.converged = h < self.h_tolerance
        return self.converged

    @override
    def reset(self) -> None:
        """Reset state to initial values."""
        self.lambda_: float = self.lambda_init
        self.rho: float = self.rho_init
        self.h_old: float | None = None
        self.converged: bool = False

    @property
    def is_converged(self) -> bool:
        """Read-only access to convergence state."""
        return self.converged
