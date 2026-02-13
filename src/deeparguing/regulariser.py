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

    @abstractmethod
    def step(self, model: GradualAACBR) -> bool:
        """
        Called after each outer iteration to update internal state.
        Returns True if the regulariser has converged (for early stopping).
        Regularisers without dynamic state should return True (always converged).
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset internal state to initial values.
        Called at start of training. Regularisers without state can no-op.
        """
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

    @override
    def step(self, model: GradualAACBR) -> bool:
        return True  # No dynamic state

    @override
    def reset(self) -> None:
        pass  # No state to reset


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

    @override
    def step(self, model: GradualAACBR) -> bool:
        return True  # No dynamic state

    @override
    def reset(self) -> None:
        pass  # No state to reset


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

    @override
    def step(self, model: GradualAACBR) -> bool:
        return True  # No dynamic state

    @override
    def reset(self) -> None:
        pass  # No state to reset


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
        A = A.permute(2, 0, 1)
        m = torch.matrix_exp(A * A)
        h = torch.diagonal(m, dim1=1, dim2=2).sum(1) - d
        result = h + self.alpha * h**2
        result = result.sum()

        return result

    @override
    def step(self, model: GradualAACBR) -> bool:
        return True  # No dynamic state

    @override
    def reset(self) -> None:
        pass  # No state to reset


class NOTEARSRegulariser(Regulariser):
    """
    NOTEARS regulariser using augmented Lagrangian method.

    Computes: λ * h(A) + (ρ/2) * h(A)²

    Where h(A) = trace(e^{A⊙A}) - d is the acyclicity constraint.

    After each outer iteration, call step() to update λ and ρ:
        λ ← λ + ρ * h
        ρ ← ρ * rho_factor  (if h didn't improve by factor γ)

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

        # Store initial values for reset()
        self.lambda_init = lambda_init
        self.rho_init = rho_init

        # Hyperparameters
        self.rho_max = rho_max
        self.rho_factor = rho_factor
        self.gamma = gamma
        self.h_tolerance = h_tolerance

        # Initialize state
        self.reset()

    def compute_h(self, model: GradualAACBR) -> Tensor:
        """
        Compute h(A) = trace(e^{A⊙A}) - d

        This equals 0 if and only if A is a DAG.
        """
        assert model.A is not None
        A = self.filter_func(model.A)
        d = A.shape[0]
        A = A.permute(2, 0, 1)  # (d, d, batch) -> (batch, d, d)
        m = torch.matrix_exp(A * A)
        h = torch.diagonal(m, dim1=1, dim2=2).sum(1) - d  # (batch,)
        return h.sum()

    @override
    def forward(self, model: GradualAACBR) -> Tensor:
        """Compute λ * h + (ρ/2) * h²"""
        h = self.compute_h(model)
        return self.lambda_ * h + (self.rho / 2) * h**2

    @override
    def step(self, model: GradualAACBR) -> bool:
        """
        Update λ and ρ after an outer iteration.
        Returns True if converged (h < tolerance).
        """
        h = self.compute_h(model).detach().item()

        # Update λ
        self.lambda_ = self.lambda_ + self.rho * h

        # Update ρ if constraint didn't improve enough
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

    @override
    def step(self, model: GradualAACBR) -> bool:
        """
        Call step() on all child regularisers.
        Returns True only if ALL children have converged.
        """
        all_converged = True
        for reg, _ in self.regularisers:
            converged = reg.step(model)
            all_converged = all_converged and converged
        return all_converged

    @override
    def reset(self) -> None:
        """Reset all child regularisers."""
        for reg, _ in self.regularisers:
            reg.reset()


filter_to_attacks: FilterFunc = lambda A: torch.where(A < 0, A, 0)
filter_to_supports: FilterFunc = lambda A: torch.where(A > 0, A, 0)


def threshold_adjacency(
    A: Tensor,
    threshold: float = 0.3,
    mode: str = "absolute",
) -> Tensor:
    """
    Apply thresholding to adjacency matrix to produce a cleaner DAG.

    This is the final step in NOTEARS: after training, small edge weights
    are set to zero to obtain a clean graph structure.

    Parameters
    ----------
    A : Tensor
        Adjacency matrix of shape (n, n) or (n, n, d) for batched.
    threshold : float
        Threshold value. Default 0.3 (common in NOTEARS literature).
    mode : str
        Thresholding mode:
        - "absolute": Zero out edges where |A[i,j]| < threshold
        - "relative": Zero out edges where |A[i,j]| < threshold * max(|A|)

    Returns
    -------
    Tensor
        Thresholded adjacency matrix with same shape as input.

    Examples
    --------
    >>> A = model.A.detach().clone()
    >>> A_clean = threshold_adjacency(A, threshold=0.3)
    >>> model.A.data = A_clean  # Update model in-place
    """
    if mode == "absolute":
        mask = torch.abs(A) >= threshold
    elif mode == "relative":
        max_val = torch.abs(A).max()
        mask = torch.abs(A) >= threshold * max_val
    else:
        raise ValueError(
            f"Unknown thresholding mode: {mode}. Use 'absolute' or 'relative'."
        )

    return A * mask


def apply_threshold_to_model(
    model: GradualAACBR,
    threshold: float = 0.3,
    mode: str = "absolute",
) -> None:
    """
    Apply thresholding to a trained model's adjacency matrix in-place.

    This is a convenience function for the final step in NOTEARS:
    after training converges, apply thresholding to obtain a clean DAG.

    Parameters
    ----------
    model : GradualAACBR
        Trained model with adjacency matrix.
    threshold : float
        Threshold value. Default 0.3 (common in NOTEARS literature).
    mode : str
        Thresholding mode: "absolute" or "relative".

    Raises
    ------
    Exception
        If model has not been fit (A is None).

    Examples
    --------
    >>> trainer.train(model, ...)
    >>> apply_threshold_to_model(model, threshold=0.3)
    """
    if model.A is None:
        raise Exception("Model has not been fit. Call model.fit() first.")

    with torch.no_grad():
        model.A.data = threshold_adjacency(model.A.data, threshold, mode)
