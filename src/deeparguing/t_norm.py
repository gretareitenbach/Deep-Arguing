from abc import ABC, abstractmethod
from typing import override

import torch
from torch import Tensor


class TNorm(ABC):
    """Abstract base class for fuzzy t-norm strategies."""

    @abstractmethod
    def and_op(self, a: Tensor, b: Tensor) -> Tensor:
        pass

    @abstractmethod
    def or_op(self, a: Tensor, b: Tensor) -> Tensor:
        pass

    def not_op(self, a: Tensor) -> Tensor:
        # Standard negation (used by all three here)
        return 1 - a

    @abstractmethod
    def aggregate(self, x: Tensor, dim: int) -> Tensor:
        pass

    @abstractmethod
    def or_aggregate(self, x: Tensor, dim: int) -> Tensor:
        pass


class ProductTNorm(TNorm):

    @override
    def and_op(self, a: Tensor, b: Tensor):
        return a * b

    @override
    def or_op(self, a: Tensor, b: Tensor):
        return a + b - a * b

    @override
    def aggregate(self, x: Tensor, dim: int) -> Tensor:
        eps = 1e-10
        return torch.exp(torch.sum(torch.log(x + eps), dim=dim))

    @override
    def or_aggregate(self, x: Tensor, dim: int) -> Tensor:
        eps = 1e-10
        # 1 - ∏ (1 - x_i)
        return 1.0 - torch.exp(torch.sum(torch.log(1.0 - x + eps), dim=dim))


class GodelTNorm(TNorm):

    @override
    def and_op(self, a: Tensor, b: Tensor):
        return torch.minimum(a, b)

    @override
    def or_op(self, a: Tensor, b: Tensor):
        return torch.maximum(a, b)

    @override
    def aggregate(self, x: Tensor, dim: int) -> Tensor:
        return torch.min(x, dim=dim).values

    @override
    def or_aggregate(self, x: Tensor, dim: int) -> Tensor:
         return torch.max(x, dim=dim).values


class LukasiewiczTNorm(TNorm):

    @override
    def and_op(self, a: Tensor, b: Tensor):
        return torch.clamp(a + b - 1, min=0.0)

    @override
    def or_op(self, a: Tensor, b: Tensor):
        return torch.clamp(a + b, max=1.0)

    @override
    def aggregate(self, x: Tensor, dim: int) -> Tensor:
        result = x.select(dim, 0)
        for i in range(1, x.size(dim)):
            result = torch.clamp(result + x.select(dim, i) - 1, min=0.0)
        return result

    @override
    def or_aggregate(self, x: Tensor, dim: int) -> Tensor:
        return torch.clamp(torch.sum(x, dim=dim), max=1.0)
