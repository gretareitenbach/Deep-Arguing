from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Union

from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR

type FilterFunc = Callable[[Tensor], Tensor]
type RegulariserType = Union[Callable[[Any], float], "Regulariser"]


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
