from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Union

from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR

type FilterFunc = Callable[[Tensor], Tensor]
type CriterionType = Union[Callable[[Any, Any, Any], float], "Criterion"]


class Criterion(metaclass=ABCMeta):

    @abstractmethod
    def forward(self, model: GradualAACBR, predictions: Tensor, targets: Tensor) -> Tensor:
        pass

    def step(self, model: GradualAACBR) -> bool:
        """
        Called after each outer iteration to update internal state.
        Returns True if the criterion has converged (for early stopping).
        Criteria without dynamic state should return True (always converged).
        """
        return True

    def reset(self) -> None:
        """
        Reset internal state to initial values.
        Called at start of training. Criteria without state can no-op.
        """
        pass

    def __call__(self, model: GradualAACBR, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(model, predictions, targets)
