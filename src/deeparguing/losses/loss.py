from abc import ABCMeta, abstractmethod
from typing import Tuple, override

from torch import Tensor


class Loss(metaclass=ABCMeta):

    @abstractmethod
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        pass

    def __call__(self, predictions: Tensor, targets: Tensor):
        return self.forward(predictions, targets)


class LossList(Loss):

    def __init__(self, losses: list[Tuple[Loss, float]]):
        super().__init__()
        self.losses = losses

    @override
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:

        total = self.losses[0][0](predictions, targets) * self.losses[0][1]

        for i in range(1, len(self.losses)):
            loss_func = self.losses[i][0]
            weight = self.losses[i][1]
            if weight == 0:
                continue
            total += weight * loss_func(predictions, targets)

        return total
