from abc import ABCMeta, abstractmethod
from typing import Optional

from numpy.typing import ArrayLike


class Model(metaclass=ABCMeta):

    def __init__(self):
        super(Model, self).__init__()

    @abstractmethod
    def forward(self, input: ArrayLike):
        pass

    @abstractmethod
    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_default: Optional[ArrayLike] = None,
        y_default: Optional[ArrayLike] = None,
        batch_size: Optional[int] = None,
    ):
        pass

    def __call__(self, input: ArrayLike):
        return self.forward(input)
