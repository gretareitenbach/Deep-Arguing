from abc import ABCMeta, abstractmethod
from torch import Tensor

from deeparguing import GradualAACBR

class Trainer(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(
        self,
        model: GradualAACBR,
        X_casebase: Tensor,
        y_casebase: Tensor,
        X_new_cases: Tensor,
        y_new_cases: Tensor,
        X_default: Tensor,
        y_default: Tensor,
        disable_tqdm: bool = False,
        X_val: Tensor | None = None,
        y_val: Tensor | None = None,
    ) -> tuple[float, float]:
        pass
