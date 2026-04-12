from typing import override

from torch import Tensor
from torch.nn import *

from deeparguing.criterion.criterion import Criterion


class TorchLoss(Criterion):

    def __init__(self, class_name: str, **kwargs):
        self.loss_func = globals()[class_name](**kwargs)

    @override
    def forward(self, model, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.loss_func(predictions, targets)
