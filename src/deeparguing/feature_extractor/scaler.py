from typing import override
import torch
from torch import Tensor

from deeparguing.feature_extractor.feature_extractor import FeatureExtractor


class Scaler(FeatureExtractor):

    def __init__(
        self,
        no_features: int,
        mean: float = 0,
        std: float = 1,
        weight: float | None = None,
    ):
        super(Scaler, self).__init__(no_features)
        if weight == None:
            self.W = torch.nn.Parameter(Tensor(1))
            torch.nn.init.normal_(self.W, mean=mean, std=std)
        else:
            self.W = torch.nn.Parameter(torch.tensor(weight))

    @override
    def forward(self, case: Tensor) -> Tensor:
        return torch.mul(case, self.W)

    @override
    def get_output_features(self) -> int:
        return self.no_features

    @override
    def plot_parameters(self):
        print("SCALE FACTOR", self.W)
