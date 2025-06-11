from typing import Callable, override

import matplotlib.pyplot as plt
import torch
import torch.nn.init as init
from torch import Tensor

from deeparguing.feature_extractor.feature_extractor import FeatureExtractor


class FeatureWeightedExtractor(FeatureExtractor):

    def __init__(
        self,
        no_features: int,
        inital_weights: Tensor | None = None,
        initialisation_method: Callable[
            [Tensor], Tensor
        ] = lambda x: init.xavier_uniform_(x.unsqueeze(1)),
    ):
        super(FeatureWeightedExtractor, self).__init__(no_features)
        if inital_weights is None:
            self.W = torch.nn.Parameter(Tensor(no_features))
            initialisation_method(self.W)
        else:
            self.W = torch.nn.Parameter(inital_weights)

    @override
    def forward(self, case: Tensor) -> Tensor:
        W = self.W.squeeze()
        return torch.matmul(case, W)

    @override
    def get_output_features(self) -> int:
        return self.no_features

    @override
    def plot_parameters(self):
        weights = self.W.squeeze().detach().cpu().numpy()
        plt.figure(figsize=(20, 5))
        plt.bar(range(len(weights)), weights)
        for i, value in enumerate(weights):
            plt.text(
                i,
                value + (0.1 * (-1 if value <= 0 else 1)),
                str(round(value, 3)),
                ha="center",
                fontsize=6,
            )
        plt.xlabel("Features")
        plt.ylabel("Weights")
        plt.title("Feature Attribution Weights")
        plt.show()
