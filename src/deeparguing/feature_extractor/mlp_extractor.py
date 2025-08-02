from typing import override

import torch
from torch import Tensor

from deeparguing.feature_extractor.feature_extractor import FeatureExtractor


class MLPExtractor(FeatureExtractor):

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        output_activation: torch.nn.Module | None = None,
        bias: bool = True,
        dropout: None | float = None,
    ):
        super(MLPExtractor, self).__init__(output_size)

        hidden_sizes = [i for i in hidden_sizes if i > 0]

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias))
            if i < len(layer_sizes) - 2:
                self.layers.append(torch.nn.ReLU())
                if dropout:
                    self.layers.append(torch.nn.Dropout(p=dropout))

        if output_activation:
            self.layers.append(output_activation)

    @override
    def forward(self, case: Tensor) -> Tensor:
        for layer in self.layers:
            case = layer(case)

        case = case.squeeze()

        return case

    @override
    def get_output_features(self) -> int:
        return self.no_features

    @override
    def plot_parameters(self):
        print("Not plotting NN params")
