from collections import OrderedDict
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
        batch_norm: bool = False,
    ):
        super(MLPExtractor, self).__init__(output_size)
        hidden_sizes = [i for i in hidden_sizes if i > 0]
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        layers = OrderedDict()
        for i in range(len(layer_sizes) - 1):
            # Add linear layer with stable name
            layers[f"linear_{i}"] = torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias)
            
            # Add activation and dropout for hidden layers only
            if i < len(layer_sizes) - 2:
                if batch_norm:
                    layers[f"batchnorm_{i}"] = torch.nn.BatchNorm1d(layer_sizes[i + 1])
                layers[f"relu_{i}"] = torch.nn.ReLU()
                if dropout:
                    layers[f"dropout_{i}"] = torch.nn.Dropout(p=dropout)
        
        # Add output activation if specified
        if output_activation:
            layers["output_activation"] = output_activation
        
        self.layers = torch.nn.Sequential(layers)    

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
