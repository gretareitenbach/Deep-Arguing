from collections import OrderedDict
from typing import override

import torch
from torch import Tensor, nn

from deeparguing.feature_extractor.feature_extractor import FeatureExtractor


class MultiHeadMLPExtractor(FeatureExtractor):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        no_heads: int,
        output_activation: nn.Module | None = None,
        bias: bool = True,
        dropout: float | None = None,
    ):
        """
        Multi-head MLP extractor with fully independent heads, fully parallel.
        """
        super().__init__(output_size * no_heads)
        self.no_heads = no_heads
        self.output_size = output_size
        self.output_activation = output_activation
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.hidden_activation = nn.ReLU()

        # Use ParameterList to ensure weights move with model.to(device)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList() if bias else None

        layer_sizes = [input_size] + [h for h in hidden_sizes if h > 0] + [output_size]

        for i in range(len(layer_sizes) - 1):
            in_f = layer_sizes[i]
            out_f = layer_sizes[i + 1]

            # Shape: (no_heads, in_features, out_features)
            w = nn.Parameter(torch.empty(no_heads, in_f, out_f))
            # Kaiming Init (approximated for 3D tensors)
            nn.init.kaiming_uniform_(w, a=5**0.5)
            self.weights.append(w)

            if bias:
                # Shape: (no_heads, out_features)
                b = nn.Parameter(torch.zeros(no_heads, out_f))
                self.biases.append(b)

    @override
    def forward(self, case: Tensor) -> Tensor:
        """
        Args:
            case: (batch, input_size)
        Returns:
            Tensor: (batch, no_heads * output_size)
        """
        batch_size = case.size(0)

        # Expand input for parallel heads
        # Shape: (no_heads, batch, input_size)
        out = case.unsqueeze(0).expand(self.no_heads, batch_size, -1)

        for i, weight in enumerate(self.weights):
            # 1. Matrix Multiplication
            # (no_heads, batch, in) @ (no_heads, in, out) -> (no_heads, batch, out)
            out = torch.bmm(out, weight)

            # 2. Bias Addition
            if self.biases is not None:
                # Add (no_heads, 1, out) to broadcast over batch dimension
                out = out + self.biases[i].unsqueeze(1)

            # 3. Activation & Dropout (skip for last layer)
            if i < len(self.weights) - 1:
                out = self.hidden_activation(out)
                if self.dropout:
                    out = self.dropout(out)

        # Optional output activation
        if self.output_activation:
            out = self.output_activation(out)

        # Current shape: (no_heads, batch, output_size)
        # Target shape:  (batch, no_heads * output_size)
        out = out.permute(1, 0, 2).contiguous()

        return out.view(batch_size, -1)

    @override
    def get_output_features(self) -> int:
        return self.no_features

    @override
    def plot_parameters(self):
        print("Not plotting NN params")
