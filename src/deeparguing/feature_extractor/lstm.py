from typing import Optional, override

import torch
import torch.nn as nn
from torch import Tensor

from deeparguing.feature_extractor.feature_extractor import FeatureExtractor

class LSTMFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_features: int = 1,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        weights_path: Optional[str] = None,
        freeze_weights: bool = False,
    ):
        super(LSTMFeatureExtractor, self).__init__(no_features=output_features)
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc_out = nn.Linear(lstm_out_dim, output_features)

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

        if freeze_weights:
            for p in self.parameters():
                p.requires_grad = False

    @override
    def forward(self, case: Tensor) -> Tensor:
        if case.ndim == 2:
            case = case.unsqueeze(-1)
            
        _, (h_n, _) = self.lstm(case)
        
        if self.lstm.bidirectional:
            h = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=-1)
        else:
            h = h_n[-1, :, :]
            
        x = self.fc_out(h)
        return x.squeeze(-1)

    @override
    def get_output_features(self) -> int:
        return self.fc_out.out_features

    @override
    def plot_parameters(self):
        print("LSTM PARAMETERS ARE NOT PRINTED")
        pass
