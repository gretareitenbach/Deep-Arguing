import torch
import matplotlib.pyplot as plt
from deeparguing.feature_extractor.feature_extractor import FeatureExtractor

class Scaler(FeatureExtractor):


    def __init__(self):
        super(Scaler, self).__init__()
        self.W = torch.nn.Parameter(torch.Tensor(1))
        torch.nn.init.normal_(self.W, mean=0, std=3)

    def forward(self, case: torch.tensor) -> torch.tensor:
        return torch.mul(case, self.W)


    def get_output_features(self) -> int:
        return None 

    def plot_parameters(self):
        print("SCALE FACTOR", self.W)
