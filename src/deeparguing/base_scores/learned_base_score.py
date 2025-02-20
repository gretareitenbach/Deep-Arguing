import torch
import torch.nn.functional as F
from deeparguing.base_scores.compute_base_scores import ComputeBaseScores
import matplotlib.pyplot as plt
from deeparguing.feature_extractor.feature_extractor import FeatureExtractor
from typing import List


class LearnedBaseScore(ComputeBaseScores):

    def __init__(self,  feature_extractors: List[FeatureExtractor], activation=lambda x: x, temperature = 1):
        super(LearnedBaseScore, self).__init__()

        self.feature_extractors = torch.nn.ModuleList(feature_extractors)
        self.activation = activation
        self.temperature = temperature


    def forward(self, features: torch.Tensor) -> torch.Tensor:

        for feature_extractor in self.feature_extractors:
            features = feature_extractor(features)
        return self.activation(features/self.temperature) 


    def plot_parameters(self):
        for feature_extractor in self.feature_extractors:
            feature_extractor.plot_parameters()

