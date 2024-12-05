import torch
import torch.nn.functional as F
from deeparguing.casebase_edge_weights.compute_partial_order import ComputePartialOrder
import matplotlib.pyplot as plt
from deeparguing.feature_extractor.feature_extractor import FeatureExtractor
from typing import List


class LearnedPartialOrder(ComputePartialOrder):

    def __init__(self,  feature_extractors: List[FeatureExtractor], sharpness = 1, activation = torch.sigmoid):
        super(LearnedPartialOrder, self).__init__()

        self.sharpness = sharpness
        self.feature_extractors = torch.nn.ModuleList(feature_extractors)
        self.activation = activation


    def forward(self, attacker: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        # attacker (n, no_features)
        # target (n, no_features)
        if target.ndim == 1:
            target = target.unsqueeze(0)

        if attacker.ndim == 1:
            attacker = attacker.unsqueeze(0)
        
        # TODO: Dimension reshaping to support handling different size attackers/targets
        # When dealing with newcases attacking the casebase: Need to handle this differently
        reshape = False

        if attacker.shape[1] == 1 and target.shape[0] == 1:
            attacker = attacker.squeeze()
            target = target.squeeze()
            reshape = True
        
        for feature_extractor in self.feature_extractors:

            attacker = feature_extractor(attacker)
            target = feature_extractor(target)

        if reshape:
            attacker = attacker.unsqueeze(1)
            target = target.unsqueeze(0)


        result = self.activation((attacker - target) * self.sharpness) 
        return result


    def plot_parameters(self):
        for feature_extractor in self.feature_extractors:
            feature_extractor.plot_parameters()

