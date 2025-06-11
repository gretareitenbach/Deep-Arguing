from typing import Callable, List, Union, cast, override

import torch
from torch import Tensor

from deeparguing.base_scores.compute_base_scores import ComputeBaseScores
from deeparguing.feature_extractor.feature_extractor import FeatureExtractor


type ActivationType = Union[Callable[[Tensor], Tensor], torch.nn.Module]

class LearnedBaseScore(ComputeBaseScores):

    def __init__(
        self,
        feature_extractors: List[FeatureExtractor],
        activation: ActivationType = lambda x: x,
        temperature: float = 1.0,
        batch_size: int | None = None,
    ):
        super(LearnedBaseScore, self).__init__()

        self.feature_extractors = torch.nn.ModuleList(feature_extractors)
        self.activation = activation
        self.temperature = temperature
        self.batch_size = batch_size

    @override
    def forward(self, nodes: Tensor) -> Tensor:

        for feature_extractor in self.feature_extractors:
            feature_extractor = cast(FeatureExtractor, feature_extractor)

            if self.batch_size == None:
                nodes = feature_extractor(nodes)
            else:
                nodes = self.split_apply_fe(
                    nodes, feature_extractor, self.batch_size
                )

        return self.activation(nodes / self.temperature)

    @override
    def plot_parameters(self):
        for feature_extractor in self.feature_extractors:
            feature_extractor = cast(FeatureExtractor, feature_extractor)
            feature_extractor.plot_parameters()

    def split_apply_fe(
        self, nodes: Tensor, feature_extractor: FeatureExtractor, batch_size: int
    ) -> Tensor:
        split_nodes = torch.split(nodes, batch_size)
        result = [feature_extractor(nodes_i) for nodes_i in split_nodes]
        result = torch.cat(result, dim=0)
        return result
