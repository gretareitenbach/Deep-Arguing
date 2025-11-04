from typing import cast, override

import torch
from torch import Tensor

from deeparguing.casebase_edge_weights.compute_partial_order import (
    CompareCases, ComputePartialOrder)
from deeparguing.feature_extractor.feature_extractor import FeatureExtractor


class LearnedPartialOrder(ComputePartialOrder):

    def __init__(
        self,
        feature_extractors: list[FeatureExtractor],
        comparison_func: CompareCases,
        batch_size: int | None = None,
    ):
        super(LearnedPartialOrder, self).__init__()

        self.feature_extractors = torch.nn.ModuleList(feature_extractors)
        self.comparison_func = comparison_func
        self.batch_size = batch_size

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        for feature_extractor in self.feature_extractors:
            feature_extractor = cast(FeatureExtractor, feature_extractor)
            if self.batch_size is None:
                x = feature_extractor(x)
            else:
                x = self.split_apply_fe(x, feature_extractor, self.batch_size)
        return x

    @override
    def forward(self, attacker: Tensor, target: Tensor) -> Tensor:

        # attacker (n, no_features)
        # target (n, no_features)
        if target.ndim == 1:
            target = target.unsqueeze(0)
        if attacker.ndim == 1:
            attacker = attacker.unsqueeze(0)

        # If they are the same object, avoid redundant feature extraction
        if attacker is target:
            attacker_emb = target_emb = self._extract_features(attacker)
        else:
            attacker_emb = self._extract_features(attacker)
            target_emb = self._extract_features(target)

        attacker_emb = attacker_emb.unsqueeze(1)
        target_emb = target_emb.unsqueeze(0)

        result = self.comparison_func(attacker_emb, target_emb).squeeze()

        return result

    @override
    def plot_parameters(self):
        for feature_extractor in self.feature_extractors:
            feature_extractor.plot_parameters()

    def split_apply_fe(
        self, cases: Tensor, feature_extractor: FeatureExtractor, batch_size: int
    ) -> Tensor:
        split_cases = torch.split(cases, batch_size)
        result = [feature_extractor(cases_i) for cases_i in split_cases]
        result = torch.cat(result, dim=0)
        return result
