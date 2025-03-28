from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch


class Cluster(metaclass=ABCMeta):

    @abstractmethod
    def cluster_data(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def __call__(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cluster_data(X, y)


class IdentityCluster(Cluster):

    def cluster_data(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return X, y
