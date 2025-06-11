from abc import ABCMeta, abstractmethod
from typing import Tuple, override

from torch import Tensor


class Cluster(metaclass=ABCMeta):

    @abstractmethod
    def cluster_data(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        pass

    def __call__(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        return self.cluster_data(X, y)


class IdentityCluster(Cluster):

    @override
    def cluster_data(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        return X, y
