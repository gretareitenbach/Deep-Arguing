import torch
from abc import abstractmethod, ABCMeta

class FeatureExtractor(torch.nn.Module, metaclass=ABCMeta):

    def __init__(self, no_features):
        super(FeatureExtractor, self).__init__()
        self.no_features = no_features

    @abstractmethod
    def forward(self, case: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_output_features(self) -> int:
        pass

    @abstractmethod
    def plot_parameters(self):
        pass
