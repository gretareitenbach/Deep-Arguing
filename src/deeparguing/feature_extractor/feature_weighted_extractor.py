import torch
import matplotlib.pyplot as plt
from deeparguing.feature_extractor.feature_extractor import FeatureExtractor

class FeatureWeightedExtractor(FeatureExtractor):


    def __init__(self, no_features):
        super(FeatureWeightedExtractor, self).__init__(no_features)
        self.W = torch.nn.Parameter(torch.Tensor(no_features))
        torch.nn.init.normal_(self.W)

    def forward(self, case: torch.Tensor) -> torch.Tensor:
        return torch.matmul(case, self.W)


    def get_output_features(self) -> int:
        return self.no_features

    def plot_parameters(self):
        weights = self.W.detach().cpu().numpy()
        plt.figure(figsize=(20, 5))
        plt.bar(range(len(weights)), weights)
        for i, value in enumerate(weights):
            plt.text(i, value + (0.1 * (-1 if value <= 0 else 1)),
                     str(round(value, 3)), ha='center', fontsize=6)
        plt.xlabel('Features')
        plt.ylabel('Weights')
        plt.title('Feature Attribution Weights')
        plt.show()
