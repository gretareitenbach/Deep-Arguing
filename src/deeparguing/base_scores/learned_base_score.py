import torch
import torch.nn.functional as F
from deeparguing.base_scores.compute_base_scores import ComputeBaseScores
import matplotlib.pyplot as plt
from deeparguing.feature_extractor.feature_extractor import FeatureExtractor


class LearnedBaseScore(ComputeBaseScores):

    def __init__(self,  feature_extractor: FeatureExtractor):
        super(LearnedBaseScore, self).__init__()

        self.feature_extractor = feature_extractor
        self.W = torch.nn.Parameter(torch.Tensor(feature_extractor.get_output_features()))
        torch.nn.init.normal_(self.W)


    def forward(self, nodes: torch.Tensor) -> torch.Tensor:

        features = self.feature_extractor(nodes)
        result = torch.sigmoid(torch.matmul(features, self.W)) 
        return result


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

        print("FEATURE EXTRACTOR PARAMETERS:")
        self.feature_extractor.plot_parameters()

