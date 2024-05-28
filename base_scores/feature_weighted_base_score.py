import torch
from .compute_base_scores import ComputeBaseScores
import matplotlib.pyplot as plt


class FeatureWeightedBaseScore(ComputeBaseScores):

    def __init__(self, no_features):
        super(FeatureWeightedBaseScore, self).__init__()
        # TODO: Consider other methods of initialising weights
        self.W = torch.nn.Parameter(torch.Tensor(no_features))
        torch.nn.init.normal_(self.W)

    def forward(self, nodes):
        return torch.sigmoid(torch.matmul(nodes, self.W))
    
    def plot_parameters(self):
        weights = self.W.detach().numpy()
        plt.figure(figsize=(20, 5))
        plt.bar(range(len(weights)), weights)
        for i, value in enumerate(weights):
            plt.text(i, value + (0.1 * (-1 if value <= 0 else 1)), str(round(value, 3)), ha='center', fontsize=6)
        plt.xlabel('Features')
        plt.ylabel('Weights')
        plt.title('Feature Attribution Weights')
        plt.show()
