import torch
import torch.nn.functional as F
from .compute_partial_order import ComputePartialOrder
import matplotlib.pyplot as plt
from feature_extractor.feature_extractor import FeatureExtractor


class LearnedPartialOrder(ComputePartialOrder):

    def __init__(self,  feature_extractor: FeatureExtractor, sharpness = 1):
        super(LearnedPartialOrder, self).__init__()

        self.sharpness = sharpness
        self.feature_extractor = feature_extractor
        self.W = torch.nn.Parameter(torch.Tensor(feature_extractor.get_output_features()))
        torch.nn.init.normal_(self.W)


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

        attacker = self.feature_extractor(attacker)
        target = self.feature_extractor(target)

        if reshape:
            attacker = attacker.unsqueeze(1)
            target = target.unsqueeze(0)

        attacker_score = torch.matmul(attacker, self.W)
        target_score = torch.matmul(target, self.W)

        result = torch.sigmoid((attacker_score - target_score) * self.sharpness) 
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

