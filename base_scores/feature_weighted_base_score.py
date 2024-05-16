import torch


class FeatureWeightedBaseScore(torch.nn.Module):

    def __init__(self, no_features):
        super(FeatureWeightedBaseScore, self).__init__()
        # TODO: Consider other methods of initialising weights
        self.W = torch.nn.Parameter(torch.Tensor(no_features))
        torch.nn.init.normal_(self.W)

    def forward(self, nodes):
        return torch.sigmoid(torch.matmul(nodes, self.W))
