import torch
from .compute_edge_weights import ComputeEdgeWeights
import matplotlib.pyplot as plt

class LearnedEdgeWeights(ComputeEdgeWeights):

    def __init__(self, no_nodes: int):
        super(LearnedEdgeWeights, self).__init__()
        #TODO: Consider other initialisation methods
        self._W = torch.nn.Parameter(torch.Tensor(no_nodes, no_nodes))
        torch.nn.init.normal_(self._W)

    @property
    def W(self):
        return torch.abs(self._W)

    def forward(self, A: torch.Tensor, nodes: torch.Tensor):
        return torch.mul(A, self.W)
    
    def plot_parameters(self):
        edge_weights = self.W.detach().numpy()
        plt.figure(figsize=(20, 5))
        plt.imshow(edge_weights, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Value')
        plt.xlabel('Nodes')
        plt.ylabel('Nodes')
        plt.title('Edges')
        plt.show()