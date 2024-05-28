import torch
from .compute_edge_weights import ComputeEdgeWeights

class IdentityEdgeWeights(ComputeEdgeWeights):

    def forward(self, A: torch.Tensor, nodes: torch.Tensor):
        return A
    
    def plot_parameters(self):
        print("No Edge Parameters")
