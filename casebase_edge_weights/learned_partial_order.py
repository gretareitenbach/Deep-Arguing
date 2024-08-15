import torch
import torch.nn.functional as F
from .compute_partial_order import ComputePartialOrder
import matplotlib.pyplot as plt


class LearnedPartialOrder(ComputePartialOrder):

    def __init__(self, no_features, no_hidden, sharpness = 1):
        super(LearnedPartialOrder, self).__init__()

        # TODO: Consider other methods of initialising weights
        self.sharpness = sharpness

        self.fc1 = torch.nn.Linear(no_features * 2, 1)
        # self.fc2 = torch.nn.Linear(no_hidden, 1)




    def forward(self, attacker: torch.Tensor, target: torch.Tensor, reshape=False) -> torch.Tensor:

        # attacker (n, no_features)
        # target (n, no_features)

        if target.ndim == 1:
            target = target.unsqueeze(0)

        if attacker.ndim == 1:
            attacker = attacker.unsqueeze(0)

        concat = torch.cat((attacker, target), dim=1)
        out = F.sigmoid(self.fc1(concat))
        # out = F.sigmoid(self.fc2(out))

        return torch.sigmoid(out * self.sharpness).squeeze() 


    def plot_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print (name, param.data)
