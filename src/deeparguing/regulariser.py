from abc import ABCMeta, abstractmethod

import torch

from deeparguing.gradual_aacbr import GradualAACBR


class Regulariser(metaclass=ABCMeta):

    @abstractmethod
    def forward(self, model:GradualAACBR) -> torch.Tensor:
        pass
    
    def __call__(self, model) -> torch.Tensor:
        return self.forward(model)


class SparsityRegulariser(Regulariser):

    def __init__(self, filter_func=lambda A: A):
        super().__init__()
        self.filter_func = filter_func

    def forward(self, model: GradualAACBR):
        assert model.A != None
        A = self.filter_func(model.A)
        result = torch.sum(torch.abs(A))
        result = result / len(model.A)
        return result


class CommunityPreservationRegulariser(Regulariser):

    def __init__(self, filter_func=lambda A: A):
        super().__init__()
        self.filter_func = filter_func

    def forward(self, model: GradualAACBR):
        assert model.A != None
        A = self.filter_func(model.A)
        A = torch.abs(A)  # This regulariser expects values between 0 and 1
        return torch.sum(torch.svd(A).S)


class ConnectivityRegulariser(Regulariser):

    def __init__(self, filter_func=lambda A: A, epsilon=1e-8):
        super().__init__()
        self.filter_func = filter_func
        self.epsilon = epsilon

    def forward(self, model: GradualAACBR):
        assert model.A != None
        A = self.filter_func(model.A)
        A = torch.abs(A)  # This regulariser expects values between 0 and 1
        A = torch.sum(A, dim=1) + self.epsilon
        result = -torch.sum(torch.log(A))
        result = result / len(model.A)
        return result

class RegulariserList(Regulariser):

    def __init__(self, regularisers):
        super().__init__()
        self.regularisers = regularisers

    def forward(self, model: GradualAACBR):
        assert model.A != None

        total = torch.tensor(0.0, device = model.device)

        for reg_func, weight in self.regularisers:
            total += weight * reg_func(model)

        return total


filter_to_attacks = lambda A: torch.where(A < 0, A, 0)
filter_to_supports = lambda A: torch.where(A > 0, A, 0)

