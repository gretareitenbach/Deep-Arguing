from abc import ABC, abstractmethod
from typing import Any


class NewCasesAttacks(ABC):

    @abstractmethod
    def forward(self, base_scores, A, new_cases, new_cases_attacks, compute_base_score, gradual_semantics):
        pass

    def __call__(self, base_scores, A, new_cases, new_cases_attacks, compute_base_score, gradual_semantics) -> Any:
        return self.forward(base_scores, A, new_cases, new_cases_attacks, compute_base_score, gradual_semantics)