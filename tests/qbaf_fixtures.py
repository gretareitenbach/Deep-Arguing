"""Shared small synthetic EW-QBAF, reused across several test modules.

     0 <---> 1
     ^  _    ^
     | |╲s   |
    s|   4   |s
     |  a ╲| |
          -
     2  ---> 3
         a

5 casebase arguments (0-4, indices 0/2/4 are class 0, 1/3 are class 1) plus
a single default argument (index 5, appended by ``fit()``), and two new
cases (6, 7) whose edges into the casebase are given by the same lookup
table. This is the exact graph exercised by
``tests/gradual-aacbr_test.py::test_semantics``, so the resulting
adjacency/strengths are known-good.

Used by ``grae_test.py``, ``contest_test.py``, and ``batch_contest_test.py``
(the latter fits with ``ReluSemantics`` instead of ``SigmoidSemantics`` --
pass whichever semantics instance the caller needs to ``make_fitted_model``).
"""

import torch

from deeparguing import GradualAACBR
from deeparguing.semantics.gradual_semantics import GradualSemantics

EDGE_WEIGHTS = torch.tensor(
    [
        # 0, 1, 2, 3, 4, 5, 6, 7
        [0, 1, 0, 0, 0, 0, 0, 0],  # 0
        [1, 0, 0, 0, 0, 0, 0, 0],  # 1
        [1, 0, 0, 1, 0, 0, 0, 0],  # 2
        [0, 1, 0, 0, 0, 0, 0, 0],  # 3
        [1, 0, 0, 1, 0, 0, 0, 0],  # 4
        [0, 0, 0, 0, 0, 0, 0, 0],  # 5
        [0, 0, 0, 0, 0, 0, 0, 0],  # 6
        [0, 0, 0, 0, 1, 0, 0, 0],  # 7
    ],
    dtype=torch.float32,
)

BASE_SCORES = torch.tensor([0.5, 0.5, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5])

# Only one default argument exists in this casebase, so it is always the
# (only) topic/target argument.
TARGET_INDEX = 0


def base_score_fn(case: torch.Tensor) -> torch.Tensor:
    case = case.to(dtype=torch.int).squeeze(-1)
    return BASE_SCORES[case].unsqueeze(-1)


def edge_weights_fn(attacker: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    attacker = attacker.to(dtype=torch.int).squeeze(-1)
    target = target.to(dtype=torch.int).squeeze(-1)
    return EDGE_WEIGHTS[attacker.unsqueeze(1), target.unsqueeze(0)]


def irrelevance_fn(new_cases: torch.Tensor, casebase: torch.Tensor) -> torch.Tensor:
    new_cases = new_cases.unsqueeze(1).to(dtype=torch.int)
    casebase = casebase.unsqueeze(0).to(dtype=torch.int)
    r = EDGE_WEIGHTS[new_cases, casebase].squeeze(-1)
    return r.unsqueeze(-1)


def make_fitted_model(semantics: GradualSemantics) -> GradualAACBR:
    """Fit a ``GradualAACBR`` over the 5-argument casebase / single default
    argument above, using whichever semantics instance the caller built
    (e.g. ``SigmoidSemantics(max_iters=5, epsilon=0)`` or
    ``ReluSemantics(max_iters=5, epsilon=0)``)."""
    model = GradualAACBR(semantics, base_score_fn, irrelevance_fn, edge_weights_fn)
    model.use_symmetric_attacks = True
    model.use_supports = True
    model.use_blockers = False

    X_train = torch.tensor([[0], [1], [2], [3], [4]])
    y_train = torch.tensor([[0], [1], [0], [1], [0]])
    X_default = torch.tensor([[5]], dtype=torch.float32)
    y_default = torch.tensor([[5]], dtype=torch.float32)
    model.fit(X_train, y_train, X_default, y_default)
    return model
