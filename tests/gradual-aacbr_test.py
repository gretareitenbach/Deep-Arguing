import pytest
import torch

from deeparguing import GradualAACBR
from deeparguing.semantics.relu_semantics import ReluSemantics

# Semantics is tested separately, we just have to have one to
# initalise gradual aa-cbr
mock_gradual_semantics = ReluSemantics(max_iters=0, epsilon=0)
mock_compute_base_score = lambda _: torch.tensor(0.5)
mock_casebase_edge_weights = lambda a, b: torch.where(
    torch.all(a >= b, dim=-1),
    1.0,
    0.0,
)
mock_irrelevance_edge_weights = lambda a, b: 1 - mock_casebase_edge_weights(a, b)

model = GradualAACBR(
    mock_gradual_semantics,
    mock_compute_base_score,
    mock_irrelevance_edge_weights,
    mock_casebase_edge_weights,
)


def test_gradual_aacbr_simple():
    """
    Test the construction of the following:
            
            ([1, 0, 0, 0], 0)   
                |          \\ 
                |att        \\ supp
                v            VV 
        ([0, 0, 0, 0], 1)      ([0, 0, 0, 0], 0)

    """

    y_train = torch.tensor([[0.0]])
    X_train = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    y_default = torch.tensor([[1.0], [0.0]])
    X_default = torch.zeros((y_default.shape[0], X_train.shape[1]))

    expected_attacks = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    model.fit(X_train, y_train, X_default, y_default, use_blockers=False)
    A = model.A
    assert torch.all(A == expected_attacks), "Simple: attacks are wrong"

    expected_supports = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    model.fit(
        X_train,
        y_train,
        X_default,
        y_default,
        use_supports=True,
        use_blockers=False,
    )
    A = model.A
    assert torch.all(A == (expected_attacks + expected_supports)), (
        "Simple: supports are wrong",
    )


def test_gradual_aacbr_minimal():
    """
    Test the construction of the following:

            ([1, 1, 0, 0], 0)
                    || supp 
                    VV
            ([1, 0, 0, 0], 0)   
                |          \\ 
                |att        \\ supp
                v            VV 
        ([0, 0, 0, 0], 1)      ([0, 0, 0, 0], 0)

    """

    y_train = torch.tensor([[0.0], [0.0]])
    X_train = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])

    y_default = torch.tensor([[1.0], [0.0]])
    X_default = torch.zeros((y_default.shape[0], X_train.shape[1]))

    expected_attacks_no_blockers = torch.tensor(
        [
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    model.fit(X_train, y_train, X_default, y_default, use_blockers=False)
    A = model.A
    assert torch.all(A == expected_attacks_no_blockers), (
        "Minimal: no blockers attacks are wrong",
    )

    expected_supports_no_blockers = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    model.fit(
        X_train,
        y_train,
        X_default,
        y_default,
        use_supports=True,
        use_blockers=False,
    )
    A = model.A
    assert torch.all(
        A == (expected_attacks_no_blockers + expected_supports_no_blockers)
    ), ("Minimal: no blockers supports are wrong",)

    expected_attacks = torch.tensor(
        [
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    model.fit(X_train, y_train, X_default, y_default, use_blockers=True)
    A = model.A
    assert torch.all(A == expected_attacks), (
        "Minimal: attacks with blockers are wrong",
    )

    expected_supports = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    model.fit(X_train, y_train, X_default, y_default, use_supports=True)
    A = model.A
    assert torch.all(A == (expected_attacks + expected_supports)), (
        "Minimal: supports with blockers are wrong",
    )


def test_gradual_aacbr_symmetric():
    """
    Test the construction of the following:

        ([1, 0, 0, 0], 1) < - > ([1, 0, 0, 0], 0)
               ||          ╲ ╱       ||
               ||supp       x att    ||supp
               vv          v v       vv
        ([0, 0, 0, 0], 1)      ([0, 0, 0, 0], 0)

    """

    y_train = torch.tensor([[0.0], [1.0]])
    X_train = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0, 0.0, 0.0]])

    y_default = torch.tensor([[1.0], [0.0]])
    X_default = torch.zeros((y_default.shape[0], X_train.shape[1]))

    expected_attacks = torch.tensor(
        [
            [0.0, -1.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    model.fit(
        X_train,
        y_train,
        X_default,
        y_default,
        use_symmetric_attacks=True,
    )
    A = model.A
    assert torch.all(A == expected_attacks), ("Symmetric: attacks are wrong",)

    expected_supports = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    model.fit(
        X_train,
        y_train,
        X_default,
        y_default,
        use_supports=True,
        use_symmetric_attacks=True,
    )
    A = model.A
    assert torch.all(A == (expected_attacks + expected_supports)), (
        "Symmetric: supports are wrong",
    )


batch_size = [1, 2, 3, 4, 5]
@pytest.mark.parametrize("batch_size", batch_size)
def test_gradual_aacbr_batching(batch_size):
    """
    Test the construction of the following:
            
            ([1, 0, 0, 0], 0)   
                |          \\ 
                |att        \\ supp
                v            VV 
        ([0, 0, 0, 0], 1)      ([0, 0, 0, 0], 0)
    """

    y_train = torch.tensor([[0.0]])
    X_train = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    y_default = torch.tensor([[1.0], [0.0]])
    X_default = torch.zeros((y_default.shape[0], X_train.shape[1]))

    expected_attacks = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    model.fit(
        X_train,
        y_train,
        X_default,
        y_default,
        use_blockers=True,
        batch_size=batch_size,
    )
    A = model.A
    assert torch.all(A == expected_attacks), "Simple: attacks are wrong"

    expected_supports = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    model.fit(
        X_train,
        y_train,
        X_default,
        y_default,
        use_supports=True,
        use_blockers=True,
        batch_size=batch_size,
    )
    A = model.A
    assert torch.all(A == (expected_attacks + expected_supports)), (
        "Simple: supports are wrong",
    )
