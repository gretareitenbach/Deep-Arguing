import pytest
import torch

from deeparguing import GradualAACBR, SlowGradualAACBR
from deeparguing.base_scores.constant_base_score import ConstantBaseScore
from deeparguing.irrelevance_edge_weights.feature_weighted_irrelevance import \
    FeatureWeightedIrrelevance
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


ns = [5, 10, 20, 50]


@pytest.mark.parametrize("N", ns)
@pytest.mark.parametrize("use_supports", [True, False])
def test_gradual_aacbr_has_correct_logic(use_supports, N):
    """
    Tests if the implementation of GradualAACBR behaves the same as the
    implementation that is a translation of its mathemathical definition.
    """

    torch.manual_seed(42)
    X_train = torch.arange(0, N - 1).unsqueeze(1)
    y_train = torch.randint(2, (N - 1, 1))
    edge_weights = torch.randint(10, (N, N)) * 0.1

    X_default = torch.tensor([[N - 1]], dtype=torch.float32)
    y_default = torch.tensor([[0]], dtype=torch.float32)

    def edge_weights_test(attacker, target):
        attacker = attacker.to(dtype=torch.int)
        target = target.to(dtype=torch.int)
        return edge_weights[attacker, target]


    no_features = X_train.shape[-1]
    model = GradualAACBR(
        ReluSemantics(max_iters=5, epsilon=0),
        ConstantBaseScore(1),
        FeatureWeightedIrrelevance(no_features),
        edge_weights_test,
    )

    slow_model = SlowGradualAACBR(
        ReluSemantics(max_iters=5, epsilon=0),
        ConstantBaseScore(1),
        FeatureWeightedIrrelevance(no_features),
        edge_weights_test,
    )
    slow_model.fit(
        X_train,
        y_train,
        X_default,
        y_default,
        use_symmetric_attacks=True,
        use_supports=use_supports,
    )
    model.fit(
        X_train,
        y_train,
        X_default,
        y_default,
        use_symmetric_attacks=True,
        use_supports=use_supports,
    )

    assert torch.all(model.A == slow_model.A)

