import pytest
import torch

from deeparguing import GradualAACBR, SlowGradualAACBR
from deeparguing.semantics.sigmoid_semantics import SigmoidSemantics

# Semantics is tested separately, we just have to have one to
# initalise/test gradual aa-cbr
mock_gradual_semantics = SigmoidSemantics(max_iters=0, epsilon=0)
mock_compute_base_score = lambda x: torch.full((x.shape[0], 1), 0.5)
mock_casebase_edge_weights = lambda a, b: torch.where(
    torch.all(a.unsqueeze(1) >= b.unsqueeze(0), dim=-1),
    1.0,
    0.0,
).unsqueeze(-1)  # (n, n) -> (n, n, 1)
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
            [[0.0], [-1.0], [0.0]],
            [[0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0]],
        ]
    )
    model.use_blockers = False
    model.use_supports = False
    model.fit(X_train, y_train, X_default, y_default)
    A = model.A
    assert torch.all(A == expected_attacks), "Simple: attacks are wrong"

    expected_supports = torch.tensor(
        [
            [[0.0], [0.0], [1.0]],
            [[0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0]],
        ]
    )

    model.use_blockers = False
    model.use_supports = True
    model.fit(X_train, y_train, X_default, y_default)
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
            [[0.0], [0.0], [-1.0], [0.0]],
            [[0.0], [0.0], [-1.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
        ]
    )
    model.use_blockers = False
    model.use_supports = False
    model.fit(X_train, y_train, X_default, y_default)
    A = model.A
    assert torch.all(A == expected_attacks_no_blockers), (
        "Minimal: no blockers attacks are wrong",
    )

    expected_supports_no_blockers = torch.tensor(
        [
            [[0.0], [0.0], [0.0], [1.0]],
            [[1.0], [0.0], [0.0], [1.0]],
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
        ]
    )

    model.use_supports = True
    model.use_blockers = False
    model.fit(X_train, y_train, X_default, y_default)
    A = model.A
    assert torch.all(
        A == (expected_attacks_no_blockers + expected_supports_no_blockers)
    ), ("Minimal: no blockers supports are wrong",)

    expected_attacks = torch.tensor(
        [
            [[0.0], [0.0], [-1.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
        ]
    )

    model.use_blockers = True
    model.use_supports = False
    model.fit(X_train, y_train, X_default, y_default)
    A = model.A
    assert torch.allclose(A, expected_attacks, atol=1e-9), (
        "Minimal: attacks with blockers are wrong",
    )

    expected_supports = torch.tensor(
        [
            [[0.0], [0.0], [0.0], [1.0]],
            [[1.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
        ]
    )
    model.use_blockers = True
    model.use_supports = True
    model.fit(X_train, y_train, X_default, y_default)
    A = model.A
    assert torch.allclose(A, expected_attacks + expected_supports, atol=1e-9), (
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
            [[0.0], [-1.0], [-1.0], [0.0]],
            [[-1.0], [0.0], [0.0], [-1.0]],
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
        ]
    )
    model.use_supports = False
    model.use_symmetric_attacks = True
    model.fit(X_train, y_train, X_default, y_default)
    A = model.A
    assert torch.all(A == expected_attacks), ("Symmetric: attacks are wrong",)

    expected_supports = torch.tensor(
        [
            [[0.0], [0.0], [0.0], [1.0]],
            [[0.0], [0.0], [1.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0]],
        ]
    )

    model.use_supports = True
    model.use_symmetric_attacks = True
    model.fit(X_train, y_train, X_default, y_default)
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
            [[0.0], [-1.0], [0.0]],
            [[0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0]],
        ]
    )
    model.use_supports = False
    model.use_blockers = True
    model.fit(
        X_train,
        y_train,
        X_default,
        y_default,
        batch_size=batch_size,
    )
    A = model.A
    assert torch.all(A == expected_attacks), "Simple: attacks are wrong"

    expected_supports = torch.tensor(
        [
            [[0.0], [0.0], [1.0]],
            [[0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0]],
        ]
    )

    model.use_supports = True
    model.use_blockers = True
    model.fit(
        X_train,
        y_train,
        X_default,
        y_default,
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
        # Handle both GradualAACBR (2D input: n x features) and
        # SlowGradualAACBR (3D input: n x n x features) calling conventions
        if attacker.ndim == 2:
            # Fast model: attacker/target are (n, features)
            # Return (n, n) pairwise edge weights
            attacker = attacker.to(dtype=torch.int).squeeze(-1)  # (n,)
            target = target.to(dtype=torch.int).squeeze(-1)  # (n,)
            return edge_weights[attacker.unsqueeze(1), target.unsqueeze(0)]  # (n, n)
        else:
            # Slow model: attacker/target are (n, n, features)
            # Return scalar per pair, shape matches input grid
            attacker = attacker.to(dtype=torch.int)
            target = target.to(dtype=torch.int)
            # Extract first feature (the index value)
            attacker_idx = attacker[..., 0]  # (n, n)
            target_idx = target[..., 0]  # (n, n)
            return edge_weights[attacker_idx, target_idx]  # (n, n)

    model = GradualAACBR(
        mock_gradual_semantics,
        mock_compute_base_score,
        mock_irrelevance_edge_weights,
        edge_weights_test,
    )

    slow_model = SlowGradualAACBR(
        mock_gradual_semantics,
        mock_compute_base_score,
        mock_irrelevance_edge_weights,
        edge_weights_test,
    )
    slow_model.use_symmetric_attacks = True
    slow_model.use_supports = use_supports
    slow_model.fit(X_train, y_train, X_default, y_default)

    model.use_symmetric_attacks = True
    model.use_supports = use_supports
    model.fit(X_train, y_train, X_default, y_default)

    # Squeeze model.A to compare with slow_model.A (which is 2D for d=1)
    assert torch.allclose(
        model.A.squeeze(-1), slow_model.A, atol=1e-9
    ), "Logic: Gradual AA-CBR's adjacency matrix does not match Slow Gradual AA-CBR"


def test_semantics():
    """
         a
     0 <---> 1
     ^  _    ^
     | |╲s   |              default (5) unattacked and does not attack
    s|   4   |s
     |  a ╲| |              N1 (6) attacks nothing
          -                 N2 (7) attacks 4
     2  ---> 3
         a

    This test constructs a gradual AA-CBR framework and ensures the final semantics
    computed is as expected. We check the final semantics according to values we expect
    from the semantics tests.

    """
    X_train = torch.tensor(
        [
            [0],
            [1],
            [2],
            [3],
            [4],
        ]
    )
    y_train = torch.tensor(
        [
            [0],  # 0
            [1],  # 1
            [0],  # 2
            [1],  # 3
            [0],  # 4
        ]
    )
    edge_weights = torch.tensor(
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

    X_default = torch.tensor([[5]], dtype=torch.float32)
    y_default = torch.tensor([[5]], dtype=torch.float32)

    bs = torch.tensor([0.5, 0.5, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5])

    def base_score_test(case):
        # case is (n, 1) tensor, want to return (n, d) = (n, 1)
        case = case.to(dtype=torch.int).squeeze(-1)  # (n, 1) -> (n,)
        return bs[case].unsqueeze(-1)  # (n,) -> (n, 1)

    def edge_weights_test(attacker, target):
        # attacker and target are (n, 1) tensors
        # We need to return (n, n) pairwise edge weights
        attacker = attacker.to(dtype=torch.int).squeeze(-1)  # (n,)
        target = target.to(dtype=torch.int).squeeze(-1)  # (n,)
        # Compute pairwise: edge_weights[attacker[i], target[j]] for all i,j
        return edge_weights[attacker.unsqueeze(1), target.unsqueeze(0)]  # (n, n)

    def irrelevance_test(new_cases, casebase):
        new_cases = new_cases.unsqueeze(1).to(dtype=torch.int)  # (B, 1, no_features)
        casebase = casebase.unsqueeze(0).to(dtype=torch.int)  # (1, n, no_features)
        r = edge_weights[new_cases, casebase].squeeze(-1)  # Squeeze features, not d
        return r.unsqueeze(-1)  # Add dimension for d=1

    semantics = SigmoidSemantics(max_iters=5, epsilon=0)
    model = GradualAACBR(
        semantics,
        base_score_test,
        irrelevance_test,
        edge_weights_test,
    )

    model.use_symmetric_attacks=True
    model.use_supports=True
    # We are just testing the forward function so blockers can be off
    model.use_blockers=False  
    model.fit(X_train, y_train, X_default, y_default)

    result = model(torch.tensor([[6], [7]]), return_all_strengths=True)
    expected_result = torch.tensor(
        [
            [0.7647, 0.4215, 0.7000, 0.4468, 0.9000, 0.5000],
            [0.7536, 0.4275, 0.7000, 0.4604, 0.8452, 0.5000],
        ]
    )
    assert torch.allclose(
        expected_result, result, atol=1e-4
    ), "Semantics: Semantics computation is incorrect"
