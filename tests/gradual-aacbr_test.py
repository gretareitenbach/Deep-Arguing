import pytest
import torch

from deeparguing import GradualAACBR, SlowGradualAACBR
from deeparguing.semantics.sigmoid_semantics import SigmoidSemantics

# Semantics is tested separately, we just have to have one to
# initalise/test gradual aa-cbr
mock_gradual_semantics = SigmoidSemantics(max_iters=0, epsilon=0)
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

    assert torch.all(
        model.A == slow_model.A
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
        case = case.to(dtype=torch.int)
        return bs[case].squeeze()

    def edge_weights_test(attacker, target):
        attacker = attacker.to(dtype=torch.int)
        target = target.to(dtype=torch.int)
        return edge_weights[attacker, target]

    def irrelevance_test(new_cases, casebase):
        new_cases = new_cases.unsqueeze(1).to(dtype=torch.int)  # (B, 1, no_features)
        casebase = casebase.unsqueeze(0).to(dtype=torch.int)  # (1, n, no_features)
        r = edge_weights[new_cases, casebase].squeeze()
        return r

    semantics = SigmoidSemantics(max_iters=5, epsilon=0)
    model = GradualAACBR(
        semantics,
        base_score_test,
        irrelevance_test,
        edge_weights_test,
    )

    model.fit(
        X_train,
        y_train,
        X_default,
        y_default,
        use_symmetric_attacks=True,
        use_supports=True,
        use_blockers=False,  # We are just testing the forward function so blockers can be off
    )

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


if __name__ == "__main__":
    test_semantics()
