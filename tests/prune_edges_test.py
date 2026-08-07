import torch

from deeparguing.contest.scripts.prune_edges import DEFAULT_THRESHOLD, prune_edges


def _sample_A() -> torch.Tensor:
    # Mix of strong/weak attacks (negative) and supports (positive), plus
    # entries already at zero (non-edges).
    return torch.tensor([0.5, -0.5, 0.09, -0.09, 0.1, -0.1, 0.0])


def test_prune_edges_zeroes_weak_edges_of_either_sign():
    A = _sample_A()
    result = prune_edges(A, threshold=0.1)

    assert torch.equal(
        result.pruned_A, torch.tensor([0.5, -0.5, 0.0, 0.0, 0.1, -0.1, 0.0])
    )


def test_prune_edges_threshold_is_exclusive():
    """An edge with abs(weight) exactly equal to the threshold survives --
    only weights strictly below it are pruned."""
    A = torch.tensor([0.1, -0.1])
    result = prune_edges(A, threshold=0.1)
    assert torch.equal(result.pruned_A, A)


def test_prune_edges_reports_correct_counts():
    A = _sample_A()
    result = prune_edges(A, threshold=0.1)

    assert result.num_edges_before == 6  # every nonzero entry
    assert result.num_edges_after == 4   # the two 0.09/-0.09 entries pruned
    assert result.num_pruned == 2
    assert result.threshold == 0.1


def test_prune_edges_does_not_mutate_input():
    A = _sample_A()
    original = A.clone()
    prune_edges(A, threshold=0.1)
    assert torch.equal(A, original)


def test_prune_edges_default_threshold():
    A = _sample_A()
    result = prune_edges(A)
    assert result.threshold == DEFAULT_THRESHOLD


def test_prune_edges_zero_threshold_prunes_nothing():
    A = _sample_A()
    result = prune_edges(A, threshold=0.0)
    assert torch.equal(result.pruned_A, A)
    assert result.num_pruned == 0
