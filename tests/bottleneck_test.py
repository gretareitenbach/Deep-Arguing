import torch

from deeparguing import GradualAACBR
from deeparguing.counterfactuals.bottleneck import (
    find_and_escape_bottleneck,
    find_bottleneck,
    select_bottleneck_edges,
)
from deeparguing.counterfactuals.contest import MARGIN, THRESHOLD, ContestResult, contest
from deeparguing.counterfactuals.grae import compute_grae
from deeparguing.semantics.relu_semantics import ReluSemantics

# ---------------------------------------------------------------------------
# Small synthetic ReLU EW-QBAF, hand-built (bypassing fit(), same approach as
# grae_test.py's property-based fixtures) so a hard-ReLU bottleneck sits
# exactly where these tests need it:
#
#   node 0: target/default argument (the only one -- single-topic-argument
#           setting, same convention as contest_test.py's fixture)
#   node 1: "bottleneck" -- attacked by 2 and 3, supports 0
#   node 2: weak attacker of 1   (base score 0.3 -> low leverage)
#   node 3: strong attacker of 1 (base score 0.9 -> high leverage)
#
# The new case attacks node 1 directly (weight 1.0), which combined with
# 2 and 3's attacks pins node 1's own strength at exactly 0 (its ReLU is
# saturated). node 0's own base score is <= 0, and node 1 is node 0's only
# attacker, so with node 1 at 0, node 0's own final ReLU is *also* saturated
# at exactly 0 -- and since the aggregation step is dense (every entry of A
# has a well-defined one-hop gradient contribution equal to its source
# node's strength, whether or not that entry is currently nonzero), a
# uniformly-zero gradient requires exactly this: target_class's own strength
# pinned at 0, which zeroes the derivative through its outer ReLU and kills
# every path uniformly. Escaping node 1 (raising its strength above 0) then
# cascades forward and un-sticks node 0 too.
# ---------------------------------------------------------------------------

_TARGET = 0
_BOTTLENECK = 1
_WEAK_ATTACKER = 2
_STRONG_ATTACKER = 3
_N = 4

# index 4 is the new case's own (otherwise unused) base score row
_BASE_SCORES = torch.tensor([-0.5, 0.5, 0.3, 0.9, 0.5])
_HOPELESS_BASE_SCORES = torch.tensor([-0.5, 0.5, -0.5, -0.5, 0.5])

_SAMPLE = torch.tensor([[4.0]])


def _edge_weights_fn(attacker: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("model.A is set directly; fit() is never called.")


def _irrelevance_fn(new_cases: torch.Tensor, casebase: torch.Tensor) -> torch.Tensor:
    """The new case attacks only the bottleneck node, weight 1.0."""
    batch = new_cases.shape[0]
    n = casebase.shape[0]
    row = torch.zeros(n)
    row[_BOTTLENECK] = 1.0
    return row.unsqueeze(0).expand(batch, -1).unsqueeze(-1)


def _make_model(
    target_support: float,
    base_scores: torch.Tensor = _BASE_SCORES,
    max_iters: int = 10,
) -> GradualAACBR:
    def base_score_fn(x: torch.Tensor) -> torch.Tensor:
        idx = x.to(dtype=torch.long).squeeze(-1)
        return base_scores[idx].unsqueeze(-1)

    semantics = ReluSemantics(max_iters=max_iters, epsilon=0)
    model = GradualAACBR(semantics, base_score_fn, _irrelevance_fn, _edge_weights_fn)

    A = torch.zeros(_N, _N, 1)
    A[_WEAK_ATTACKER, _BOTTLENECK, 0] = -0.4
    A[_STRONG_ATTACKER, _BOTTLENECK, 0] = -0.9
    A[_BOTTLENECK, _TARGET, 0] = target_support

    model.A = A
    model.X_train = torch.arange(_N).unsqueeze(-1).float()
    model.default_indexes = torch.tensor([_TARGET])
    return model


def _flat_index(source: int, target: int, dim: int = 0, n: int = _N, d: int = 1) -> int:
    return source * (n * d) + target * d + dim


# ---------------------------------------------------------------------------
# find_bottleneck / select_bottleneck_edges
# ---------------------------------------------------------------------------


def test_find_bottleneck_locates_the_saturated_node():
    model = _make_model(target_support=0.2)

    result = find_bottleneck(model, _SAMPLE, _TARGET)

    assert result is not None
    bottleneck_node, node_strengths = result
    assert bottleneck_node == _BOTTLENECK
    assert node_strengths[_BOTTLENECK].item() == 0.0


def test_select_bottleneck_edges_prefers_the_higher_leverage_source():
    model = _make_model(target_support=0.2)
    bottleneck_node, node_strengths = find_bottleneck(model, _SAMPLE, _TARGET)

    edge_indices = select_bottleneck_edges(node_strengths, model.A, bottleneck_node, k=1)

    assert edge_indices.tolist() == [_flat_index(_STRONG_ATTACKER, _BOTTLENECK)]


# ---------------------------------------------------------------------------
# find_and_escape_bottleneck
# ---------------------------------------------------------------------------


def test_find_and_escape_bottleneck_picks_the_higher_leverage_edge():
    model = _make_model(target_support=0.2)
    grae_vector = compute_grae(model, _SAMPLE, [_TARGET]).casebase_edges.reshape(-1)
    assert grae_vector.abs().max().item() == 0.0  # sanity: confirms the dead-gradient premise

    escape = find_and_escape_bottleneck(model, _SAMPLE, _TARGET, grae_vector, k=1)

    assert escape is not None
    edge_indices, alpha, new_A, new_target_strength, new_rival_class, new_rival_strength = escape
    assert edge_indices.tolist() == [_flat_index(_STRONG_ATTACKER, _BOTTLENECK)]
    assert alpha > 0
    assert new_A.shape == model.A.shape
    # node 0's strength must have risen now that node 1 is no longer pinned at 0
    assert new_target_strength > 0.1
    assert new_rival_class is None  # single-default-argument casebase -> no real rival
    assert new_rival_strength == THRESHOLD


def test_find_and_escape_bottleneck_returns_none_when_structurally_hopeless():
    """Every attacker of the bottleneck node is itself pinned at 0 (base
    scores <= 0), so there is no edge anywhere with any leverage to escape
    it."""
    model = _make_model(target_support=0.2, base_scores=_HOPELESS_BASE_SCORES)
    grae_vector = compute_grae(model, _SAMPLE, [_TARGET]).casebase_edges.reshape(-1)

    escape = find_and_escape_bottleneck(model, _SAMPLE, _TARGET, grae_vector, k=1)

    assert escape is None


# ---------------------------------------------------------------------------
# contest() end-to-end
# ---------------------------------------------------------------------------


def test_contest_starts_dead_escapes_then_finishes_via_ordinary_steps():
    model = _make_model(target_support=0.2)
    new_case = _SAMPLE

    before = model(new_case)[0, _TARGET].item()
    assert before < THRESHOLD + MARGIN  # sanity: starts below the boundary

    result = contest(model, new_case, _TARGET, k=1, max_iters=30)

    assert isinstance(result, ContestResult)
    assert result.success
    assert result.final_target_strength is not None
    assert result.final_target_strength >= THRESHOLD + MARGIN
    # at least the bottleneck-escape step, plus >=1 ordinary gradient step to
    # finish crossing the margin
    assert len(result.edge_trace) >= 2
    # the first accepted step must be the bottleneck escape, on the
    # strong-leverage edge -- not an ordinary top-k step (grae started dead)
    assert result.edge_trace[0].edge_ids == [_flat_index(_STRONG_ATTACKER, _BOTTLENECK)]


def test_contest_reports_failure_cleanly_when_structurally_hopeless():
    model = _make_model(target_support=0.2, base_scores=_HOPELESS_BASE_SCORES)
    new_case = _SAMPLE

    result = contest(model, new_case, _TARGET, k=1, max_iters=10)

    assert not result.success
    assert result.iterations == 1  # bails on the first dead iteration, no wasted looping
    assert result.edge_trace == []
