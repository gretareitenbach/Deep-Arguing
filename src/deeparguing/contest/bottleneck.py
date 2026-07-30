"""
src/deeparguing/contest/bottleneck.py

Escape logic for the dead-gradient case ``contest()`` hits when a hard-ReLU
node upstream of ``target_class`` has saturated (its strength pinned at
exactly 0). Walks backward from the target to find the saturated node, then
grows a step along its highest-leverage incoming edge until it un-sticks.
"""

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR

from .contest import (DEFAULT_K, THRESHOLD, _default_source_mask,
                       _forward_strengths, _perturb_adjacency,
                       _target_and_rival, select_top_k)

# ---- Config -------------------------------------------------------------

BOTTLENECK_ALPHA_INIT = 1e-3     # initial step size for the expanding search
BOTTLENECK_GROWTH_FACTOR = 2.0   # growth factor per trial
MAX_EXPANSIONS = 20              # expansion-phase retry cap


def _node_strengths(model: GradualAACBR, sample: Tensor, A: Tensor) -> Tensor:
    """Forward pass of ``sample`` with ``model.A`` temporarily swapped for ``A``.

    Returns every casebase node's own converged strength, shape (n, d).
    """
    original_A = model.A
    try:
        model.A = A
        with torch.no_grad():
            strengths = model(sample, return_all_strengths=True)
    finally:
        model.A = original_A
    result = strengths[0]
    return result if result.ndim == 2 else result.unsqueeze(-1)


def find_bottleneck(
    model: GradualAACBR, sample: Tensor, target_class: int
) -> tuple[int, Tensor] | None:
    """Walk the influence graph backward from ``target_class``'s default
    argument, following the strongest incoming edge at each hop, looking
    for the first ancestor whose own strength is pinned at exactly 0.

    Parameters
    ----------
    model : GradualAACBR
        A fitted model.
    sample : Tensor
        A single new case, shape (1, x1, ..., xn).
    target_class : int
        Which entry of ``model.default_indexes`` to walk backward from.

    Returns
    -------
    tuple[int, Tensor] | None
        ``(bottleneck_node, node_strengths)`` where ``node_strengths`` is
        the (n, d) per-node strength tensor computed during the walk, or
        ``None`` if no saturated node is found.
    """
    assert model.A is not None
    A = model.post_process_func(model.A)
    node_strengths = _node_strengths(model, sample, model.A)
    target_idx = int(model.default_indexes[target_class].item())

    def is_saturated(node: int) -> bool:
        return bool((node_strengths[node] == 0).all())

    current = target_idx
    visited = {current}
    while True:
        incoming = A[:, current].abs().sum(dim=-1).clone()
        incoming[list(visited)] = -1.0
        nxt = int(incoming.argmax().item())
        if incoming[nxt] <= 0:
            break  # dead end
        visited.add(nxt)
        current = nxt
        if is_saturated(current):
            return current, node_strengths

    if is_saturated(target_idx):
        return target_idx, node_strengths
    return None


def _bottleneck_leverage_vector(
    node_strengths: Tensor,
    A: Tensor,
    bottleneck_node: int,
    default_indexes: Tensor | None = None,
) -> Tensor:
    """Flat (n*n*d,) vector matching ``_casebase_grae``'s layout: zero
    everywhere except ``bottleneck_node``'s incoming edges
    (``A[:, bottleneck_node, :]``), where source node j's entry is
    ``node_strengths[j]``.

    Parameters
    ----------
    node_strengths : Tensor
        Per-node strengths, shape (n, d).
    A : Tensor
        Casebase adjacency, shape (n, n, d).
    bottleneck_node : int
        Index of the saturated node.
    default_indexes : Tensor | None
        If given, zeroes out any entry whose source is one of them.

    Returns
    -------
    Tensor
        Flat (n*n*d,) leverage vector.
    """
    n, m, d = A.shape
    leverage = torch.zeros(n, m, d, dtype=node_strengths.dtype, device=A.device)
    leverage[:, bottleneck_node, :] = node_strengths
    leverage = leverage.reshape(-1)
    if default_indexes is not None:
        leverage = leverage.masked_fill(_default_source_mask(default_indexes, A.shape), 0.0)
    return leverage


def select_bottleneck_edges(
    node_strengths: Tensor,
    A: Tensor,
    bottleneck_node: int,
    k: int,
    default_indexes: Tensor | None = None,
) -> Tensor:
    """Indices (into the flattened ``model.A``, same convention as
    ``select_top_k``) of the k edges feeding into ``bottleneck_node`` with
    the largest ``|node_strengths[source]|``.
    """
    return select_top_k(
        _bottleneck_leverage_vector(node_strengths, A, bottleneck_node, default_indexes), k
    )


def expanding_step_search(
    model: GradualAACBR,
    sample: Tensor,
    edge_indices: Tensor,
    direction: Tensor,
    bottleneck_node: int,
    alpha_init: float = BOTTLENECK_ALPHA_INIT,
    growth_factor: float = BOTTLENECK_GROWTH_FACTOR,
    max_steps: int = MAX_EXPANSIONS,
) -> tuple[float, Tensor] | None:
    """Grow ``alpha`` geometrically from ``alpha_init`` until
    ``bottleneck_node``'s own strength is no longer pinned at exactly 0, or
    ``max_steps`` is hit.

    Returns
    -------
    tuple[float, Tensor] | None
        ``(alpha, new_A)`` for the first un-stuck trial, or ``None`` if it
        never un-sticks within budget.
    """
    assert model.A is not None
    alpha = alpha_init
    for _ in range(max_steps):
        trial_A = _perturb_adjacency(model.A, edge_indices, direction, alpha)
        node_strengths = _node_strengths(model, sample, trial_A)
        if not bool((node_strengths[bottleneck_node] == 0).all()):
            return alpha, trial_A
        alpha *= growth_factor
    return None


def find_and_escape_bottleneck(
    model: GradualAACBR,
    sample: Tensor,
    target_class: int,
    grae_vector: Tensor,
    k: int = DEFAULT_K,
    threshold: float = THRESHOLD,
) -> tuple[Tensor, float, Tensor, float, int | None, float] | None:
    """Find the saturated bottleneck node, rank its incoming edges by local
    leverage, and grow a step along the highest-leverage one until it
    un-sticks.

    Parameters
    ----------
    model : GradualAACBR
        A fitted model.
    sample : Tensor
        A single new case, shape (1, x1, ..., xn).
    target_class : int
        Which entry of ``model.default_indexes`` is being contested.
    grae_vector : Tensor
        The (dead) gradient vector from ``contest()``, used as a fallback
        direction if the local leverage vector is entirely zero.
    k, threshold
        See ``contest.py``'s module-level defaults.

    Returns
    -------
    tuple | None
        ``(edge_indices, alpha, new_A, new_target_strength,
        new_rival_class, new_rival_strength)``, matching
        ``bisection_line_search``'s result shape, or ``None`` if no
        bottleneck exists or it can't be escaped within budget.
    """
    assert model.A is not None
    bottleneck = find_bottleneck(model, sample, target_class)
    if bottleneck is None:
        return None
    bottleneck_node, node_strengths = bottleneck

    default_indexes = model.default_indexes if model.defaults_not_attack else None
    edge_indices = select_bottleneck_edges(
        node_strengths, model.A, bottleneck_node, k, default_indexes
    )
    leverage_vector = _bottleneck_leverage_vector(
        node_strengths, model.A, bottleneck_node, default_indexes
    )
    direction = leverage_vector[edge_indices]
    if not bool(direction.any()):
        direction = grae_vector[edge_indices]
        if not bool(direction.any()):
            return None

    step = expanding_step_search(model, sample, edge_indices, direction, bottleneck_node)
    if step is None:
        return None
    alpha, new_A = step

    new_strengths = _forward_strengths(model, sample, new_A)
    new_target_strength, new_rival_class, new_rival_strength = _target_and_rival(
        new_strengths, target_class, threshold
    )
    return edge_indices, alpha, new_A, new_target_strength, new_rival_class, new_rival_strength
