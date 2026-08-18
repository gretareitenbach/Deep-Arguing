"""Gradient-based Relation Attribution Explanations (G-RAEs).

Computes the gradient of an argument's final strength with respect to
individual edge weights of model.A (casebase-internal adjacency) and
model.new_cases_attacks_adjacency (new case's own edges), using PyTorch
autograd.
"""

import itertools
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.semantics.gradual_semantics import GradualSemantics


@dataclass
class GRAEResult:
    """Container for the two edge-weight gradients that make up a G-RAE.

    Attributes
    ----------
    casebase_edges : Tensor
        Gradient of the (summed, batched) target strength with respect to
        model.A. Shape matches model.A (n, n, d), or (B, n, n, d)
        if per_sample was requested.
    new_case_edges : Tensor
        Gradient of each sample's own target strength with respect to its
        row of model.new_cases_attacks_adjacency. Shape (B, n, d).
    target_indices : Sequence[int]
        The default-argument index used as the topic argument for each
        sample in the batch.
    """

    casebase_edges: Tensor
    new_case_edges: Tensor
    target_indices: Sequence[int]


def _batched_casebase_base_scores(model: GradualAACBR, batch_size: int) -> Tensor:
    """Recompute and batch-tile the casebase's own base scores.

    Returns
    -------
    Tensor
        Shape (B, n, d).
    """
    with torch.no_grad():
        scores = model.compute_base_scores(model.X_train)
    return torch.tile(scores.unsqueeze(0), (batch_size, 1, 1))


def _replay_default_strengths(
    model: GradualAACBR,
    A: Tensor,
    E: Tensor,
    casebase_base_scores: Tensor,
    new_cases_base_scores: Tensor,
    semantics: GradualSemantics | None = None,
) -> Tensor:
    """Replay __new_case_influence + gradual_semantics with A/E
    swapped in for model.A/model.new_cases_attacks_adjacency.

    Parameters
    ----------
    model : GradualAACBR
    A : Tensor
        Casebase adjacency to use instead of model.A.
    E : Tensor
        New-case adjacency to use instead of model.new_cases_attacks_adjacency.
    casebase_base_scores : Tensor
    new_cases_base_scores : Tensor
    semantics : GradualSemantics | None
        If given, replaces model.gradual_semantics for this replay only.

    Returns
    -------
    Tensor
        Every default class's strength, shape (B, D).
    """
    semantics = semantics or model.gradual_semantics
    aggregations = semantics.aggregation_func(
        E.unsqueeze(1), new_cases_base_scores
    )
    influenced_base_scores = semantics.influence_func(
        casebase_base_scores, aggregations
    )
    strengths = semantics(model.post_process_func(A), influenced_base_scores)

    if model.dimensions > 1:
        final_strengths = torch.matmul(strengths, model.W)
    else:
        final_strengths = strengths.squeeze(-1)

    return final_strengths[:, model.default_indexes]


def compute_grae(
    model: GradualAACBR,
    new_cases: Tensor,
    target_indices: Sequence[int],
    per_sample: bool = False,
    rival_indices: Sequence[int | None] | None = None,
    semantics_override: GradualSemantics | None = None,
) -> GRAEResult:
    """Compute G-RAEs for a batch of new cases against a chosen target argument each.

    Parameters
    ----------
    model : GradualAACBR
        A fitted model (model.A populated).
    new_cases : Tensor
        Batch of new case characterisations, shape (B, x1, ..., xn).
    target_indices : Sequence[int]
        Length-B sequence giving, for each sample, which entry of
        model.default_indexes to differentiate the strength of.
    per_sample : bool, default False
        If True, also recover a per-sample casebase_edges gradient at
        the cost of B extra backward passes. If False, casebase_edges
        is the aggregate gradient across the whole batch.
    rival_indices : Sequence[int | None] | None, default None
        If given, length-B, one entry per sample: differentiate
        target_strength - rival_strength instead of just
        target_strength for that sample (None for a sample means
        differentiate target only).
    semantics_override : GradualSemantics | None, default None
        If given, replaces model.gradual_semantics for this replay only.

    Returns
    -------
    GRAEResult
        All returned tensors are detached.
    """
    if model.A is None:
        raise Exception("Ensure the model has been fit first.")

    batch_size = new_cases.shape[0]
    target_indices = list(target_indices)
    if len(target_indices) != batch_size:
        raise ValueError(
            "target_indices must have exactly one entry per new case: "
            f"got {len(target_indices)} for a batch of {batch_size}."
        )

    with torch.no_grad():
        model(new_cases)

    A_leaf = model.A.detach().clone().requires_grad_(True)
    E_leaf = model.new_cases_attacks_adjacency.detach().clone().requires_grad_(True)

    casebase_base_scores = _batched_casebase_base_scores(model, batch_size)
    new_cases_base_scores = model.new_cases_base_scores.detach()

    default_strengths = _replay_default_strengths(
        model, A_leaf, E_leaf, casebase_base_scores, new_cases_base_scores,
        semantics=semantics_override,
    )

    target_indices_t = torch.as_tensor(target_indices, dtype=torch.long)
    target_strengths = default_strengths[torch.arange(batch_size), target_indices_t]

    if rival_indices is None:
        objective = target_strengths
    else:
        if len(rival_indices) != batch_size:
            raise ValueError(
                "rival_indices must have exactly one entry per new case: "
                f"got {len(rival_indices)} for a batch of {batch_size}."
            )
        terms = [
            target_strengths[i] if rival is None
            else target_strengths[i] - default_strengths[i, rival]
            for i, rival in enumerate(rival_indices)
        ]
        objective = torch.stack(terms)

    objective.sum().backward(retain_graph=per_sample)
    assert A_leaf.grad is not None and E_leaf.grad is not None

    casebase_edges = A_leaf.grad.detach().clone()
    new_case_edges = E_leaf.grad.detach().clone()

    if per_sample:
        casebase_edges = torch.zeros(
            (batch_size, *model.A.shape), dtype=A_leaf.dtype, device=A_leaf.device
        )
        for i in range(batch_size):
            A_leaf.grad = None
            objective[i].backward(retain_graph=True)
            assert A_leaf.grad is not None
            casebase_edges[i] = A_leaf.grad.detach().clone()

    return GRAEResult(
        casebase_edges=casebase_edges,
        new_case_edges=new_case_edges,
        target_indices=target_indices,
    )


def finite_difference_grae(
    model: GradualAACBR,
    new_case: Tensor,
    target_index: int,
    epsilon: float = 1e-4,
) -> GRAEResult:
    """Approximate G-RAEs via perturbation, as a cross-check for
    compute_grae's analytic gradients.

    Parameters
    ----------
    model : GradualAACBR
        A fitted model.
    new_case : Tensor
        A single new case characterisation, shape (1, x1, ..., xn).
    target_index : int
        Which entry of model.default_indexes to treat as the topic argument.
    epsilon : float, default 1e-4
        Perturbation size.

    Returns
    -------
    GRAEResult
        Same shape/structure as compute_grae's output.
    """
    if model.A is None:
        raise Exception("Ensure the model has been fit first.")
    if new_case.shape[0] != 1:
        raise ValueError(
            "finite_difference_grae expects a single new case (batch size 1), "
            f"got batch of {new_case.shape[0]}."
        )

    with torch.no_grad():
        model(new_case)

    A0 = model.A.detach().clone()
    E0 = model.new_cases_attacks_adjacency.detach().clone()

    casebase_base_scores = _batched_casebase_base_scores(model, batch_size=1)
    new_cases_base_scores = model.new_cases_base_scores.detach()

    def target_strength(A: Tensor, E: Tensor) -> float:
        with torch.no_grad():
            default_strengths = _replay_default_strengths(
                model, A, E, casebase_base_scores, new_cases_base_scores
            )
        return default_strengths[0, target_index].item()

    casebase_edges = torch.zeros_like(A0)
    for idx in itertools.product(*(range(size) for size in A0.shape)):
        A_plus, A_minus = A0.clone(), A0.clone()
        A_plus[idx] += epsilon
        A_minus[idx] -= epsilon
        casebase_edges[idx] = (
            target_strength(A_plus, E0) - target_strength(A_minus, E0)
        ) / (2 * epsilon)

    new_case_edges = torch.zeros_like(E0)
    for idx in itertools.product(*(range(size) for size in E0.shape)):
        E_plus, E_minus = E0.clone(), E0.clone()
        E_plus[idx] += epsilon
        E_minus[idx] -= epsilon
        new_case_edges[idx] = (
            target_strength(A0, E_plus) - target_strength(A0, E_minus)
        ) / (2 * epsilon)

    return GRAEResult(
        casebase_edges=casebase_edges,
        new_case_edges=new_case_edges,
        target_indices=[target_index],
    )
