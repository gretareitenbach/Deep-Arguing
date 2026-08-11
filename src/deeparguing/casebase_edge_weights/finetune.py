"""
src/deeparguing/casebase_edge_weights/finetune.py

Three-term loss for distilling ``new_case_contest``'s E-only corrections into
``LearnedPartialOrder``'s trainable ``feature_weights_1`` extractor, per
``week7_checklist.md``'s Tuesday plan: ``correction_loss`` (fit the touched
pairs), ``preservation_loss`` (MSE anchor on ``partial_order``'s raw output,
weighted by ``|model.A|``), and ``protect_loss`` (output-level margin hinge
on currently-correct held-out samples -- a backstop for pairs
``preservation_loss`` doesn't cover; see ``protect_loss``'s docstring).

Target-space note (decided/verified 2026-08-10, see updates.md): the
``corrected_E`` values logged by ``contest_all_irrelevance.py`` are
``new_case_contest``'s internal ``E`` -- i.e.
``model.new_cases_attacks_adjacency`` -- which is ``-irrelevance_edge_weights(...)
= -(1 - partial_order(...)) = partial_order(...) - 1``. The regression target
for ``partial_order``'s raw output is therefore ``corrected_E + 1``, NOT
``1 - corrected_E`` (that formula is for the *unnegated* irrelevance value,
which is not what's stored). See ``corrected_E_to_partial_order_target``.

This only applies when ``irrelevance_edge_weights`` is a ``RegularIrrelevance``
sharing its ``compute_partial_order`` with ``casebase_edge_weights`` (true for
``tuning/cifar10/resnet/relu/model_cifar10_image.yaml``, not true in general --
``FeatureWeightedIrrelevance`` has its own independent weights and no
``partial_order`` to regress toward at all). ``assert_shares_partial_order``
guards this precondition.
"""

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from torch import Tensor
from torch.nn import Parameter

from deeparguing.casebase_edge_weights.learned_partial_order import \
    LearnedPartialOrder
from deeparguing.contest.core.contest import (MARGIN, THRESHOLD,
                                               _target_and_rival_batch)
from deeparguing.contest.core.grae import (_batched_casebase_base_scores,
                                            _replay_default_strengths)
from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.irrelevance_edge_weights.regular_irrelevance import \
    RegularIrrelevance

# Index into LearnedPartialOrder.feature_extractors that Week 7's plan
# fine-tunes; feature_extractors[0] ("feature_weights") is the frozen
# ResNet shared with base_score, feature_extractors[1] ("feature_weights_1")
# is the trainable MLP -- see tuning/cifar10/resnet/relu/model_cifar10_image.yaml.
TRAINABLE_FEATURE_EXTRACTOR_INDEX = 1

DEFAULT_CHUNK_SIZE = 128  # bounds the quadratic cost of the diagonal trick below
EPS = 1e-12


def corrected_E_to_partial_order_target(corrected_E: Tensor) -> Tensor:
    """Convert a logged ``corrected_E`` value (``new_case_contest``'s
    internal, already-negated E) to the regression target for
    ``partial_order``'s raw output. See module docstring -- this is
    ``corrected_E + 1``, not ``1 - corrected_E``.
    """
    return corrected_E + 1.0


def assert_shares_partial_order(model: GradualAACBR) -> None:
    """Raise if this model's irrelevance channel isn't a ``RegularIrrelevance``
    sharing the same ``compute_partial_order`` instance as
    ``model.casebase_edge_weights`` -- the precondition
    ``corrected_E_to_partial_order_target`` and this whole fine-tuning
    approach depend on. Fails loudly rather than silently regressing a
    disconnected (e.g. ``FeatureWeightedIrrelevance``) irrelevance channel
    toward a target it has no relationship to.
    """
    irrelevance = model.irrelevance_edge_weights
    if not isinstance(irrelevance, RegularIrrelevance):
        raise TypeError(
            f"model.irrelevance_edge_weights is a {type(irrelevance).__name__}, "
            "not RegularIrrelevance -- the corrected_E <-> partial_order target "
            "conversion this module relies on doesn't hold for it."
        )
    if irrelevance.compute_partial_order is not model.casebase_edge_weights:
        raise ValueError(
            "model.irrelevance_edge_weights.compute_partial_order is not the "
            "same instance as model.casebase_edge_weights -- fine-tuning it "
            "wouldn't affect this model's irrelevance channel."
        )


def _partial_order_module(model: GradualAACBR) -> LearnedPartialOrder:
    assert_shares_partial_order(model)
    partial_order = model.casebase_edge_weights
    if not isinstance(partial_order, LearnedPartialOrder):
        raise TypeError(
            f"model.casebase_edge_weights is a {type(partial_order).__name__}, "
            "not LearnedPartialOrder -- no feature_extractors to fine-tune."
        )
    return partial_order


def trainable_parameters(model: GradualAACBR) -> Iterator[Parameter]:
    """The parameters Week 7's plan fine-tunes: only
    ``casebase_edge_weights.feature_extractors[TRAINABLE_FEATURE_EXTRACTOR_INDEX]``
    (``feature_weights_1``), never the frozen ResNet or the comparison
    function's own parameters (if any).
    """
    partial_order = _partial_order_module(model)
    extractor = partial_order.feature_extractors[TRAINABLE_FEATURE_EXTRACTOR_INDEX]
    return extractor.parameters()


def freeze_all_except_trainable(model: GradualAACBR) -> None:
    """Set ``requires_grad=False`` on every parameter, then ``True`` on just
    ``trainable_parameters(model)``. Also puts the whole model in ``eval()``
    mode: even with ``requires_grad=False``, BatchNorm layers inside the
    frozen ResNet would otherwise keep updating their running stats in
    ``train()`` mode. ``eval()`` mode doesn't block gradients into
    ``feature_weights_1`` -- it only disables dropout/BN-stat-updates, which
    is the right call for fine-tuning against a small touched-pairs dataset
    anyway (avoids noisy BN drift from a handful of steps).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for p in trainable_parameters(model):
        p.requires_grad_(True)
    model.eval()


def correction_loss(
    model: GradualAACBR,
    new_cases: Tensor,
    casebase_items: Tensor,
    targets: Tensor,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Tensor:
    """MSE between ``partial_order(new_cases[i], casebase_items[i])`` and
    ``targets[i]``, for aligned pairs (NOT ``LearnedPartialOrder.forward``'s
    usual cross-product).

    ``LearnedPartialOrder.forward(attacker, target)`` broadcasts to every
    (attacker, target) combination, shape (len(attacker), len(target), d).
    For a batch of specific, already-paired (new_case, casebase_item) pairs
    we only want the diagonal of that -- computing the full cross-product
    for the whole dataset at once would be quadratic in memory (thousands of
    touched pairs squared). Chunking into ``chunk_size``-sized pieces and
    taking the diagonal of each keeps the wasted off-diagonal compute
    bounded while still reusing ``LearnedPartialOrder.forward`` as-is (no
    new broadcasting logic).

    Parameters
    ----------
    new_cases, casebase_items : Tensor
        Aligned, shape (T, ...) each -- new_cases[i] pairs with casebase_items[i].
    targets : Tensor
        Shape (T,), see ``corrected_E_to_partial_order_target``.
    chunk_size : int

    Returns
    -------
    Tensor
        Scalar mean squared error over all T pairs.
    """
    partial_order = _partial_order_module(model)
    T = new_cases.shape[0]
    if T == 0:
        return torch.zeros((), device=targets.device, dtype=targets.dtype)

    squared_errors = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk_new = new_cases[start:end]
        chunk_case = casebase_items[start:end]
        chunk_targets = targets[start:end]

        cross = partial_order(chunk_new, chunk_case)  # (b, b, d)
        if cross.shape[-1] != 1:
            raise NotImplementedError(
                f"partial_order's raw output has d={cross.shape[-1]} (this "
                "checkpoint's is d=1) -- the (b,) diagonal-vs-targets "
                "comparison below assumes a scalar comparison per pair; "
                "extend this before using a multi-dimensional partial_order."
            )
        b = end - start
        diag = cross[torch.arange(b), torch.arange(b)].squeeze(-1)  # (b,)
        squared_errors.append((diag - chunk_targets) ** 2)

    return torch.cat(squared_errors).mean()


def preservation_loss(
    model: GradualAACBR,
    X_train: Tensor,
    frozen_raw_partial_order: Tensor,
) -> Tensor:
    """Weighted MSE between the current (live-weights) raw ``partial_order``
    output over every casebase-internal pair and a frozen pre-finetune
    snapshot of the same, weighted by ``|model.A|`` -- near-zero for pairs
    ``fit()`` already masked out (same label, blocked, non-minimal), since
    there's nothing there worth protecting.

    Normalized by total weight (a weighted *mean*, not a weighted sum), so
    ``lambda`` in ``combine_losses`` means the same thing regardless of
    casebase size -- important since Wednesday's sweep varies it.

    Parameters
    ----------
    model : GradualAACBR
    X_train : Tensor
        Shape (n, ...), same casebase ``frozen_raw_partial_order`` was
        computed from (``model.X_train``).
    frozen_raw_partial_order : Tensor
        ``model.casebase_edge_weights(X_train, X_train)``, captured once
        before fine-tuning starts (detached).

    Returns
    -------
    Tensor
        Scalar weighted MSE.
    """
    assert model.A is not None
    partial_order = _partial_order_module(model)
    current = partial_order(X_train, X_train)  # (n, n, d), same shape as model.A
    weight = model.A.detach().abs()
    squared_error = (current - frozen_raw_partial_order) ** 2
    return (weight * squared_error).sum() / weight.sum().clamp_min(EPS)


def protect_loss(
    model: GradualAACBR,
    protect_samples: Tensor,
    protect_target_classes: Sequence[int],
    protect_margin: float = MARGIN,
) -> Tensor:
    """Output-level backstop for ``preservation_loss``: a margin hinge over
    samples that are currently (pre-finetune) correctly classified, rather
    than a proxy over ``partial_order``'s raw value.

    ``preservation_loss`` only ever protects pairs already live in
    ``model.A`` (weighted by ``|model.A|``); it puts zero cost on
    ``partial_order`` drifting at pairs ``fit()`` currently masks out
    (``__minimal_attacks``'s continuous blocking product), which could in
    principle become live if ``model.A`` were ever recomputed from this
    fine-tuned network. This term instead penalizes the thing we actually
    care about directly: for each sample in ``protect_samples``, does its
    target class still beat the best rival by ``protect_margin``, evaluated
    through the model's *current* (training) ``feature_weights_1`` but the
    *frozen* ``model.A`` -- the exact forward path ``new_case_contest``/
    ``grae.py`` already use, never ``model.forward()`` (which doesn't detach
    ``model.A``). Same hinge formula as ``batch_contest.py``'s existing
    ``protect_lambda`` mechanism (``clamp(protect_margin - margin, min=0)``),
    just evaluated in weight-space instead of A-space.

    Parameters
    ----------
    model : GradualAACBR
    protect_samples : Tensor
        Shape (M, ...) -- samples known to be correctly classified before
        fine-tuning started (e.g. from ``global_optimize.py``'s
        ``_build_protect_set``). Empty batches are a no-op.
    protect_target_classes : Sequence[int]
        Length-M true class of each protect sample.
    protect_margin : float

    Returns
    -------
    Tensor
        Scalar mean hinge loss.
    """
    assert model.A is not None
    if protect_samples.shape[0] == 0:
        return torch.zeros((), device=protect_samples.device)

    A = model.A.detach()
    with torch.no_grad():
        casebase_base_scores = _batched_casebase_base_scores(model, protect_samples.shape[0])
        new_cases_base_scores = model.compute_base_scores(protect_samples).unsqueeze(-1)
    irrelevance = model.irrelevance_edge_weights(protect_samples, model.X_train)
    E = -irrelevance

    strengths = _replay_default_strengths(model, A, E, casebase_base_scores, new_cases_base_scores)
    target, _, rival = _target_and_rival_batch(strengths, protect_target_classes, THRESHOLD)
    return torch.clamp(protect_margin - (target - rival), min=0.0).mean()


@dataclass
class IrrelevanceFinetuneLosses:
    correction: Tensor
    preservation: Tensor
    protect: Tensor
    combined: Tensor


def compute_losses(
    model: GradualAACBR,
    new_cases: Tensor,
    casebase_items: Tensor,
    targets: Tensor,
    X_train: Tensor,
    frozen_raw_partial_order: Tensor,
    lam: float,
    protect_samples: Tensor,
    protect_target_classes: Sequence[int],
    protect_margin: float,
    protect_lambda: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> IrrelevanceFinetuneLosses:
    """One call per training step: correction loss over the given batch of
    touched pairs, preservation loss over the full (frozen-vs-current)
    casebase, protect loss over currently-correct held-out samples, and
    their ``lambda``-weighted combination.
    """
    correction = correction_loss(model, new_cases, casebase_items, targets, chunk_size)
    preservation = preservation_loss(model, X_train, frozen_raw_partial_order)
    protect = protect_loss(model, protect_samples, protect_target_classes, protect_margin)
    combined = correction + lam * preservation + protect_lambda * protect
    return IrrelevanceFinetuneLosses(correction, preservation, protect, combined)
