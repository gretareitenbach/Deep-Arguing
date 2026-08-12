"""
src/deeparguing/casebase_edge_weights/finetune.py

Two-term loss for distilling ``new_case_contest``'s E-only corrections into
``LearnedPartialOrder``'s trainable ``feature_weights_1`` extractor, per
``week7_checklist.md``'s Tuesday plan: ``correction_loss`` (fit the touched
pairs) and ``protect_loss`` (output-level margin hinge on currently-correct
held-out samples -- see ``protect_loss``'s docstring).

An earlier third term, ``preservation_loss`` (an MSE anchor pulling
``partial_order``'s raw output back toward its pre-finetune values, weighted
by ``|model.A|``), was removed 2026-08-12 -- see updates.md. The rationale:
it anchored the network's internal representation for its own sake, but what
actually matters is whether currently-correct *predictions* stay correct,
which is exactly what ``protect_loss`` already checks directly. A same-day
ablation (``lam=0``, sweeping ``protect_lambda``) confirmed ``protect_loss``
alone catches real margin violations that occur when nothing anchors the
raw output -- it isn't just redundant with the removed term.

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

import copy
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import torch
from torch import Tensor
from torch.nn import Parameter
from tqdm import tqdm

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


def protect_loss(
    model: GradualAACBR,
    protect_samples: Tensor,
    protect_target_classes: Sequence[int],
    protect_margin: float = MARGIN,
) -> Tensor:
    """Margin hinge over samples that are currently (pre-finetune) correctly
    classified: penalizes the thing we actually care about directly, not a
    proxy over ``partial_order``'s raw value. For each sample in
    ``protect_samples``, does its target class still beat the best rival by
    ``protect_margin``, evaluated through the model's *current* (training)
    ``feature_weights_1`` but the *frozen* ``model.A`` -- the exact forward
    path ``new_case_contest``/``grae.py`` already use, never
    ``model.forward()`` (which doesn't detach ``model.A``). Same hinge
    formula as ``batch_contest.py``'s existing ``protect_lambda`` mechanism
    (``clamp(protect_margin - margin, min=0)``), just evaluated in
    weight-space instead of A-space.

    The only regularizer against ``correction_loss`` since ``preservation_loss``
    (an MSE anchor on ``partial_order``'s raw output) was removed -- see this
    module's docstring. Unlike that removed term, this one is a hinge: it's
    exactly 0, with no gradient, whenever every protected sample's margin is
    already comfortably above ``protect_margin``, so it only pushes back once
    something concrete is actually at risk.

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
    protect: Tensor
    combined: Tensor


def compute_losses(
    model: GradualAACBR,
    new_cases: Tensor,
    casebase_items: Tensor,
    targets: Tensor,
    protect_samples: Tensor,
    protect_target_classes: Sequence[int],
    protect_margin: float,
    protect_lambda: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> IrrelevanceFinetuneLosses:
    """One call per training step: correction loss over the given batch of
    touched pairs, protect loss over currently-correct held-out samples, and
    their ``protect_lambda``-weighted combination.
    """
    correction = correction_loss(model, new_cases, casebase_items, targets, chunk_size)
    protect = protect_loss(model, protect_samples, protect_target_classes, protect_margin)
    combined = correction + protect_lambda * protect
    return IrrelevanceFinetuneLosses(correction, protect, combined)


def _sample_batch(tensors: dict[str, Tensor], batch_size: int | None) -> dict[str, Tensor]:
    n = tensors["targets"].shape[0]
    if batch_size is None or batch_size >= n:
        return tensors
    idx = torch.randperm(n)[:batch_size]
    return {k: v[idx] for k, v in tensors.items()}


@dataclass
class FinetuneRunResult:
    """Everything a caller (the CLI script, a hyperparameter sweep, ...)
    needs after ``run_finetune`` returns. ``best_*`` tracks the lowest
    val-``combined``-loss point seen across every eval (not just the final
    step) -- see ``run_finetune``'s docstring for why. ``history`` has one
    entry per eval (every ``log_every`` steps, plus the final step).
    """
    best_step: int | None
    best_val_losses: IrrelevanceFinetuneLosses | None
    best_extractor_state: dict[str, Tensor] | None
    final_extractor_state: dict[str, Tensor]
    final_train_losses: IrrelevanceFinetuneLosses
    final_val_losses: IrrelevanceFinetuneLosses | None
    history: list[dict[str, float]] = field(default_factory=list)


def run_finetune(
    model: GradualAACBR,
    train_tensors: dict[str, Tensor],
    val_tensors: dict[str, Tensor],
    lr: float,
    steps: int,
    batch_size: int | None,
    chunk_size: int,
    protect_samples: Tensor,
    protect_target_classes: Sequence[int],
    protect_margin: float,
    protect_lambda: float,
    log_every: int = 10,
    on_eval: Callable[[int, IrrelevanceFinetuneLosses, IrrelevanceFinetuneLosses | None], None] | None = None,
    show_progress: bool = True,
    progress_desc: str = "Fine-tuning irrelevance channel",
) -> FinetuneRunResult:
    """Train ``feature_weights_1`` for ``steps`` Adam updates against
    ``train_tensors``, tracking both the final-step weights and the
    best-val-``combined``-loss weights seen along the way (2026-08-11's
    200-step run plateaued/overfit on val well before its final step -- see
    ``finetune_irrelevance.py``'s module docstring -- so callers should
    normally prefer ``best_extractor_state`` over ``final_extractor_state``).

    Assumes ``freeze_all_except_trainable(model)`` has already been called
    and the trainable extractor already holds whatever weights this run
    should start from -- this function trains in place and never resets
    them itself, so a caller running several fine-tunes back to back (e.g.
    one per hyperparameter combo in a sweep) must reset the extractor's
    weights (``load_state_dict`` on a saved pre-finetune snapshot) between
    calls.

    Parameters
    ----------
    model, protect_samples, protect_target_classes :
        See ``compute_losses``.
    train_tensors, val_tensors : dict[str, Tensor]
        ``{"new_cases", "casebase_items", "targets"}``, as produced by
        ``build_irrelevance_finetune_dataset.py``. ``val_tensors`` may have
        0 pairs (no val split), in which case ``best_*`` falls back to the
        final step and ``on_eval``'s second argument is always ``None``.
    lr, steps, batch_size, chunk_size, protect_margin, protect_lambda :
        See ``finetune_irrelevance.py``'s CLI flags of the same name.
    log_every : int
        Eval (and ``on_eval`` callback) frequency, in steps.
    on_eval : callable, optional
        Called with ``(step, train_losses, val_losses)`` at every eval point,
        for callers that want their own logging (markdown log, sweep
        summary, ...) without this function taking an opinion on format.
    show_progress : bool
        Whether to wrap the step loop in a ``tqdm`` bar (off for sweeps,
        which drive their own outer progress bar over combos).
    progress_desc : str
        ``tqdm`` bar description, when ``show_progress`` is true.

    Returns
    -------
    FinetuneRunResult
    """
    trainable_extractor = model.casebase_edge_weights.feature_extractors[TRAINABLE_FEATURE_EXTRACTOR_INDEX]
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=lr)

    def _eval(tensors: dict[str, Tensor]) -> IrrelevanceFinetuneLosses | None:
        if tensors["targets"].shape[0] == 0:
            return None
        with torch.no_grad():
            return compute_losses(
                model, tensors["new_cases"], tensors["casebase_items"], tensors["targets"],
                protect_samples, protect_target_classes, protect_margin, protect_lambda,
                chunk_size,
            )

    best_step: int | None = None
    best_val_losses: IrrelevanceFinetuneLosses | None = None
    best_extractor_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []

    step_iterable: Iterator[int] = range(1, steps + 1)
    progress = tqdm(step_iterable, desc=progress_desc, unit="step") if show_progress else step_iterable
    train_losses: IrrelevanceFinetuneLosses | None = None
    val_losses: IrrelevanceFinetuneLosses | None = None
    for step in progress:
        batch = _sample_batch(train_tensors, batch_size)
        train_losses = compute_losses(
            model, batch["new_cases"], batch["casebase_items"], batch["targets"],
            protect_samples, protect_target_classes, protect_margin, protect_lambda,
            chunk_size,
        )
        optimizer.zero_grad()
        train_losses.combined.backward()
        optimizer.step()

        if show_progress:
            progress.set_postfix(  # type: ignore[union-attr]
                correction=f"{train_losses.correction.item():.4f}",
                protect=f"{train_losses.protect.item():.4f}",
                combined=f"{train_losses.combined.item():.4f}",
            )

        if step % log_every == 0 or step == steps:
            val_losses = _eval(val_tensors)
            history.append(
                {
                    "step": step,
                    "train_correction": train_losses.correction.item(),
                    "train_protect": train_losses.protect.item(),
                    "train_combined": train_losses.combined.item(),
                    **(
                        {
                            "val_correction": val_losses.correction.item(),
                            "val_protect": val_losses.protect.item(),
                            "val_combined": val_losses.combined.item(),
                        }
                        if val_losses is not None
                        else {}
                    ),
                }
            )
            if val_losses is not None and (
                best_val_losses is None or val_losses.combined.item() < best_val_losses.combined.item()
            ):
                best_step = step
                best_val_losses = val_losses
                best_extractor_state = copy.deepcopy(trainable_extractor.state_dict())
            if on_eval is not None:
                on_eval(step, train_losses, val_losses)

    assert train_losses is not None  # steps >= 1 is the only supported case
    return FinetuneRunResult(
        best_step=best_step,
        best_val_losses=best_val_losses,
        best_extractor_state=best_extractor_state,
        final_extractor_state=copy.deepcopy(trainable_extractor.state_dict()),
        final_train_losses=train_losses,
        final_val_losses=val_losses,
        history=history,
    )
