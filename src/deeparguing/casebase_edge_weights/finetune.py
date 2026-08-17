"""Loss terms for distilling new_case_contest's and batch_contest's
corrections into LearnedPartialOrder's trainable feature_weights_1
extractor: correction_loss, casebase_correction_loss, and protect_loss.
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

TRAINABLE_FEATURE_EXTRACTOR_INDEX = 1

DEFAULT_CHUNK_SIZE = 128


def corrected_E_to_partial_order_target(corrected_E: Tensor) -> Tensor:
    """Convert a corrected_E value to the regression target for
    partial_order's raw output
    """
    return corrected_E + 1.0


def assert_shares_partial_order(model: GradualAACBR) -> None:
    """Raise if this model's irrelevance channel isn't a RegularIrrelevance
    sharing the same compute_partial_order instance as
    model.casebase_edge_weights.
    """
    irrelevance = model.irrelevance_edge_weights
    if not isinstance(irrelevance, RegularIrrelevance):
        raise TypeError(
            f"model.irrelevance_edge_weights is a {type(irrelevance).__name__}, "
            "not RegularIrrelevance "
        )
    if irrelevance.compute_partial_order is not model.casebase_edge_weights:
        raise ValueError(
            "model.irrelevance_edge_weights.compute_partial_order is not the "
            "same instance as model.casebase_edge_weights "
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
    partial_order = _partial_order_module(model)
    extractor = partial_order.feature_extractors[TRAINABLE_FEATURE_EXTRACTOR_INDEX]
    return extractor.parameters()


def freeze_all_except_trainable(model: GradualAACBR) -> None:
    """Freeze every parameter except trainable_parameters(model), and put the
    whole model in eval() mode.
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
    """MSE between partial_order(new_cases[i], casebase_items[i]) and
    targets[i].
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


@dataclass(frozen=True)
class CasebaseCorrectionBatch:
    X_casebase: Tensor
    y_casebase: Tensor
    X_default: Tensor
    y_default: Tensor
    source_idx: Tensor
    target_idx: Tensor
    dim_idx: Tensor
    targets: Tensor


def casebase_correction_loss(model: GradualAACBR, batch: CasebaseCorrectionBatch) -> Tensor:
    """MSE between model.A's touched entries and their contest_all.py-
    corrected values, after differentiably re-fitting model.A from the
    current feature_weights_1.
    """
    if batch.targets.shape[0] == 0:
        return torch.zeros((), device=batch.targets.device, dtype=batch.targets.dtype)
    model.fit(batch.X_casebase, batch.y_casebase, batch.X_default, batch.y_default)
    predicted = model.A[batch.source_idx, batch.target_idx, batch.dim_idx]
    return ((predicted - batch.targets) ** 2).mean()


def protect_loss(
    model: GradualAACBR,
    protect_samples: Tensor,
    protect_target_classes: Sequence[int],
    protect_margin: float = MARGIN,
) -> Tensor:
    """Margin hinge over samples that are currently (pre-finetune) correctly
    classified.
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
class FinetuneLosses:
    correction: Tensor
    protect: Tensor
    casebase_correction: Tensor
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
    casebase_batch: CasebaseCorrectionBatch | None = None,
    casebase_lambda: float = 0.0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> FinetuneLosses:
    """One call per training step: correction loss, casebase-internal
    correction loss (if casebase_batch is given), protect loss, and their
    weighted combination.
    """
    correction = correction_loss(model, new_cases, casebase_items, targets, chunk_size)
    casebase_correction = (
        casebase_correction_loss(model, casebase_batch)
        if casebase_batch is not None
        else torch.zeros((), device=targets.device, dtype=targets.dtype)
    )
    protect = protect_loss(model, protect_samples, protect_target_classes, protect_margin)
    combined = correction + protect_lambda * protect + casebase_lambda * casebase_correction
    return FinetuneLosses(correction, protect, casebase_correction, combined)


def _sample_batch(tensors: dict[str, Tensor], batch_size: int | None) -> dict[str, Tensor]:
    n = tensors["targets"].shape[0]
    if batch_size is None or batch_size >= n:
        return tensors
    idx = torch.randperm(n)[:batch_size]
    return {k: v[idx] for k, v in tensors.items()}


@dataclass(frozen=True)
class CasebaseFinetuneConfig:
    """Optional casebase-internal-edge correction alongside the
    new-case correction.
    """
    X_casebase: Tensor
    y_casebase: Tensor
    X_default: Tensor
    y_default: Tensor
    train_edges: dict[str, Tensor]
    val_edges: dict[str, Tensor]
    casebase_lambda: float = 0.0

    def _batch(self, edges: dict[str, Tensor]) -> CasebaseCorrectionBatch:
        return CasebaseCorrectionBatch(
            X_casebase=self.X_casebase, y_casebase=self.y_casebase,
            X_default=self.X_default, y_default=self.y_default,
            source_idx=edges["source_idx"], target_idx=edges["target_idx"],
            dim_idx=edges["dim_idx"], targets=edges["targets"],
        )

    def train_batch(self) -> CasebaseCorrectionBatch:
        return self._batch(self.train_edges)

    def val_batch(self) -> CasebaseCorrectionBatch:
        return self._batch(self.val_edges)


@dataclass
class FinetuneRunResult:
    best_step: int | None
    best_val_losses: FinetuneLosses | None
    best_extractor_state: dict[str, Tensor] | None
    final_extractor_state: dict[str, Tensor]
    final_train_losses: FinetuneLosses
    final_val_losses: FinetuneLosses | None
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
    casebase_config: CasebaseFinetuneConfig | None = None,
    log_every: int = 10,
    on_eval: Callable[[int, FinetuneLosses, FinetuneLosses | None], None] | None = None,
    show_progress: bool = True,
    progress_desc: str = "Fine-tuning edge weights",
) -> FinetuneRunResult:
    """Train feature_weights_1 for steps Adam updates against train_tensors,
    tracking both the final-step weights and the best-val-combined-loss
    weights seen along the way.
    """
    trainable_extractor = model.casebase_edge_weights.feature_extractors[TRAINABLE_FEATURE_EXTRACTOR_INDEX]
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=lr)

    casebase_lambda = casebase_config.casebase_lambda if casebase_config is not None else 0.0

    def _eval(tensors: dict[str, Tensor]) -> FinetuneLosses | None:
        if tensors["targets"].shape[0] == 0:
            return None
        with torch.no_grad():
            return compute_losses(
                model, tensors["new_cases"], tensors["casebase_items"], tensors["targets"],
                protect_samples, protect_target_classes, protect_margin, protect_lambda,
                casebase_config.val_batch() if casebase_config is not None else None, casebase_lambda,
                chunk_size,
            )

    best_step: int | None = None
    best_val_losses: FinetuneLosses | None = None
    best_extractor_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []

    step_iterable: Iterator[int] = range(1, steps + 1)
    progress = tqdm(step_iterable, desc=progress_desc, unit="step") if show_progress else step_iterable
    train_losses: FinetuneLosses | None = None
    val_losses: FinetuneLosses | None = None
    for step in progress:
        batch = _sample_batch(train_tensors, batch_size)
        train_losses = compute_losses(
            model, batch["new_cases"], batch["casebase_items"], batch["targets"],
            protect_samples, protect_target_classes, protect_margin, protect_lambda,
            casebase_config.train_batch() if casebase_config is not None else None, casebase_lambda,
            chunk_size,
        )
        optimizer.zero_grad()
        train_losses.combined.backward()
        optimizer.step()

        if show_progress:
            progress.set_postfix(  # type: ignore[union-attr]
                correction=f"{train_losses.correction.item():.4f}",
                protect=f"{train_losses.protect.item():.4f}",
                casebase=f"{train_losses.casebase_correction.item():.4f}",
                combined=f"{train_losses.combined.item():.4f}",
            )

        if step % log_every == 0 or step == steps:
            val_losses = _eval(val_tensors)
            history.append(
                {
                    "step": step,
                    "train_correction": train_losses.correction.item(),
                    "train_protect": train_losses.protect.item(),
                    "train_casebase_correction": train_losses.casebase_correction.item(),
                    "train_combined": train_losses.combined.item(),
                    **(
                        {
                            "val_correction": val_losses.correction.item(),
                            "val_protect": val_losses.protect.item(),
                            "val_casebase_correction": val_losses.casebase_correction.item(),
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

    assert train_losses is not None
    return FinetuneRunResult(
        best_step=best_step,
        best_val_losses=best_val_losses,
        best_extractor_state=best_extractor_state,
        final_extractor_state=copy.deepcopy(trainable_extractor.state_dict()),
        final_train_losses=train_losses,
        final_val_losses=val_losses,
        history=history,
    )
