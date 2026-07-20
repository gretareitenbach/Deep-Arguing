"""
src/deeparguing/evals/global_contest_eval.py

Global (full-test-set) metrics for a *contested* ``model.A``, reported
relative to a fixed baseline.

This module is deliberately agnostic to how the contested adjacency was
produced -- a single-sample ``contest()`` step, ``batch_contest``, or a
hand-crafted edit all just end up as a tensor the same shape as ``model.A``.
``evaluate_contested_model`` only ever swaps that tensor onto an
already-fitted model, evaluates it on the full test set the same way
``evaluate_model`` evaluates a freshly-fit one, and diffs the result against
a caller-supplied baseline. It never calls ``model.fit()`` -- doing so would
rebuild ``model.A`` from the casebase edge-weight functions and silently
discard whatever edits produced ``contested_A``.
"""

from dataclasses import dataclass

from numpy.typing import NDArray
from torch import Tensor

from deeparguing.evals.evals import evaluate_model
from deeparguing.gradual_aacbr import GradualAACBR


@dataclass(frozen=True)
class GlobalEvalMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: NDArray


@dataclass(frozen=True)
class GlobalContestEvalResult:
    metrics: GlobalEvalMetrics
    baseline: GlobalEvalMetrics
    delta_accuracy: float
    delta_precision: float
    delta_recall: float
    delta_f1: float


def compute_baseline_metrics(
    model: GradualAACBR,
    X_new_cases: Tensor,
    y_new_cases: Tensor,
    batch_size: int | None = None,
) -> GlobalEvalMetrics:
    """Evaluate ``model``'s current (uncontested) ``model.A`` on the full
    test set. Call this once, before any contestation edits are applied, to
    get the reference point ``evaluate_contested_model`` reports deltas
    against.
    """
    if model.A is None:
        raise Exception("Ensure the model has been fit first.")

    accuracy, precision, recall, f1, cm = evaluate_model(
        model,
        None,
        None,
        None,
        None,
        X_new_cases,
        y_new_cases,
        batch_size=batch_size,
        refit=False,
    )
    return GlobalEvalMetrics(accuracy, precision, recall, f1, cm)


def evaluate_contested_model(
    model: GradualAACBR,
    contested_A: Tensor,
    X_new_cases: Tensor,
    y_new_cases: Tensor,
    baseline: GlobalEvalMetrics,
    batch_size: int | None = None,
) -> GlobalContestEvalResult:
    """Swap ``contested_A`` onto ``model``, evaluate it on the full test
    set, and report the delta against ``baseline`` (e.g. from
    ``compute_baseline_metrics``).

    ``model`` must already be fitted (has ``X_train``/``default_indexes``
    from a prior ``fit()`` call). ``fit()`` is never called here, so this
    works no matter how ``contested_A`` was produced -- the function only
    looks at the tensor itself, temporarily standing in for ``model.A`` for
    the duration of the eval. ``model.A`` is restored afterwards regardless
    of success or failure.
    """
    if model.A is None:
        raise Exception("Ensure the model has been fit first.")

    original_A = model.A
    try:
        model.A = contested_A
        accuracy, precision, recall, f1, cm = evaluate_model(
            model,
            None,
            None,
            None,
            None,
            X_new_cases,
            y_new_cases,
            batch_size=batch_size,
            refit=False,
        )
    finally:
        model.A = original_A

    metrics = GlobalEvalMetrics(accuracy, precision, recall, f1, cm)
    return GlobalContestEvalResult(
        metrics=metrics,
        baseline=baseline,
        delta_accuracy=accuracy - baseline.accuracy,
        delta_precision=precision - baseline.precision,
        delta_recall=recall - baseline.recall,
        delta_f1=f1 - baseline.f1,
    )
