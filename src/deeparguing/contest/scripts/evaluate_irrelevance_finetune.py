"""Evaluate an irrelevance-finetuned checkpoint's real classification impact
on the held-out eval split. Everything ``contest.scripts.run_finetune`` logs
during training (correction/protect losses) only ever measures
``partial_order``'s raw output or a *sampled* protect set's margins -- this
is the first point in the pipeline that checks what fine-tuning actually did
to real classification accuracy, on the full eval split.

Compares two checkpoints, each evaluated by loading it exactly as saved and
running ``model(X_eval)`` -- i.e. what would actually happen if a new image
went through the model, no extra fitting or adjustment at eval time:

- ``baseline``: the pre-finetune checkpoint.
- ``finetuned``: ``contest.scripts.run_finetune``'s output. Since 2026-08-12 (see
  updates.md) that script recomputes ``model.A`` from the fine-tuned
  ``feature_weights_1`` before saving, so this checkpoint's ``A`` is already
  consistent with its own network -- no refit needed here. (Older
  checkpoints saved before that change still load and evaluate fine here,
  just with whatever ``A`` they were saved with -- this script always
  evaluates the checkpoint exactly as loaded, honestly reflecting what
  deploying that specific file would do.)

Hyperparameters and paths come from a YAML config file (default
``tuning/contest/evaluate_irrelevance_finetune.yaml``); any CLI flag
overrides the corresponding config value -- same pattern as
``contest.scripts.run_finetune``.

Usage::

    python -m deeparguing.contest.scripts.evaluate_irrelevance_finetune
    python -m deeparguing.contest.scripts.evaluate_irrelevance_finetune \\
        --finetuned-checkpoint outputs/12Aug2026/finetuned_checkpoint.pt --eval-split test
"""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from numpy.typing import NDArray

from deeparguing.contest.scripts.config_cli import load_config, required, resolved
from deeparguing.contest.scripts.run_contest import load_fitted_model_and_data
from deeparguing.evals.evals import evaluate_model
from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.md_log import write_markdown_log
from deeparguing.output_paths import resolve_read_path, today_output_dir

DEFAULT_CONFIG_PATH = "tuning/contest/evaluate_irrelevance_finetune.yaml"
DEFAULT_EVAL_SPLIT = "test"
DEFAULT_OUTPUT_FILENAME = "evaluate_irrelevance_finetune.md"


def _evaluate(
    label: str,
    model: GradualAACBR,
    X_eval: torch.Tensor,
    y_eval: torch.Tensor,
    batch_size: int | None,
) -> dict[str, Any]:
    """One row of the report: evaluate ``model`` exactly as loaded (no
    ``fit()`` call -- ``model.A`` is whatever the checkpoint saved) on
    ``X_eval``/``y_eval``, the same forward path a real prediction takes.
    """
    accuracy, precision, recall, f1, cm = evaluate_model(
        model, None, None, None, None,
        X_eval, y_eval, batch_size=batch_size, refit=False,
    )
    return {
        "label": label,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def _results_table(rows: list[dict[str, Any]]) -> str:
    header = "| Variant | Accuracy | Precision | Recall | F1 | Δ Accuracy vs baseline |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    baseline_accuracy = rows[0]["accuracy"]
    for r in rows:
        delta = r["accuracy"] - baseline_accuracy
        lines.append(
            f"| {r['label']} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['f1']:.4f} | {delta:+.4f} |"
        )
    return "\n".join(lines)


def _confusion_matrix_block(label: str, cm: NDArray) -> str:
    df = pd.DataFrame(
        cm,
        index=[f"Actual {i}" for i in range(cm.shape[0])],
        columns=[f"Pred {i}" for i in range(cm.shape[1])],
    )
    return f"{label} confusion matrix:\n```\n{df.to_string()}\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--finetuned-checkpoint", default=None)
    parser.add_argument(
        "--eval-split", default=None, choices=["val", "test"],
        help="Held-out split (from the checkpoint's own data config) to evaluate on.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Eval forward-pass batch size (default: whole split at once).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--output-filename", default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    baseline_checkpoint = resolve_read_path(
        required(args.baseline_checkpoint, config, "baseline_checkpoint", args.config)
    )
    finetuned_checkpoint = resolve_read_path(
        required(args.finetuned_checkpoint, config, "finetuned_checkpoint", args.config)
    )
    eval_split = resolved(args.eval_split, config, "eval_split", DEFAULT_EVAL_SPLIT)
    if eval_split not in ("val", "test"):
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}.")
    batch_size = resolved(args.batch_size, config, "batch_size", None)
    device = resolved(args.device, config, "device", "cuda" if torch.cuda.is_available() else "cpu")
    log_dir_str = resolved(args.log_dir, config, "log_dir", str(today_output_dir()))
    output_filename = resolved(args.output_filename, config, "output_filename", DEFAULT_OUTPUT_FILENAME)

    log_dir = Path(log_dir_str)
    log_dir.mkdir(parents=True, exist_ok=True)
    md_path = str(log_dir / output_filename)

    write_markdown_log(
        [
            "--- IRRELEVANCE FINETUNE EVALUATION ---",
            f"baseline_checkpoint={baseline_checkpoint}",
            f"finetuned_checkpoint={finetuned_checkpoint}",
            f"eval_split={eval_split}, batch_size={batch_size}, device={device}",
        ],
        md_path,
        mode="w",
    )

    print(f"Loading baseline checkpoint from {baseline_checkpoint} ...")
    baseline_model, baseline_data = load_fitted_model_and_data(baseline_checkpoint, device)
    X_eval, y_eval = baseline_data[f"X_{eval_split}"], baseline_data[f"y_{eval_split}"]
    print(f"Evaluating baseline on {X_eval.shape[0]} {eval_split} samples ...")
    baseline_row = _evaluate("baseline", baseline_model, X_eval, y_eval, batch_size)
    print(f"  accuracy={baseline_row['accuracy']:.4f} f1={baseline_row['f1']:.4f}")

    print(f"Loading finetuned checkpoint from {finetuned_checkpoint} ...")
    ft_model, ft_data = load_fitted_model_and_data(finetuned_checkpoint, device)
    X_eval_ft, y_eval_ft = ft_data[f"X_{eval_split}"], ft_data[f"y_{eval_split}"]
    print(f"Evaluating finetuned checkpoint on {X_eval_ft.shape[0]} {eval_split} samples ...")
    finetuned_row = _evaluate("finetuned", ft_model, X_eval_ft, y_eval_ft, batch_size)
    print(f"  accuracy={finetuned_row['accuracy']:.4f} f1={finetuned_row['f1']:.4f}")

    rows = [baseline_row, finetuned_row]

    write_markdown_log(
        [
            "--- RESULTS ---",
            _results_table(rows),
            "\n\n".join(_confusion_matrix_block(r["label"], r["confusion_matrix"]) for r in rows),
        ],
        md_path,
        mode="a",
    )
    print(f"Wrote report to {md_path}")


if __name__ == "__main__":
    main()
