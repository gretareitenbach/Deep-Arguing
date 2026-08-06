"""Compute global (full-split) test metrics for a contested model, relative
to its uncontested baseline. Rebuilds the model + held-out split from the
baseline checkpoint, evaluates it, swaps in the contested checkpoint's
``A`` and evaluates again, and reports the delta.

Usage::

    python -m deeparguing.contest.scripts.run_global_contest_eval
    python -m deeparguing.contest.scripts.run_global_contest_eval \\
        --checkpoint model_checkpoint.pt \\
        --contested-checkpoint contested_checkpoint.pt \\
        --split test
"""

import argparse
import datetime
import logging

import pandas as pd
import torch
from numpy.typing import NDArray

from deeparguing.contest.scripts.run_contest import load_fitted_model_and_data
from deeparguing.evals.global_contest_eval import (GlobalContestEvalResult,
                                                     GlobalEvalMetrics,
                                                     compute_baseline_metrics,
                                                     evaluate_contested_model)
from deeparguing.md_log import write_markdown_log
from deeparguing.output_paths import resolve_read_path, resolve_write_path

DEFAULT_LOG_FILENAME = "global_contest_eval.md"


def load_model_and_split(checkpoint_path: str, device: str, split: str):
    """Rebuild the model + reload its fitted state from a checkpoint.

    Returns
    -------
    tuple
        (model, X_<split>, y_<split>).
    """
    model, data_dict = load_fitted_model_and_data(checkpoint_path, device)
    X = data_dict[f"X_{split}"]
    y = data_dict[f"y_{split}"]
    return model, X, y


def _metrics_table(baseline: GlobalEvalMetrics, result: GlobalContestEvalResult) -> str:
    """Markdown table of accuracy/precision/recall/f1 for baseline vs.
    contested, plus the signed delta."""
    deltas = {
        "Accuracy": result.delta_accuracy,
        "Precision": result.delta_precision,
        "Recall": result.delta_recall,
        "F1": result.delta_f1,
    }
    contested = result.metrics
    rows = [
        ("Accuracy", baseline.accuracy, contested.accuracy),
        ("Precision", baseline.precision, contested.precision),
        ("Recall", baseline.recall, contested.recall),
        ("F1", baseline.f1, contested.f1),
    ]
    lines = ["| Metric | Baseline | Contested | Delta |", "|---|---|---|---|"]
    for name, base_value, contested_value in rows:
        lines.append(
            f"| {name} | {base_value:.4f} | {contested_value:.4f} | {deltas[name]:+.4f} |"
        )
    return "\n".join(lines)


def _confusion_matrix_block(title: str, cm: NDArray) -> str:
    df = pd.DataFrame(
        cm,
        index=[f"Actual {i}" for i in range(cm.shape[0])],
        columns=[f"Pred {i}" for i in range(cm.shape[1])],
    )
    return f"{title} confusion matrix:\n```\n{df.to_string()}\n```"


def _confusion_matrix_delta_block(baseline_cm: NDArray, contested_cm: NDArray) -> str:
    """Contested-minus-baseline confusion matrix, cell by cell."""
    delta = contested_cm - baseline_cm
    df = pd.DataFrame(
        delta,
        index=[f"Actual {i}" for i in range(delta.shape[0])],
        columns=[f"Pred {i}" for i in range(delta.shape[1])],
    ).map(lambda v: f"{v:+d}")
    return f"Contested confusion matrix (delta from baseline):\n```\n{df.to_string()}\n```"


def _touched_edges(original_A: torch.Tensor, contested_A: torch.Tensor) -> list[dict]:
    """Every ``model.A`` entry that differs between baseline and contested,
    with its source/target/dim and old/new weight -- same edge addressing
    ``contest_all.py`` uses for its own touched-edges log."""
    n1, n2, d = original_A.shape
    original_flat = original_A.reshape(-1)
    contested_flat = contested_A.reshape(-1)
    diff_indices = (original_flat != contested_flat).nonzero(as_tuple=True)[0].tolist()
    return [
        {
            "edge_id": idx,
            "source": (idx // d) // n2,
            "target": (idx // d) % n2,
            "dim": idx % d,
            "old_weight": original_flat[idx].item(),
            "new_weight": contested_flat[idx].item(),
        }
        for idx in diff_indices
    ]


def _touched_edges_block(touched_edges: list[dict]) -> str:
    if not touched_edges:
        return "Touched edges: none (contested A is identical to baseline)."
    lines = [
        "Touched edges:",
        "| Source | Target | Dim | Old weight | New weight | Delta |",
        "|---|---|---|---|---|---|",
    ]
    for e in touched_edges:
        delta = e["new_weight"] - e["old_weight"]
        lines.append(
            f"| {e['source']} | {e['target']} | {e['dim']} | "
            f"{e['old_weight']:.6f} | {e['new_weight']:.6f} | {delta:+.6f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="model_checkpoint.pt",
        help="Baseline (uncontested) checkpoint produced by cli/run.py.",
    )
    parser.add_argument(
        "--contested-checkpoint",
        default="contested_checkpoint.pt",
        help="Checkpoint holding the contested model.A, produced by contest_all.py.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "val"],
        help="Which held-out split (from the baseline checkpoint's own data "
        "config) to compute global metrics over.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--log", default="info", choices=["debug", "info", "warning", "error"]
    )
    parser.add_argument(
        "--log-path",
        default=DEFAULT_LOG_FILENAME,
        help="Markdown file to append a results table to (created if missing). "
        f"Default: today's outputs/<date>/{DEFAULT_LOG_FILENAME}. Pass an "
        "empty string to skip logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log.upper(), format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args.checkpoint = resolve_read_path(args.checkpoint)
    args.contested_checkpoint = resolve_read_path(args.contested_checkpoint)

    logging.info(f"Loading baseline checkpoint from {args.checkpoint} ...")
    model, X, y = load_model_and_split(args.checkpoint, args.device, args.split)

    logging.info(
        f"Computing baseline metrics on the {args.split} split ({X.shape[0]} samples)..."
    )
    baseline = compute_baseline_metrics(model, X, y, batch_size=args.batch_size)

    logging.info(f"Loading contested adjacency from {args.contested_checkpoint} ...")
    contested_checkpoint = torch.load(args.contested_checkpoint, map_location=args.device)
    contested_A = contested_checkpoint["A"].to(args.device)
    original_A = model.A.detach().clone()

    result = evaluate_contested_model(
        model, contested_A, X, y, baseline, batch_size=args.batch_size
    )
    touched_edges = _touched_edges(original_A, contested_A)

    def _signed(value: float) -> str:
        return f"{value:+.4f}"

    logging.info(
        f"Baseline:  accuracy={baseline.accuracy:.4f} precision={baseline.precision:.4f} "
        f"recall={baseline.recall:.4f} f1={baseline.f1:.4f}"
    )
    logging.info(
        f"Contested: accuracy={result.metrics.accuracy:.4f} precision={result.metrics.precision:.4f} "
        f"recall={result.metrics.recall:.4f} f1={result.metrics.f1:.4f}"
    )
    logging.info(
        f"Delta:     accuracy={_signed(result.delta_accuracy)} precision={_signed(result.delta_precision)} "
        f"recall={_signed(result.delta_recall)} f1={_signed(result.delta_f1)}"
    )
    logging.info(f"Touched edges: {len(touched_edges)}")

    if args.log_path:
        log_path = resolve_write_path(args.log_path)
        write_markdown_log(
            [
                "--- GLOBAL CONTEST EVAL ---",
                f"Run: {datetime.datetime.now().isoformat(timespec='seconds')}",
                f"Baseline checkpoint: {args.checkpoint}",
                f"Contested checkpoint: {args.contested_checkpoint}",
                f"Split: {args.split} ({X.shape[0]} samples)",
                _metrics_table(baseline, result),
                _touched_edges_block(touched_edges),
                _confusion_matrix_block("Baseline", baseline.confusion_matrix),
                _confusion_matrix_delta_block(baseline.confusion_matrix, result.metrics.confusion_matrix),
            ],
            log_path,
        )
        logging.info(f"Appended results table to {log_path}")


if __name__ == "__main__":
    main()
