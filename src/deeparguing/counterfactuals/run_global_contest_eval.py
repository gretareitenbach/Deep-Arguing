"""Compute global (full-split) test metrics for a contested model, relative
to its uncontested baseline.

Consumes exactly what ``contest_all.py`` already produces in
``outputs/contestation/``: the *baseline* checkpoint it started from
(``outputs/checkpoints/model_checkpoint.pt``, from ``cli/run.py``) and the
*contested* checkpoint it wrote out (``outputs/contestation/contested_checkpoint.pt``,
same format plus the edited ``A``). Rebuilds the live model + held-out split
from the baseline checkpoint's own config, evaluates it once (the baseline),
swaps in the contested checkpoint's ``A`` and evaluates again, and reports
the delta -- see ``deeparguing.evals.global_contest_eval`` for the two calls
this wraps.

By default the results are also appended as a markdown table to
``outputs/logs/global_contest_eval.md`` (one section per run, mirroring
``deeparguing.md_log``'s convention used elsewhere -- e.g.
``outputs/logs/summary.md``); pass ``--log-path ""`` to skip writing it.

Usage::

    python -m deeparguing.counterfactuals.run_global_contest_eval
    python -m deeparguing.counterfactuals.run_global_contest_eval \\
        --checkpoint outputs/checkpoints/model_checkpoint.pt \\
        --contested-checkpoint outputs/contestation/contested_checkpoint.pt \\
        --split test
"""

import argparse
import datetime
import logging

import pandas as pd
import torch
from numpy.typing import NDArray

from deeparguing.cli.parse_yaml import parse_model_config, read_config_files
from deeparguing.evals.global_contest_eval import (GlobalContestEvalResult,
                                                     GlobalEvalMetrics,
                                                     compute_baseline_metrics,
                                                     evaluate_contested_model)
from deeparguing.md_log import write_markdown_log

DEFAULT_LOG_PATH = "outputs/logs/global_contest_eval.md"


def load_model_and_split(checkpoint_path: str, device: str, split: str):
    """Rebuild the model architecture + held-out data split from a
    checkpoint's config, and reload the model's fitted state (weights +
    ``A``/``X_train``/``y_train``/``default_indexes``) -- same approach as
    ``run_contest.load_model``, but also keeps the ``data_dict`` around
    since we need ``X_<split>``/``y_<split>`` here, not just the model.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = read_config_files(checkpoint["config_paths"])
    data_dict, instances = parse_model_config(model_config, trial=None, device=device)
    model = instances["model"]
    if hasattr(model, "to"):
        model = model.to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.A = checkpoint["A"].to(device)
    model.X_train = checkpoint["X_train"].to(device)
    model.y_train = checkpoint["y_train"].to(device)
    model.default_indexes = checkpoint["default_indexes"].to(device)
    model.eval()

    X = data_dict[f"X_{split}"]
    y = data_dict[f"y_{split}"]
    return model, X, y


def _metrics_table(baseline: GlobalEvalMetrics, result: GlobalContestEvalResult) -> str:
    """Markdown table of accuracy/precision/recall/f1 for baseline vs.
    contested, plus the signed delta -- the same four numbers logged to the
    console, rendered as a table instead of one line each."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/model_checkpoint.pt",
        help="Baseline (uncontested) checkpoint produced by cli/run.py.",
    )
    parser.add_argument(
        "--contested-checkpoint",
        default="outputs/contestation/contested_checkpoint.pt",
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
        default=DEFAULT_LOG_PATH,
        help="Markdown file to append a results table to (created if missing). "
        f"Default: {DEFAULT_LOG_PATH}. Pass an empty string to skip logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log.upper(), format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info(f"Loading baseline checkpoint from {args.checkpoint} ...")
    model, X, y = load_model_and_split(args.checkpoint, args.device, args.split)

    logging.info(
        f"Computing baseline metrics on the {args.split} split ({X.shape[0]} samples)..."
    )
    baseline = compute_baseline_metrics(model, X, y, batch_size=args.batch_size)

    logging.info(f"Loading contested adjacency from {args.contested_checkpoint} ...")
    contested_checkpoint = torch.load(args.contested_checkpoint, map_location=args.device)
    contested_A = contested_checkpoint["A"].to(args.device)

    result = evaluate_contested_model(
        model, contested_A, X, y, baseline, batch_size=args.batch_size
    )

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

    if args.log_path:
        write_markdown_log(
            [
                "--- GLOBAL CONTEST EVAL ---",
                f"Run: {datetime.datetime.now().isoformat(timespec='seconds')}",
                f"Baseline checkpoint: {args.checkpoint}",
                f"Contested checkpoint: {args.contested_checkpoint}",
                f"Split: {args.split} ({X.shape[0]} samples)",
                _metrics_table(baseline, result),
                _confusion_matrix_block("Baseline", baseline.confusion_matrix),
                _confusion_matrix_block("Contested", result.metrics.confusion_matrix),
            ],
            args.log_path,
        )
        logging.info(f"Appended results table to {args.log_path}")


if __name__ == "__main__":
    main()
