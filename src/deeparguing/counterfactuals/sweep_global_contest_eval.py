"""Sweep ``contest_all.py``'s first-N-misclassified-samples truncation across
several N, and record each resulting ``model.A``'s global test-set impact --
one CSV row per N.

This is purely an orchestration script: it does not reimplement anything.
Per N it reuses, in order:
  - ``load_fitted_model_and_data``/``load_all_samples`` (``run_contest.py``)
    to rebuild the model once and pull the first N misclassified samples
    out of the QBAF export, exactly like ``contest_all.py --num-samples``.
  - ``batch_contest`` (``batch_contest.py``) to jointly contest those N
    samples against a fresh copy of the baseline ``model.A``.
  - the touched-edge old/new-weight bookkeeping ``contest_all.py`` already
    computes for its own JSON log, to get weight-delta and edge-reversal
    stats.
  - ``compute_baseline_metrics``/``evaluate_contested_model``
    (``evals/global_contest_eval.py``) to score the resulting adjacency on
    the full held-out split relative to the (once-computed) baseline.

Usage::

    python -m deeparguing.counterfactuals.sweep_global_contest_eval
    python -m deeparguing.counterfactuals.sweep_global_contest_eval \\
        --ns 0,1,10,100 --seed 0 --output outputs/contestation/global_eval_sweep.csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import Tensor

from deeparguing.counterfactuals.batch_contest import (ALPHA_INIT,
                                                         DIVERGENCE_BOUND,
                                                         MAX_BACKTRACKS, TOL,
                                                         BatchContestResult,
                                                         batch_contest)
from deeparguing.counterfactuals.contest import (DEFAULT_K, MARGIN,
                                                   MAX_ITERS, THRESHOLD)
from deeparguing.counterfactuals.contest_all import (DEFAULT_CONFIG_PATH,
                                                       _load_config)
from deeparguing.counterfactuals.run_contest import (load_all_samples,
                                                       load_fitted_model_and_data)
from deeparguing.evals.global_contest_eval import (compute_baseline_metrics,
                                                     evaluate_contested_model)

DEFAULT_NS = [0, 1, 5, 10, 25, 50, 100]
DEFAULT_OUTPUT = "outputs/contestation/global_eval_sweep.csv"
DEFAULT_SEED = 0
CSV_DECIMALS = 3

# Single-series line chart -- see dataviz skill's reference palette.
_PLOT_LINE_COLOR = "#2a78d6"
_PLOT_SURFACE = "#fcfcfb"
_PLOT_GRIDLINE = "#e1e0d9"
_PLOT_AXIS = "#c3c2b7"
_PLOT_TICK_INK = "#898781"
_PLOT_PRIMARY_INK = "#0b0b0b"
_PLOT_SECONDARY_INK = "#52514e"


def _plot_acc_drop_vs_n(df: pd.DataFrame, png_path: Path) -> None:
    """Line chart of accuracy drop vs. N, each point labeled with how many
    of that N's contested samples flipped to correctly classified."""
    df = df.sort_values("N")

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=_PLOT_SURFACE)
    ax.set_facecolor(_PLOT_SURFACE)

    ax.plot(
        df["N"], df["acc_drop"],
        color=_PLOT_LINE_COLOR, linewidth=2, zorder=3,
        marker="o", markersize=8, markerfacecolor=_PLOT_LINE_COLOR,
        markeredgecolor=_PLOT_SURFACE, markeredgewidth=1.5,
    )
    for n, drop, flipped in zip(df["N"], df["acc_drop"], df["samples_flipped"]):
        ax.annotate(
            str(int(flipped)), (n, drop),
            textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=9, color=_PLOT_SECONDARY_INK,
        )

    ax.set_xlabel("N (samples contested)", color=_PLOT_PRIMARY_INK)
    ax.set_ylabel("Accuracy drop (baseline - contested)", color=_PLOT_PRIMARY_INK)
    ax.set_title(
        "Global accuracy drop vs. samples contested\n(label = samples flipped)",
        color=_PLOT_PRIMARY_INK, fontsize=11,
    )
    ax.grid(True, color=_PLOT_GRIDLINE, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_PLOT_AXIS)
    ax.spines["bottom"].set_color(_PLOT_AXIS)
    ax.tick_params(colors=_PLOT_TICK_INK)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def _touched_edge_old_new(
    original_A: Tensor, contested_A: Tensor, touched_edge_indices: list[int]
) -> tuple[list[float], list[float]]:
    """Old/new weight for each edge ``batch_contest`` touched -- same lookup
    ``contest_all.py`` does for its per-run JSON log, reused here for the
    weight-delta/edge-reversal columns."""
    original_flat = original_A.reshape(-1)
    contested_flat = contested_A.reshape(-1)
    olds = [original_flat[i].item() for i in touched_edge_indices]
    news = [contested_flat[i].item() for i in touched_edge_indices]
    return olds, news


def contest_n(
    model, original_A: Tensor, samples: Tensor, true_classes: list[int], contest_kwargs: dict[str, Any]
) -> tuple[BatchContestResult | None, Tensor]:
    """Reset ``model.A`` to a fresh copy of the baseline, then contest the
    given samples (already truncated to the first N misclassified ones by
    the caller). N=0 (empty ``samples``) skips ``batch_contest`` entirely --
    an empty batch has nothing to jointly optimize -- and just returns the
    untouched baseline adjacency."""
    model.A = original_A.clone()
    if samples.shape[0] == 0:
        return None, original_A.clone()
    result = batch_contest(model, samples, true_classes, **contest_kwargs)
    return result, model.A.detach().clone()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="YAML file holding batch_contest hyperparameters and checkpoint/qbaf paths "
        "(see tuning/contest/contest.yaml). --checkpoint/--qbaf/--device override it.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--qbaf", default=None)
    parser.add_argument(
        "--split", default="test", choices=["test", "val"],
        help="Held-out split (from the checkpoint's own data config) to compute global metrics over.",
    )
    parser.add_argument(
        "--ns",
        default=",".join(str(n) for n in DEFAULT_NS),
        help=f"Comma-separated list of N values. Default: {','.join(str(n) for n in DEFAULT_NS)}.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--batch-eval-size", type=int, default=None,
        help="batch_size for the global-eval forward pass (independent of batch_contest's own batch_size).",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--log", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper(), format="%(asctime)s - %(levelname)s - %(message)s")

    torch.manual_seed(args.seed)

    config = _load_config(args.config)
    device = args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or config.get("checkpoint")
    qbaf_path = args.qbaf or config.get("qbaf")
    if not checkpoint or not qbaf_path:
        raise ValueError(
            "checkpoint/qbaf not given on the command line and not set in "
            f"{args.config} -- add them there or pass --checkpoint/--qbaf."
        )

    contest_kwargs = dict(
        k=config.get("k", DEFAULT_K),
        threshold=config.get("threshold", THRESHOLD),
        margin=config.get("margin", MARGIN),
        max_iters=config.get("max_iters", MAX_ITERS),
        tol=config.get("tol", TOL),
        max_edits=config.get("max_edits"),
        batch_size=config.get("batch_size"),
        divergence_bound=config.get("divergence_bound", DIVERGENCE_BOUND),
        alpha_init=config.get("alpha_init", ALPHA_INIT),
        max_backtracks=config.get("max_backtracks", MAX_BACKTRACKS),
    )

    ns = [int(n) for n in args.ns.split(",")]

    logging.info(f"Loading baseline checkpoint from {checkpoint} ...")
    model, data_dict = load_fitted_model_and_data(checkpoint, device)
    X = data_dict[f"X_{args.split}"]
    y = data_dict[f"y_{args.split}"]
    original_A = model.A.detach().clone()

    logging.info(f"Computing baseline global metrics on the {args.split} split ({X.shape[0]} samples)...")
    baseline = compute_baseline_metrics(model, X, y, batch_size=args.batch_eval_size)
    logging.info(f"Baseline global accuracy: {baseline.accuracy:.4f}")

    with open(qbaf_path, "r", encoding="utf-8") as f:
        qbaf_data = json.load(f)
    all_samples, all_true_classes = load_all_samples(qbaf_data, device, num_samples=None)
    max_n = all_samples.shape[0]

    rows = []
    for n in ns:
        if n > max_n:
            logging.warning(f"N={n} exceeds the {max_n} misclassified samples available -- using all {max_n}.")
        capped_n = min(n, max_n)
        samples = all_samples[:capped_n]
        true_classes = all_true_classes[:capped_n]

        result, contested_A = contest_n(model, original_A, samples, true_classes, contest_kwargs)

        if result is None:
            samples_flipped = 0
            mean_weight_delta = 0.0
            max_weight_delta = 0.0
            max_strength = float("nan")
            edge_reversals = 0
        else:
            olds, news = _touched_edge_old_new(original_A, contested_A, result.touched_edge_indices)
            abs_deltas = [abs(new - old) for old, new in zip(olds, news)]
            samples_flipped = result.num_cleared
            mean_weight_delta = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0
            max_weight_delta = max(abs_deltas) if abs_deltas else 0.0
            max_strength = result.final_target_strengths.max().item()
            edge_reversals = sum(1 for old, new in zip(olds, news) if old * new < 0)

        eval_result = evaluate_contested_model(
            model, contested_A, X, y, baseline, batch_size=args.batch_eval_size
        )
        global_acc = eval_result.metrics.accuracy
        acc_drop = baseline.accuracy - global_acc

        logging.info(
            f"N={n}: samples_flipped={samples_flipped} global_acc={global_acc:.4f} "
            f"acc_drop={acc_drop:+.4f} mean_weight_delta={mean_weight_delta:.6f} "
            f"max_weight_delta={max_weight_delta:.6f} max_strength={max_strength:.4f} "
            f"edge_reversals={edge_reversals}"
        )

        rows.append(
            {
                "N": n,
                "samples_flipped": samples_flipped,
                "global_acc": global_acc,
                "acc_drop": acc_drop,
                "mean_weight_delta": mean_weight_delta,
                "max_weight_delta": max_weight_delta,
                "max_strength": max_strength,
                "edge_reversals": edge_reversals,
            }
        )

    model.A = original_A  # leave the shared model exactly as it was loaded

    df = pd.DataFrame(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.round(CSV_DECIMALS).to_csv(output_path, index=False)
    logging.info(f"Wrote {len(rows)} rows to {output_path}")

    png_path = output_path.with_suffix(".png")
    _plot_acc_drop_vs_n(df, png_path)
    logging.info(f"Wrote accuracy-drop plot to {png_path}")


if __name__ == "__main__":
    main()
