"""Grid-sweep the five global-optimization-specific hyperparameters of
``global_optimize.py`` -- ``protect_lambda``, ``eval_every``, ``protect_margin``,
``protect_sample_size``, ``max_acc_drop`` -- against a single fixed baseline
checkpoint, and report each combo's outcome (samples flipped, held-out
accuracy drop, edges changed, guardrail behavior) in a markdown report.

Purely an orchestration script: for each combo it resets ``model.A`` to a
fresh copy of the baseline and calls ``global_optimize()`` unchanged (see
``global_optimize.py``'s module docstring for the two-mechanism algorithm
being swept). The other, non-swept ``batch_contest`` hyperparameters
(k/threshold/margin/max_iters/etc.) come from the same YAML config
``global_optimize.py`` itself uses, or CLI overrides, and stay fixed across
every combo within one sweep run.

Usage::

    python -m deeparguing.contest.sweep_global_optimize
    python -m deeparguing.contest.sweep_global_optimize \\
        --protect-lambdas 1,5,20,50 --eval-everys 1,5,10 \\
        --max-combos 40 --output global_optimize_sweep.md

Grid size is the product of every ``--*-s`` list's length -- five 3-value
lists is already 243 full ``global_optimize`` runs. Pass ``--max-combos`` to
randomly subsample (seeded by ``--seed``) instead of running the full grid.
"""

import argparse
import itertools
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from deeparguing.contest.contest import DEFAULT_K, MARGIN, MAX_ITERS, THRESHOLD
from deeparguing.contest.batch_contest import ALPHA_INIT, DIVERGENCE_BOUND, MAX_BACKTRACKS, TOL
from deeparguing.contest.contest_all import _load_config, _required, _resolved
from deeparguing.contest.global_optimize import (DEFAULT_CONFIG_PATH,
                                                           DEFAULT_EVAL_SPLIT,
                                                           global_optimize)
from deeparguing.contest.run_contest import (load_all_samples,
                                                       load_fitted_model_and_data)
from deeparguing.evals.global_contest_eval import compute_baseline_metrics
from deeparguing.output_paths import resolve_read_path, resolve_write_path

DEFAULT_OUTPUT = "global_optimize_sweep.md"
DEFAULT_SEED = 0
CSV_DECIMALS = 6
# Above this many combos in the full grid, nudge the user toward --max-combos
# instead of silently launching a very long run.
LARGE_GRID_WARNING_THRESHOLD = 100

# Candidate values for each swept hyperparameter. Deliberately small (2 each,
# 32 combos total) so a bare invocation stays tractable -- override any of
# the five ``--*-s`` flags with a longer comma-separated list to broaden the
# search once you know how much compute budget you have.
DEFAULT_PROTECT_LAMBDAS = [5.0, 20.0]
DEFAULT_EVAL_EVERYS = [1, 5]
DEFAULT_PROTECT_MARGINS = [0.01, 0.05]
DEFAULT_PROTECT_SAMPLE_SIZES = [100, 200]
DEFAULT_MAX_ACC_DROPS = [0.01, 0.02]

SWEEP_COLUMNS = [
    "protect_lambda", "eval_every", "protect_margin",
    "protect_sample_size", "max_acc_drop",
]


@dataclass(frozen=True)
class Combo:
    protect_lambda: float
    eval_every: int
    protect_margin: float
    protect_sample_size: int
    max_acc_drop: float


def _parse_floats(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",")]


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",")]


def _run_combo(
    model, original_A, samples, true_classes, X_eval, y_eval,
    combo: Combo, fixed_kwargs: dict[str, Any], seed: int,
) -> tuple[Any, float]:
    """Reset ``model.A`` to the baseline and run ``global_optimize`` for one
    combo. Reseeded before every combo (not just once for the whole sweep)
    so that the protect-set sampling inside ``global_optimize`` starts from
    the same RNG state each time -- otherwise two combos' results could
    differ because of RNG drift accumulated by earlier combos in the sweep,
    not because of the hyperparameters actually being compared.
    """
    model.A = original_A.clone()
    torch.manual_seed(seed)
    start = time.perf_counter()
    result = global_optimize(
        model, samples, true_classes, X_eval, y_eval,
        protect_lambda=combo.protect_lambda,
        eval_every=combo.eval_every,
        protect_margin=combo.protect_margin,
        protect_sample_size=combo.protect_sample_size,
        max_acc_drop=combo.max_acc_drop,
        **fixed_kwargs,
    )
    elapsed = time.perf_counter() - start
    return result, elapsed


def _row_from_result(combo: Combo, result, elapsed: float) -> dict[str, Any]:
    br = result.batch_result
    return {
        "protect_lambda": combo.protect_lambda,
        "eval_every": combo.eval_every,
        "protect_margin": combo.protect_margin,
        "protect_sample_size": combo.protect_sample_size,
        "max_acc_drop": combo.max_acc_drop,
        "num_cleared": br.num_cleared,
        "num_total": br.num_total,
        "success_rate": br.num_cleared / max(1, br.num_total),
        "final_accuracy": result.final_metrics.accuracy,
        "acc_drop": result.acc_drop,
        "num_edges_changed": br.num_edges_changed,
        "rolled_back": result.rolled_back,
        "stopped_reason": result.stopped_reason,
        "num_rounds": len(result.rounds),
        "elapsed_sec": elapsed,
    }


def _full_results_table(df: pd.DataFrame) -> str:
    header = (
        "| protect_lambda | eval_every | protect_margin | protect_sample_size | max_acc_drop | "
        "cleared/total | success_rate | final_acc | acc_drop | edges_changed | rolled_back | "
        "stopped_reason | elapsed (s) |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['protect_lambda']:g} | {r['eval_every']:g} | {r['protect_margin']:g} | "
            f"{r['protect_sample_size']:g} | {r['max_acc_drop']:g} | "
            f"{r['num_cleared']:g}/{r['num_total']:g} | {r['success_rate']:.1%} | "
            f"{r['final_accuracy']:.4f} | {r['acc_drop']:+.4f} | {r['num_edges_changed']:g} | "
            f"{'yes' if r['rolled_back'] else 'no'} | {r['stopped_reason']} | {r['elapsed_sec']:.1f} |"
        )
    return "\n".join(lines)


def _marginal_tables(df: pd.DataFrame) -> str:
    """For each swept hyperparameter, average outcomes across every other
    axis -- a lightweight way to see which knob actually moves the needle,
    since the full grid table alone is hard to read once there are dozens
    of rows."""
    sections = []
    for col in SWEEP_COLUMNS:
        grouped = (
            df.groupby(col)
            .agg(
                mean_success_rate=("success_rate", "mean"),
                mean_acc_drop=("acc_drop", "mean"),
                mean_edges_changed=("num_edges_changed", "mean"),
                rollback_rate=("rolled_back", "mean"),
                n=("success_rate", "count"),
            )
            .reset_index()
            .sort_values(col)
        )
        lines = [
            f"### {col}",
            "",
            "| Value | Mean success rate | Mean acc drop | Mean edges changed | Rollback rate | N |",
            "|---|---|---|---|---|---|",
        ]
        for _, r in grouped.iterrows():
            lines.append(
                f"| {r[col]:g} | {r['mean_success_rate']:.1%} | {r['mean_acc_drop']:+.4f} | "
                f"{r['mean_edges_changed']:.1f} | {r['rollback_rate']:.0%} | {int(r['n'])} |"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _write_report(
    output_path: Path, df_sorted: pd.DataFrame, baseline_accuracy: float,
    eval_split: str, checkpoint: str, qbaf: str, num_samples: int | None,
    seed: int, fixed_kwargs: dict[str, Any], total_grid: int, num_run: int,
    total_elapsed: float,
) -> None:
    best = df_sorted.iloc[0]
    fixed_str = ", ".join(f"{k}={v}" for k, v in fixed_kwargs.items())
    lines = [
        "# Global Optimize Hyperparameter Sweep",
        "",
        f"Run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Checkpoint: {checkpoint}",
        f"QBAF: {qbaf} (num_samples={num_samples if num_samples is not None else 'all'})",
        f"Eval split: {eval_split}",
        f"Seed: {seed}",
        f"Fixed (non-swept) batch_contest params: {fixed_str}",
        f"Combos run: {num_run}/{total_grid} in the full grid",
        f"Total sweep wall time: {total_elapsed:.1f}s",
        "",
        "## Baseline",
        "",
        f"Baseline {eval_split} accuracy: {baseline_accuracy:.4f}",
        "",
        "## Best combo",
        "",
        "Ranked by samples cleared (desc), then accuracy drop (asc).",
        "",
        f"- protect_lambda={best['protect_lambda']:g}, eval_every={best['eval_every']:g}, "
        f"protect_margin={best['protect_margin']:g}, protect_sample_size={best['protect_sample_size']:g}, "
        f"max_acc_drop={best['max_acc_drop']:g}",
        f"- Cleared {best['num_cleared']:g}/{best['num_total']:g} samples ({best['success_rate']:.1%}), "
        f"{best['num_edges_changed']:g} edges changed",
        f"- Final accuracy: {best['final_accuracy']:.4f} (drop {best['acc_drop']:+.4f}), "
        f"rolled_back={best['rolled_back']}, stopped_reason={best['stopped_reason']}",
        "",
        "## Full results",
        "",
        _full_results_table(df_sorted),
        "",
        "## Marginal effect of each hyperparameter",
        "",
        "Mean outcome per value, averaged across every other swept axis.",
        "",
        _marginal_tables(df_sorted),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH,
        help="YAML file holding the fixed (non-swept) hyperparameters and paths "
        "(see tuning/contest/global_optimize.yaml). Any other flag overrides it.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--qbaf", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--device", default=None)

    # Fixed (non-swept) batch_contest hyperparameters -- same as global_optimize.py.
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--max-edits", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--divergence-bound", type=float, default=None)
    parser.add_argument("--alpha-init", type=float, default=None)
    parser.add_argument("--max-backtracks", type=int, default=None)
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)

    # The five swept global-optimization-specific hyperparameters.
    parser.add_argument(
        "--protect-lambdas",
        default=",".join(str(x) for x in DEFAULT_PROTECT_LAMBDAS),
        help=f"Comma-separated protect_lambda values. Default: {DEFAULT_PROTECT_LAMBDAS}.",
    )
    parser.add_argument(
        "--eval-everys",
        default=",".join(str(x) for x in DEFAULT_EVAL_EVERYS),
        help=f"Comma-separated eval_every values. Default: {DEFAULT_EVAL_EVERYS}.",
    )
    parser.add_argument(
        "--protect-margins",
        default=",".join(str(x) for x in DEFAULT_PROTECT_MARGINS),
        help=f"Comma-separated protect_margin values. Default: {DEFAULT_PROTECT_MARGINS}.",
    )
    parser.add_argument(
        "--protect-sample-sizes",
        default=",".join(str(x) for x in DEFAULT_PROTECT_SAMPLE_SIZES),
        help=f"Comma-separated protect_sample_size values. Default: {DEFAULT_PROTECT_SAMPLE_SIZES}.",
    )
    parser.add_argument(
        "--max-acc-drops",
        default=",".join(str(x) for x in DEFAULT_MAX_ACC_DROPS),
        help=f"Comma-separated max_acc_drop values. Default: {DEFAULT_MAX_ACC_DROPS}.",
    )

    parser.add_argument(
        "--max-combos", type=int, default=None,
        help="If given and smaller than the full grid, randomly subsample this "
        "many combos (seeded by --seed) instead of running every combination.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = _load_config(args.config)

    checkpoint = resolve_read_path(_required(args.checkpoint, config, "checkpoint", args.config))
    qbaf = resolve_read_path(_required(args.qbaf, config, "qbaf", args.config))
    num_samples = _resolved(args.num_samples, config, "num_samples", None)
    k = _resolved(args.k, config, "k", DEFAULT_K)
    threshold = _resolved(args.threshold, config, "threshold", THRESHOLD)
    margin = _resolved(args.margin, config, "margin", MARGIN)
    max_iters = _resolved(args.max_iters, config, "max_iters", MAX_ITERS)
    tol = _resolved(args.tol, config, "tol", TOL)
    max_edits = _resolved(args.max_edits, config, "max_edits", None)
    batch_size = _resolved(args.batch_size, config, "batch_size", None)
    divergence_bound = _resolved(args.divergence_bound, config, "divergence_bound", DIVERGENCE_BOUND)
    alpha_init = _resolved(args.alpha_init, config, "alpha_init", ALPHA_INIT)
    max_backtracks = _resolved(args.max_backtracks, config, "max_backtracks", MAX_BACKTRACKS)
    eval_split = _resolved(args.eval_split, config, "eval_split", DEFAULT_EVAL_SPLIT)
    if eval_split not in ("val", "test"):
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}.")
    eval_batch_size = _resolved(args.eval_batch_size, config, "eval_batch_size", None)
    device = _resolved(args.device, config, "device", "cuda" if torch.cuda.is_available() else "cpu")

    protect_lambdas = _parse_floats(args.protect_lambdas)
    eval_everys = _parse_ints(args.eval_everys)
    protect_margins = _parse_floats(args.protect_margins)
    protect_sample_sizes = _parse_ints(args.protect_sample_sizes)
    max_acc_drops = _parse_floats(args.max_acc_drops)

    combos = [
        Combo(*c) for c in itertools.product(
            protect_lambdas, eval_everys, protect_margins, protect_sample_sizes, max_acc_drops,
        )
    ]
    total_grid = len(combos)

    rng = random.Random(args.seed)
    if args.max_combos is not None and args.max_combos < total_grid:
        combos = rng.sample(combos, args.max_combos)
        print(f"Sampling {len(combos)}/{total_grid} combos from the full grid (seed={args.seed}).")
    else:
        if total_grid > LARGE_GRID_WARNING_THRESHOLD:
            print(
                f"Warning: full grid is {total_grid} combos -- this may take a long time. "
                "Pass --max-combos to randomly subsample instead."
            )
        print(f"Running the full grid: {total_grid} combos.")

    with open(qbaf, "r", encoding="utf-8") as f:
        qbaf_data = json.load(f)

    print(f"Loading baseline checkpoint from {checkpoint} ...")
    model, data_dict = load_fitted_model_and_data(checkpoint, device)
    assert model.A is not None, "checkpoint's model was never fit()"
    original_A = model.A.detach().clone()
    samples, true_classes = load_all_samples(qbaf_data, device, num_samples)

    X_eval = data_dict[f"X_{eval_split}"]
    y_eval = data_dict[f"y_{eval_split}"]

    baseline = compute_baseline_metrics(model, X_eval, y_eval, batch_size=eval_batch_size)
    print(f"Baseline {eval_split} accuracy: {baseline.accuracy:.4f}")
    print(f"Sweeping over {samples.shape[0]} misclassified samples...")

    fixed_kwargs = dict(
        k=k, threshold=threshold, margin=margin, max_iters=max_iters, tol=tol,
        max_edits=max_edits, batch_size=batch_size, divergence_bound=divergence_bound,
        alpha_init=alpha_init, max_backtracks=max_backtracks, eval_batch_size=eval_batch_size,
    )

    rows = []
    best_row = None
    sweep_start = time.perf_counter()
    pbar = tqdm(combos, desc="sweep", unit="combo")
    for combo in pbar:
        result, elapsed = _run_combo(
            model, original_A, samples, true_classes, X_eval, y_eval,
            combo, fixed_kwargs, args.seed,
        )
        row = _row_from_result(combo, result, elapsed)
        rows.append(row)
        if best_row is None or (row["num_cleared"], -row["acc_drop"]) > (
            best_row["num_cleared"], -best_row["acc_drop"]
        ):
            best_row = row
        pbar.set_postfix(
            cleared=f"{row['num_cleared']}/{row['num_total']}",
            acc_drop=f"{row['acc_drop']:+.4f}",
            best_cleared=best_row["num_cleared"],
        )
    total_elapsed = time.perf_counter() - sweep_start

    model.A = original_A  # leave the shared model exactly as it was loaded

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(
        ["num_cleared", "acc_drop"], ascending=[False, True]
    ).reset_index(drop=True)

    report_path = Path(resolve_write_path(args.output))
    csv_path = report_path.with_suffix(".csv")
    df_sorted.round(CSV_DECIMALS).to_csv(csv_path, index=False)
    print(f"Wrote raw sweep data to {csv_path}")

    _write_report(
        report_path, df_sorted, baseline.accuracy, eval_split, checkpoint, qbaf,
        num_samples, args.seed, fixed_kwargs, total_grid, len(combos), total_elapsed,
    )
    print(f"Wrote sweep report to {report_path}")


if __name__ == "__main__":
    main()
