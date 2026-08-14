"""Grid-sweep ``contest.scripts.run_finetune``'s hyperparameters -- ``lr``,
``batch_size``, ``protect_margin``, ``protect_lambda``, ``protect_sample_size``,
``casebase_lambda`` -- against a single fixed baseline checkpoint/dataset, and
report each combo's outcome in a markdown report. ``chunk_size`` (a
memory/compute chunking knob, not a modeling choice -- see
``correction_loss``'s docstring) and ``steps`` (superseded by this sweep's
own best-checkpoint tracking, same as ``contest.scripts.run_finetune``'s -- see
below) are fixed, not swept. (A ``lam`` axis, for the now-removed
``preservation_loss`` term, was swept here through 2026-08-11 -- see
updates.md and ``deeparguing.casebase_edge_weights.finetune``'s module
docstring for why it's gone.)

``casebase_lambda`` axis added 2026-08-13, alongside ``contest.scripts.run_finetune``
making ``--casebase-dataset`` required and defaulting ``casebase_lambda`` to
1.0 -- that default was carried over from what ``run_full_pipeline.sh`` had
already been running (untuned), and the same day's full-pipeline run
regressed test accuracy relative to baseline, so ``casebase_lambda`` needs
its own comparison rather than staying fixed at a guessed value. Unlike the
other axes, sweeping it costs the same per step as toggling it on (every
step differentiably re-fits ``model.A`` -- see ``casebase_correction_loss``'s
docstring), so this is more expensive per combo than the pre-existing axes.

Motivation (2026-08-11, see updates.md): the first real fine-tune run showed
val ``combined`` loss plateauing/overfitting well before its final step, at
fixed ``lr=0.001``. Before trusting any hyperparameter comparison, each combo
here is trained for a fixed, generous ``--steps`` budget and ranked by the
*best* val ``combined`` loss it reached along the way (via ``run_finetune``'s
built-in tracking -- the same mechanism ``contest.scripts.run_finetune`` uses to
pick its saved checkpoint), not by whatever the final step happened to land
on. That keeps the comparison fair across combos that converge at different
rates.

Usage::

    python -m deeparguing.contest.sweeps.sweep_finetune
    python -m deeparguing.contest.sweeps.sweep_finetune \\
        --lrs 0.0001,0.0003,0.001 --protect-lambdas 0,1,5 --max-combos 40 \\
        --output finetune_sweep.md
"""

import argparse
import copy
import itertools
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from deeparguing.casebase_edge_weights.finetune import (
    TRAINABLE_FEATURE_EXTRACTOR_INDEX, CasebaseFinetuneConfig,
    assert_shares_partial_order, freeze_all_except_trainable, run_finetune)
from deeparguing.contest.global_optimize import _build_protect_set
from deeparguing.contest.scripts.config_cli import load_config, required, resolved
from deeparguing.contest.scripts.run_finetune import (
    DEFAULT_CHUNK_SIZE, DEFAULT_CONFIG_PATH, DEFAULT_EVAL_SPLIT,
    DEFAULT_LOG_EVERY, DEFAULT_SEED, DEFAULT_STEPS)
from deeparguing.contest.scripts.run_contest import load_fitted_model_and_data
from deeparguing.output_paths import resolve_read_path, resolve_write_path

DEFAULT_OUTPUT = "finetune_sweep.md"
CSV_DECIMALS = 6
LARGE_GRID_WARNING_THRESHOLD = 100  # above this many combos, nudge toward --max-combos

# Candidate values for each swept hyperparameter. lr gets 3 points since
# it's the most consequential axis (see updates.md); the rest get 2.
DEFAULT_LRS = [3e-4, 1e-3, 3e-3]
DEFAULT_BATCH_SIZES = [64, 128]  # 0 means full-batch, see _parse_batch_sizes
DEFAULT_PROTECT_MARGINS = [0.01, 0.05]
DEFAULT_PROTECT_LAMBDAS = [0.0, 1.0]
DEFAULT_PROTECT_SAMPLE_SIZES = [100, 200]
DEFAULT_CASEBASE_LAMBDAS = [0.0, 0.3, 1.0, 3.0]

SWEEP_COLUMNS = [
    "lr", "batch_size", "protect_margin", "protect_lambda", "protect_sample_size",
    "casebase_lambda",
]


@dataclass(frozen=True)
class Combo:
    lr: float
    batch_size: int | None
    protect_margin: float
    protect_lambda: float
    protect_sample_size: int
    casebase_lambda: float


def _parse_floats(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",")]


def _parse_batch_sizes(raw: str) -> list[int | None]:
    """``0`` means full-batch (``run_finetune``'s ``batch_size=None``)."""
    return [None if int(x) == 0 else int(x) for x in raw.split(",")]


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",")]


def _run_combo(
    model, trainable_extractor, original_extractor_state: dict[str, Any],
    train_tensors: dict[str, torch.Tensor], val_tensors: dict[str, torch.Tensor],
    X_eval: torch.Tensor, y_eval: torch.Tensor,
    casebase_edges: dict[str, dict[str, torch.Tensor]] | None,
    combo: Combo, steps: int, chunk_size: int, log_every: int, seed: int,
) -> tuple[Any, int, float]:
    """Reset the trainable extractor to the pre-finetune baseline and run one
    combo. The protect set is rebuilt fresh per combo (its
    ``protect_sample_size`` is itself swept, and rebuilding is cheap -- one
    forward pass), reseeding first so every combo's sampling starts from the
    same RNG state. ``casebase_edges`` (fixed across combos -- only
    ``combo.casebase_lambda`` varies) is ``None`` when no
    ``--casebase-dataset`` was given, in which case ``casebase_correction``
    is always 0, same as ``contest.scripts.run_finetune`` before 2026-08-13.

    Returns
    -------
    tuple[Any, int, float]
        ``run_finetune``'s ``FinetuneRunResult``, the number of protect
        samples actually used, and elapsed wall time in seconds.
    """
    trainable_extractor.load_state_dict(original_extractor_state)
    torch.manual_seed(seed)
    protect_samples, protect_target_classes = _build_protect_set(
        model, X_eval, y_eval, combo.protect_sample_size
    )
    casebase_config = (
        CasebaseFinetuneConfig(casebase_lambda=combo.casebase_lambda, **casebase_edges)
        if casebase_edges is not None
        else None
    )
    start = time.perf_counter()
    result = run_finetune(
        model, train_tensors, val_tensors,
        combo.lr, steps, combo.batch_size, chunk_size,
        protect_samples, protect_target_classes, combo.protect_margin, combo.protect_lambda,
        casebase_config=casebase_config,
        log_every=log_every, show_progress=False,
    )
    elapsed = time.perf_counter() - start
    return result, protect_samples.shape[0], elapsed


def _row_from_result(combo: Combo, result, num_protect_samples: int, elapsed: float, steps: int) -> dict[str, Any]:
    has_val = result.best_val_losses is not None
    return {
        "lr": combo.lr,
        "batch_size": combo.batch_size if combo.batch_size is not None else 0,
        "protect_margin": combo.protect_margin,
        "protect_lambda": combo.protect_lambda,
        "protect_sample_size": combo.protect_sample_size,
        "casebase_lambda": combo.casebase_lambda,
        "num_protect_samples": num_protect_samples,
        "best_step": result.best_step if result.best_step is not None else steps,
        "best_val_combined": result.best_val_losses.combined.item() if has_val else float("nan"),
        "best_val_correction": result.best_val_losses.correction.item() if has_val else float("nan"),
        "best_val_protect": result.best_val_losses.protect.item() if has_val else float("nan"),
        "best_val_casebase_correction": (
            result.best_val_losses.casebase_correction.item() if has_val else float("nan")
        ),
        "final_val_combined": (
            result.final_val_losses.combined.item() if result.final_val_losses is not None else float("nan")
        ),
        "final_train_combined": result.final_train_losses.combined.item(),
        "overfit_gap": (
            result.final_train_losses.combined.item() - result.best_val_losses.combined.item() if has_val
            else float("nan")
        ),
        "elapsed_sec": elapsed,
    }


def _full_results_table(df: pd.DataFrame) -> str:
    header = (
        "| lr | batch_size | protect_margin | protect_lambda | protect_sample_size | "
        "casebase_lambda | best_step | best_val_combined | best_val_casebase_correction | "
        "final_val_combined | final_train_combined | overfit_gap | elapsed (s) |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['lr']:g} | {'full' if r['batch_size'] == 0 else int(r['batch_size'])} | "
            f"{r['protect_margin']:g} | {r['protect_lambda']:g} | {r['protect_sample_size']:g} | "
            f"{r['casebase_lambda']:g} | "
            f"{r['best_step']:g} | {r['best_val_combined']:.6f} | {r['best_val_casebase_correction']:.6f} | "
            f"{r['final_val_combined']:.6f} | "
            f"{r['final_train_combined']:.6f} | {r['overfit_gap']:+.6f} | {r['elapsed_sec']:.1f} |"
        )
    return "\n".join(lines)


def _marginal_tables(df: pd.DataFrame) -> str:
    """For each swept hyperparameter, average outcomes across every other axis."""
    sections = []
    for col in SWEEP_COLUMNS:
        grouped = (
            df.groupby(col)
            .agg(
                mean_best_val_combined=("best_val_combined", "mean"),
                mean_overfit_gap=("overfit_gap", "mean"),
                mean_best_step=("best_step", "mean"),
                n=("best_val_combined", "count"),
            )
            .reset_index()
            .sort_values(col)
        )
        lines = [
            f"### {col}",
            "",
            "| Value | Mean best val combined | Mean overfit gap | Mean best step | N |",
            "|---|---|---|---|---|",
        ]
        for _, r in grouped.iterrows():
            value = "full" if col == "batch_size" and r[col] == 0 else f"{r[col]:g}"
            lines.append(
                f"| {value} | {r['mean_best_val_combined']:.6f} | {r['mean_overfit_gap']:+.6f} | "
                f"{r['mean_best_step']:.1f} | {int(r['n'])} |"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _write_report(
    output_path: Path, df_sorted: pd.DataFrame, eval_split: str, checkpoint: str, dataset: str,
    seed: int, steps: int, chunk_size: int, total_grid: int, num_run: int, total_elapsed: float,
) -> None:
    best = df_sorted.iloc[0]
    lines = [
        "# Finetune Irrelevance Hyperparameter Sweep",
        "",
        f"Run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Checkpoint: {checkpoint}",
        f"Dataset: {dataset}",
        f"Eval split: {eval_split}",
        f"Seed: {seed}",
        f"Fixed (non-swept): steps={steps}, chunk_size={chunk_size}",
        f"Combos run: {num_run}/{total_grid} in the full grid",
        f"Total sweep wall time: {total_elapsed:.1f}s",
        "",
        "## Best combo",
        "",
        "Ranked by best val combined loss reached during training (lower is better; "
        "each combo picks its own best step, not just its final one -- see module docstring).",
        "",
        f"- lr={best['lr']:g}, "
        f"batch_size={'full' if best['batch_size'] == 0 else int(best['batch_size'])}, "
        f"protect_margin={best['protect_margin']:g}, protect_lambda={best['protect_lambda']:g}, "
        f"protect_sample_size={best['protect_sample_size']:g}, casebase_lambda={best['casebase_lambda']:g}",
        f"- Best at step {best['best_step']:g}/{steps}: val combined={best['best_val_combined']:.6f} "
        f"(correction={best['best_val_correction']:.6f}, protect={best['best_val_protect']:.6f}, "
        f"casebase_correction={best['best_val_casebase_correction']:.6f})",
        f"- Final step ({steps}): val combined={best['final_val_combined']:.6f}, "
        f"train combined={best['final_train_combined']:.6f}, overfit gap={best['overfit_gap']:+.6f}",
        "",
        "## Full results",
        "",
        "Sorted by best val combined loss (ascending).",
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
        help="YAML file holding the fixed (non-swept) checkpoint/dataset paths "
        "(see tuning/contest/finetune.yaml). --checkpoint/--dataset override it.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--casebase-dataset", default=None,
        help="build_casebase_finetune_dataset.py output (see tuning/contest/finetune.yaml "
        "for the default). If resolved to a path, casebase_correction_loss is wired into every combo "
        "and --casebase-lambdas is swept; if not, casebase_lambda is forced to [0.0] for the whole grid.",
    )
    parser.add_argument("--eval-split", default=None, choices=["val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None, help=f"Fixed step budget per combo. Default: {DEFAULT_STEPS}.")
    parser.add_argument("--chunk-size", type=int, default=None, help=f"Fixed. Default: {DEFAULT_CHUNK_SIZE}.")
    parser.add_argument("--log-every", type=int, default=None, help=f"Eval frequency. Default: {DEFAULT_LOG_EVERY}.")
    parser.add_argument("--seed", type=int, default=None, help=f"Default: {DEFAULT_SEED}.")

    parser.add_argument(
        "--lrs", default=",".join(str(x) for x in DEFAULT_LRS),
        help=f"Comma-separated lr values. Default: {DEFAULT_LRS}.",
    )
    parser.add_argument(
        "--batch-sizes", default=",".join(str(x) for x in DEFAULT_BATCH_SIZES),
        help=f"Comma-separated batch_size values; 0 means full-batch. Default: {DEFAULT_BATCH_SIZES}.",
    )
    parser.add_argument(
        "--protect-margins", default=",".join(str(x) for x in DEFAULT_PROTECT_MARGINS),
        help=f"Comma-separated protect_margin values. Default: {DEFAULT_PROTECT_MARGINS}.",
    )
    parser.add_argument(
        "--protect-lambdas", default=",".join(str(x) for x in DEFAULT_PROTECT_LAMBDAS),
        help=f"Comma-separated protect_lambda values. Default: {DEFAULT_PROTECT_LAMBDAS}.",
    )
    parser.add_argument(
        "--protect-sample-sizes", default=",".join(str(x) for x in DEFAULT_PROTECT_SAMPLE_SIZES),
        help=f"Comma-separated protect_sample_size values. Default: {DEFAULT_PROTECT_SAMPLE_SIZES}.",
    )
    parser.add_argument(
        "--casebase-lambdas", default=",".join(str(x) for x in DEFAULT_CASEBASE_LAMBDAS),
        help=f"Comma-separated casebase_lambda values. Default: {DEFAULT_CASEBASE_LAMBDAS}. "
        "Ignored (forced to [0.0]) if --casebase-dataset doesn't resolve to a path.",
    )

    parser.add_argument(
        "--max-combos", type=int, default=None,
        help="If given and smaller than the full grid, randomly subsample this "
        "many combos (seeded by --seed) instead of running every combination.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = load_config(args.config)

    checkpoint = resolve_read_path(required(args.checkpoint, config, "checkpoint", args.config))
    dataset_path = resolve_read_path(required(args.dataset, config, "dataset", args.config))
    casebase_dataset_arg = resolved(args.casebase_dataset, config, "casebase_dataset", None)
    casebase_dataset_path = resolve_read_path(casebase_dataset_arg) if casebase_dataset_arg else None
    eval_split = resolved(args.eval_split, config, "eval_split", DEFAULT_EVAL_SPLIT)
    if eval_split not in ("val", "test"):
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}.")
    device = resolved(args.device, config, "device", "cuda" if torch.cuda.is_available() else "cpu")
    steps = resolved(args.steps, config, "steps", DEFAULT_STEPS)
    chunk_size = resolved(args.chunk_size, config, "chunk_size", DEFAULT_CHUNK_SIZE)
    log_every = resolved(args.log_every, config, "log_every", DEFAULT_LOG_EVERY)
    seed = resolved(args.seed, config, "seed", DEFAULT_SEED)

    lrs = _parse_floats(args.lrs)
    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    protect_margins = _parse_floats(args.protect_margins)
    protect_lambdas = _parse_floats(args.protect_lambdas)
    protect_sample_sizes = _parse_ints(args.protect_sample_sizes)
    if casebase_dataset_path is not None:
        casebase_lambdas = _parse_floats(args.casebase_lambdas)
    else:
        print("No --casebase-dataset resolved -- casebase_lambda forced to [0.0] for the whole grid.")
        casebase_lambdas = [0.0]

    combos = [
        Combo(*c) for c in itertools.product(
            lrs, batch_sizes, protect_margins, protect_lambdas, protect_sample_sizes, casebase_lambdas,
        )
    ]
    total_grid = len(combos)

    rng = random.Random(seed)
    if args.max_combos is not None and args.max_combos < total_grid:
        combos = rng.sample(combos, args.max_combos)
        print(f"Sampling {len(combos)}/{total_grid} combos from the full grid (seed={seed}).")
    else:
        if total_grid > LARGE_GRID_WARNING_THRESHOLD:
            print(
                f"Warning: full grid is {total_grid} combos -- this may take a long time. "
                "Pass --max-combos to randomly subsample instead."
            )
        print(f"Running the full grid: {total_grid} combos.")

    print(f"Loading baseline checkpoint from {checkpoint} ...")
    model, data_dict = load_fitted_model_and_data(checkpoint, device)
    assert_shares_partial_order(model)
    X_eval = data_dict[f"X_{eval_split}"]
    y_eval = data_dict[f"y_{eval_split}"]

    print(f"Loading dataset from {dataset_path} ...")
    dataset = torch.load(dataset_path, map_location=device, weights_only=False)
    train_tensors, val_tensors, manifest = dataset["train"], dataset["val"], dataset["manifest"]
    print(
        f"train: {manifest['num_train_pairs']} pairs from {manifest['num_train_samples']} samples; "
        f"val: {manifest['num_val_pairs']} pairs from {manifest['num_val_samples']} samples"
    )
    if train_tensors["targets"].shape[0] == 0:
        raise ValueError(f"{dataset_path} has 0 training pairs -- nothing to fine-tune against.")

    casebase_edges = None
    if casebase_dataset_path is not None:
        print(f"Loading casebase dataset from {casebase_dataset_path} ...")
        casebase_dataset = torch.load(casebase_dataset_path, map_location=device, weights_only=False)
        casebase_manifest = casebase_dataset["manifest"]
        print(
            f"casebase-internal edges: {casebase_manifest['num_train_edges']} train, "
            f"{casebase_manifest['num_val_edges']} val "
            f"({casebase_manifest['num_unreachable_dropped']} dropped as structurally unreachable)"
        )
        X_casebase, y_casebase, X_default, y_default = model.casebase_and_defaults()
        casebase_edges = {
            "X_casebase": X_casebase, "y_casebase": y_casebase,
            "X_default": X_default, "y_default": y_default,
            "train_edges": casebase_dataset["train"], "val_edges": casebase_dataset["val"],
        }

    # Pre-finetune snapshot every combo resets to before it starts -- fixed
    # across the whole sweep, computed once from the freshly loaded
    # (untouched) checkpoint.
    freeze_all_except_trainable(model)
    trainable_extractor = model.casebase_edge_weights.feature_extractors[TRAINABLE_FEATURE_EXTRACTOR_INDEX]
    original_extractor_state = copy.deepcopy(trainable_extractor.state_dict())

    rows = []
    best_row = None
    sweep_start = time.perf_counter()
    pbar = tqdm(combos, desc="sweep", unit="combo")
    for combo in pbar:
        result, num_protect_samples, elapsed = _run_combo(
            model, trainable_extractor, original_extractor_state,
            train_tensors, val_tensors, X_eval, y_eval, casebase_edges,
            combo, steps, chunk_size, log_every, seed,
        )
        row = _row_from_result(combo, result, num_protect_samples, elapsed, steps)
        rows.append(row)
        if best_row is None or row["best_val_combined"] < best_row["best_val_combined"]:
            best_row = row
        pbar.set_postfix(
            best_val=f"{row['best_val_combined']:.5f}",
            overall_best=f"{best_row['best_val_combined']:.5f}",
        )
    total_elapsed = time.perf_counter() - sweep_start

    trainable_extractor.load_state_dict(original_extractor_state)  # leave the shared model as loaded

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values("best_val_combined", ascending=True).reset_index(drop=True)

    report_path = Path(resolve_write_path(args.output))
    csv_path = report_path.with_suffix(".csv")
    df_sorted.round(CSV_DECIMALS).to_csv(csv_path, index=False)
    print(f"Wrote raw sweep data to {csv_path}")

    _write_report(
        report_path, df_sorted, eval_split, checkpoint, dataset_path,
        seed, steps, chunk_size, total_grid, len(combos), total_elapsed,
    )
    print(f"Wrote sweep report to {report_path}")


if __name__ == "__main__":
    main()
