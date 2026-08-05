"""Turn a raw brainwear master time series (one row per 30s epoch) into
windowed, tabular train/val/test CSVs the existing ``tabular`` data loader
(``deeparguing.helper.load_tabular_data``) can read unmodified.

Each output row is one window of ``--window-epochs`` consecutive epochs,
flattened to columns ``accel_1..accel_W`` plus a trailing ``target`` column
(the majority-vote activity label). Windows never cross a real time gap in
the recording (device off/removed), and are dropped if too much of the
window was interpolated (``imputed``) rather than measured.

The split is chronological, not random: the earliest ``1 - test_frac``
fraction of windows (by time) is written to ``--outdir/brainwear_trainval.csv``,
the latest ``test_frac`` fraction to ``--outdir/brainwear_test.csv``. Point a
future ``data_brainwear.yaml`` at these via the ``tabular`` subtype's
``path``/``test_path`` params -- the loader's own 80/20 ``train_test_split``
carves val out of the trainval file, and fits its ``StandardScaler``/target
encoder on trainval only, matching every other tabular dataset in this repo.

Usage::

    python -m scripts.preprocess_brainwear
    python -m scripts.preprocess_brainwear --window-epochs 20 --test-frac 0.2
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from deeparguing.md_log import write_markdown_log
from deeparguing.output_paths import output_path, today_output_dir

LABEL_COLS = ["moderate", "sedentary", "sleep", "tasks-light", "walking"]
ACCEL_COL = "acceleration(mg)"
EPOCH_SECONDS = 30

DEFAULT_INPUT = "data/brainwear/BW_AAB_mastertimeSeries.csv"
DEFAULT_OUTDIR = "data/brainwear"
DEFAULT_WINDOW_EPOCHS = 20  # 20 * 30s = 10 minutes
DEFAULT_MAX_GAP_SECONDS = 30
DEFAULT_MAX_IMPUTED_FRAC = 0.2
DEFAULT_MIN_PURITY = 0.0
DEFAULT_TEST_FRAC = 0.2

# Categorical palette, slots 1/2 -- see dataviz skill's reference palette.
_PLOT_SERIES_TRAINVAL = "#2a78d6"
_PLOT_SERIES_TEST = "#eb6834"
_PLOT_SURFACE = "#fcfcfb"
_PLOT_GRIDLINE = "#e1e0d9"
_PLOT_AXIS = "#c3c2b7"
_PLOT_TICK_INK = "#898781"
_PLOT_PRIMARY_INK = "#0b0b0b"
_PLOT_SECONDARY_INK = "#525142"


def load_epoch_frame(path: str) -> pd.DataFrame:
    """Read the raw CSV and attach an integer/name label per epoch (argmax
    over ``LABEL_COLS``, so the ~3% of epochs with fractional/blended labels
    still get a single dominant class)."""
    df = pd.read_csv(path, parse_dates=["time"])
    label_values = df[LABEL_COLS].to_numpy()
    label_idx = label_values.argmax(axis=1)
    df["epoch_label_idx"] = label_idx
    return df


def assign_runs(df: pd.DataFrame, max_gap_seconds: int) -> pd.DataFrame:
    """Tag each epoch with a ``run_id`` that increments whenever the gap to
    the previous epoch exceeds ``max_gap_seconds`` -- so a window never
    splices across a period the device was off/removed."""
    gap = df["time"].diff().dt.total_seconds()
    new_run = gap.isna() | (gap > max_gap_seconds)
    df = df.copy()
    df["run_id"] = new_run.cumsum()
    return df


def build_windows(
    df: pd.DataFrame, window_epochs: int, stride_epochs: int
) -> pd.DataFrame:
    """Slide a ``window_epochs``-wide, ``stride_epochs``-strided window over
    each contiguous run, vectorized per run. Returns one row per window:
    ``accel_1..accel_W``, ``majority_label``, ``purity`` (majority class's
    share of the window), ``imputed_frac``, ``start_time``, ``end_time``,
    ``run_id``.
    """
    one_hot = np.eye(len(LABEL_COLS))[df["epoch_label_idx"].to_numpy()]
    accel_col_names = [f"accel_{i + 1}" for i in range(window_epochs)]

    rows: list[pd.DataFrame] = []
    for run_id, run_df in df.groupby("run_id", sort=False):
        n = len(run_df)
        if n < window_epochs:
            continue

        accel = run_df[ACCEL_COL].to_numpy()
        imputed = run_df["imputed"].to_numpy()
        times = run_df["time"].to_numpy()
        run_one_hot = one_hot[run_df.index.to_numpy()]

        accel_windows = sliding_window_view(accel, window_epochs)[::stride_epochs]
        imputed_windows = sliding_window_view(imputed, window_epochs)[::stride_epochs]

        starts = np.arange(0, n - window_epochs + 1, stride_epochs)
        cumsum = np.vstack([np.zeros(len(LABEL_COLS)), np.cumsum(run_one_hot, axis=0)])
        counts = cumsum[starts + window_epochs] - cumsum[starts]

        majority_idx = counts.argmax(axis=1)
        purity = counts.max(axis=1) / window_epochs

        run_windows = pd.DataFrame(accel_windows, columns=accel_col_names)
        run_windows["majority_label"] = np.array(LABEL_COLS)[majority_idx]
        run_windows["purity"] = purity
        run_windows["imputed_frac"] = imputed_windows.mean(axis=1)
        run_windows["has_nan"] = np.isnan(accel_windows).any(axis=1)
        run_windows["start_time"] = times[starts]
        run_windows["end_time"] = times[starts + window_epochs - 1]
        run_windows["run_id"] = run_id
        rows.append(run_windows)

    if not rows:
        raise ValueError("No windows built -- window_epochs longer than every run?")

    windows = pd.concat(rows, ignore_index=True)
    windows = windows.sort_values("start_time").reset_index(drop=True)
    windows.insert(0, "window_id", np.arange(len(windows)))
    return windows


def filter_windows(
    windows: pd.DataFrame, max_imputed_frac: float, min_purity: float
) -> tuple[pd.DataFrame, dict]:
    """Drops, in order: windows with any NaN acceleration reading (a hard
    requirement -- the model can't consume NaN, unlike ``imputed`` this isn't
    a tunable quality threshold), then over-imputed windows, then
    under-purity windows."""
    n_before = len(windows)
    nan_mask = ~windows["has_nan"]
    imputed_mask = windows["imputed_frac"] <= max_imputed_frac
    purity_mask = windows["purity"] >= min_purity
    kept = windows[nan_mask & imputed_mask & purity_mask].reset_index(drop=True)
    counts = {
        "n_before": n_before,
        "dropped_nan": int((~nan_mask).sum()),
        "dropped_imputed": int((nan_mask & ~imputed_mask).sum()),
        "dropped_purity": int((nan_mask & imputed_mask & ~purity_mask).sum()),
        "n_after": len(kept),
    }
    return kept, counts


def chronological_split(
    windows: pd.DataFrame, test_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Earliest ``1 - test_frac`` of windows (by ``start_time``) -> trainval,
    the latest ``test_frac`` -> test. No shuffling -- order is time order."""
    windows = windows.sort_values("start_time").reset_index(drop=True)
    split_idx = int(round(len(windows) * (1 - test_frac)))
    trainval = windows.iloc[:split_idx].reset_index(drop=True)
    test = windows.iloc[split_idx:].reset_index(drop=True)
    return trainval, test


def write_model_csv(windows: pd.DataFrame, window_epochs: int, path: Path) -> None:
    """Only the accel columns + target -- no metadata -- since the tabular
    loader treats every non-target column as a model input feature."""
    accel_col_names = [f"accel_{i + 1}" for i in range(window_epochs)]
    out = windows[accel_col_names].copy()
    out["target"] = windows["majority_label"]
    out.to_csv(path, index=False)


def write_metadata_csv(
    trainval: pd.DataFrame, test: pd.DataFrame, path: Path
) -> None:
    meta_cols = [
        "window_id", "run_id", "start_time", "end_time",
        "majority_label", "purity", "imputed_frac",
    ]
    trainval = trainval[meta_cols].copy()
    trainval["split"] = "trainval"
    test = test[meta_cols].copy()
    test["split"] = "test"
    pd.concat([trainval, test], ignore_index=True).to_csv(path, index=False)


def _plot_label_distribution(
    trainval: pd.DataFrame, test: pd.DataFrame, png_path: Path, window_epochs: int
) -> None:
    trainval_counts = trainval["majority_label"].value_counts().reindex(LABEL_COLS, fill_value=0)
    test_counts = test["majority_label"].value_counts().reindex(LABEL_COLS, fill_value=0)

    x = np.arange(len(LABEL_COLS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_PLOT_SURFACE)
    ax.set_facecolor(_PLOT_SURFACE)

    bars_trainval = ax.bar(
        x - width / 2, trainval_counts.values, width,
        color=_PLOT_SERIES_TRAINVAL, label="Trainval", zorder=3,
    )
    bars_test = ax.bar(
        x + width / 2, test_counts.values, width,
        color=_PLOT_SERIES_TEST, label="Test", zorder=3,
    )

    for bars in (bars_trainval, bars_test):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{int(height):,}", (bar.get_x() + bar.get_width() / 2, height),
                textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=8, color=_PLOT_SECONDARY_INK,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_COLS, color=_PLOT_PRIMARY_INK)
    ax.set_ylabel("Window count", color=_PLOT_PRIMARY_INK)
    ax.set_title(
        f"Window label distribution ({window_epochs * EPOCH_SECONDS // 60}-minute windows, majority vote)",
        color=_PLOT_PRIMARY_INK, fontsize=11,
    )
    ax.grid(True, axis="y", color=_PLOT_GRIDLINE, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_PLOT_AXIS)
    ax.spines["bottom"].set_color(_PLOT_AXIS)
    ax.tick_params(colors=_PLOT_TICK_INK)
    legend = ax.legend(frameon=False, labelcolor=_PLOT_PRIMARY_INK)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150, facecolor=_PLOT_SURFACE)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--window-epochs", type=int, default=DEFAULT_WINDOW_EPOCHS)
    parser.add_argument("--stride-epochs", type=int, default=None,
                         help="Defaults to --window-epochs (non-overlapping windows).")
    parser.add_argument("--max-gap-seconds", type=int, default=DEFAULT_MAX_GAP_SECONDS)
    parser.add_argument("--max-imputed-frac", type=float, default=DEFAULT_MAX_IMPUTED_FRAC)
    parser.add_argument("--min-purity", type=float, default=DEFAULT_MIN_PURITY,
                         help="Drop windows whose majority class share is below this. "
                              "0.0 keeps every window regardless of label ambiguity.")
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument("--md-log-path", default="", help="Empty string to skip.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    stride_epochs = args.stride_epochs or args.window_epochs

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading {args.input}")
    df = load_epoch_frame(args.input)
    df = assign_runs(df, args.max_gap_seconds)
    n_runs = df["run_id"].nunique()
    logging.info(f"{len(df):,} epochs across {n_runs} contiguous run(s)")

    windows = build_windows(df, args.window_epochs, stride_epochs)
    windows, filter_counts = filter_windows(windows, args.max_imputed_frac, args.min_purity)
    logging.info(
        f"Built {filter_counts['n_before']:,} windows, dropped "
        f"{filter_counts['dropped_nan']:,} (NaN) + "
        f"{filter_counts['dropped_imputed']:,} (imputed) + "
        f"{filter_counts['dropped_purity']:,} (purity), kept {filter_counts['n_after']:,}"
    )

    trainval, test = chronological_split(windows, args.test_frac)
    logging.info(f"Chronological split: {len(trainval):,} trainval / {len(test):,} test")

    trainval_path = outdir / "brainwear_trainval.csv"
    test_path = outdir / "brainwear_test.csv"
    metadata_path = outdir / "brainwear_window_metadata.csv"
    write_model_csv(trainval, args.window_epochs, trainval_path)
    write_model_csv(test, args.window_epochs, test_path)
    write_metadata_csv(trainval, test, metadata_path)
    logging.info(f"Wrote {trainval_path}, {test_path}, {metadata_path}")

    png_path = Path(output_path("brainwear_window_label_distribution.png"))
    _plot_label_distribution(trainval, test, png_path, args.window_epochs)
    logging.info(f"Wrote {png_path}")

    trainval_class_counts = trainval["majority_label"].value_counts().reindex(LABEL_COLS, fill_value=0)
    test_class_counts = test["majority_label"].value_counts().reindex(LABEL_COLS, fill_value=0)

    manifest = {
        "input": str(args.input),
        "window_epochs": args.window_epochs,
        "window_minutes": args.window_epochs * EPOCH_SECONDS / 60,
        "stride_epochs": stride_epochs,
        "max_gap_seconds": args.max_gap_seconds,
        "max_imputed_frac": args.max_imputed_frac,
        "min_purity": args.min_purity,
        "test_frac": args.test_frac,
        "n_epochs": int(len(df)),
        "n_runs": int(n_runs),
        "n_windows_before_filter": filter_counts["n_before"],
        "n_windows_dropped_nan": filter_counts["dropped_nan"],
        "n_windows_dropped_imputed": filter_counts["dropped_imputed"],
        "n_windows_dropped_purity": filter_counts["dropped_purity"],
        "n_windows_trainval": len(trainval),
        "n_windows_test": len(test),
        "label_cols_order": LABEL_COLS,
        "target_field_index": args.window_epochs,
        "continuous_cols_range": [0, args.window_epochs],
        "trainval_class_counts": {k: int(v) for k, v in trainval_class_counts.items()},
        "test_class_counts": {k: int(v) for k, v in test_class_counts.items()},
        "mean_purity": float(windows["purity"].mean()),
    }
    manifest_path = outdir / "window_meta.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"Wrote {manifest_path}")

    if args.md_log_path != "":
        md_path = args.md_log_path or str(today_output_dir() / "preprocess_brainwear.md")
        lines = [
            "--- BRAINWEAR PREPROCESSING ---",
            f"window_epochs={args.window_epochs} ({manifest['window_minutes']:.0f} min), "
            f"stride_epochs={stride_epochs}, max_imputed_frac={args.max_imputed_frac}, "
            f"min_purity={args.min_purity}, test_frac={args.test_frac}",
            f"epochs={len(df):,}, runs={n_runs}, "
            f"windows built={filter_counts['n_before']:,}, kept={filter_counts['n_after']:,}",
            f"trainval={len(trainval):,}, test={len(test):,}, mean purity={manifest['mean_purity']:.3f}",
            "--- TRAINVAL CLASS COUNTS ---",
            "\n".join(f"- {k}: {v:,}" for k, v in trainval_class_counts.items()),
            "--- TEST CLASS COUNTS ---",
            "\n".join(f"- {k}: {v:,}" for k, v in test_class_counts.items()),
        ]
        write_markdown_log(lines, md_path, mode="w")
        logging.info(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
