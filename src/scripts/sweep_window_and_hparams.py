"""ONE-OFF diagnostic -- answers two questions before the brainwear pipeline
is considered "final": is a 10-minute, non-overlapping window actually a good
choice, and are `tuning/brainwear/hyperparameters_brainwear.yaml`'s
CIFAR-copied values reasonable for this dataset?

Not part of the maintained pipeline -- run once, read
`outputs/<date>/brainwear_sweep_recommendation.md`, bake the winning values
into the permanent `tuning/brainwear/*.yaml` files by hand, then delete this
script and its scratch artifacts (`data/brainwear_sweep/`,
`tuning/brainwear_sweep/`).

Stage 1 (window size): for each candidate window length, re-run
preprocessing + LSTM pretraining + a single quick model fit (1 seed, val
only, no test/misclassified export) and compare validation F1. Stride is
kept equal to window length (non-overlapping) for every candidate and is
NOT swept -- overlapping windows would bias the comparison via near-
duplicate train/val leakage, so it's a fixed methodological choice, not a
free variable to search over.

Stage 2 (model hyperparameters): for the winning window size, run
`cli/run.py --tuning` (Optuna) over a generated `tune_hyperparameters_
brainwear.yaml`, optimizing validation F1.

Usage::

    python -m scripts.sweep_window_and_hparams
    python -m scripts.sweep_window_and_hparams --windows 6,10,20,40 --hparam-trials 20
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from deeparguing.output_paths import find_output, output_path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DATA_YAML = REPO_ROOT / "tuning/brainwear/data_brainwear.yaml"
BASE_MODEL_YAML = REPO_ROOT / "tuning/brainwear/model_brainwear.yaml"
BASE_HYPERPARAMS_YAML = REPO_ROOT / "tuning/brainwear/hyperparameters_brainwear.yaml"
SWEEP_DATA_DIR = REPO_ROOT / "data/brainwear_sweep"
SWEEP_CONFIG_DIR = REPO_ROOT / "tuning/brainwear_sweep"

DEFAULT_WINDOWS = [6, 10, 20, 40]  # epochs -- 3/5/10/20 minutes
DEFAULT_HPARAM_TRIALS = 15


def run_cmd(cmd: list[str]) -> str:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result.stdout + result.stderr


def parse_val_metrics(summary_text: str) -> tuple[float, float]:
    f1_match = re.search(r"Average Val F1:\s*([0-9.eE+-]+)", summary_text)
    acc_match = re.search(r"Average Val Acc:\s*([0-9.eE+-]+)", summary_text)
    if not f1_match or not acc_match:
        raise ValueError("Could not find 'Average Val F1'/'Average Val Acc' in summary.md")
    return float(f1_match.group(1)), float(acc_match.group(1))


def parse_best_trial(summary_text: str) -> tuple[float, dict[str, str]]:
    idx = summary_text.find("## BEST TRIAL")
    if idx == -1:
        raise ValueError("No '## BEST TRIAL' section found in summary.md")
    section = summary_text[idx:]
    value_match = re.search(r"-\s*Value:\s*([0-9.eE+-]+)", section)
    value = float(value_match.group(1))
    params: dict[str, str] = {}
    for line in section.splitlines()[1:]:
        line = line.strip()
        if not line.startswith("- ") or "Value:" in line or "Params:" in line:
            continue
        if ":" not in line:
            continue
        key, val = line[2:].split(":", 1)
        key, val = key.strip(), val.strip()
        if key and val:
            params[key] = val
    return value, params


def make_variant_configs(window_epochs: int, outdir: Path, checkpoint_name: str) -> tuple[Path, Path]:
    SWEEP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(BASE_DATA_YAML) as f:
        data_cfg = yaml.safe_load(f)
    data_cfg["data"]["params"]["target_field"]["value"] = window_epochs
    data_cfg["data"]["params"]["path"]["value"] = f"./{outdir.relative_to(REPO_ROOT).as_posix()}/brainwear_trainval.csv"
    data_cfg["data"]["params"]["test_path"]["value"] = f"./{outdir.relative_to(REPO_ROOT).as_posix()}/brainwear_test.csv"
    data_cfg["data"]["params"]["continuous_cols"]["value"][1]["value"] = window_epochs
    data_yaml_path = SWEEP_CONFIG_DIR / f"data_w{window_epochs}.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False)

    with open(BASE_MODEL_YAML) as f:
        model_cfg = yaml.safe_load(f)
    model_cfg["feature_weights"]["params"]["weights_path"]["value"] = checkpoint_name
    model_yaml_path = SWEEP_CONFIG_DIR / f"model_w{window_epochs}.yaml"
    with open(model_yaml_path, "w") as f:
        yaml.safe_dump(model_cfg, f, sort_keys=False)

    return data_yaml_path, model_yaml_path


def sweep_windows(windows: list[int], hidden_size: int, embedding_size: int) -> pd.DataFrame:
    rows = []
    for w in windows:
        print(f"\n{'=' * 80}\nWindow = {w} epochs ({w * 0.5:.1f} min)\n{'=' * 80}")
        outdir = SWEEP_DATA_DIR / f"w{w}"

        run_cmd([
            sys.executable, "-m", "scripts.preprocess_brainwear",
            "--window-epochs", str(w), "--outdir", str(outdir), "--md-log-path", "",
        ])

        checkpoint_name = f"lstm_w{w}.pt"
        run_cmd([
            sys.executable, "-m", "scripts.pretrain_lstm",
            "--trainval-path", str(outdir / "brainwear_trainval.csv"),
            "--window-epochs", str(w),
            "--hidden-size", str(hidden_size), "--embedding-size", str(embedding_size),
            "--checkpoint-name", checkpoint_name, "--md-log-path", "",
        ])

        data_yaml, model_yaml = make_variant_configs(w, outdir, checkpoint_name)
        out = run_cmd([
            sys.executable, "src/deeparguing/cli/run.py",
            "--config", str(data_yaml), str(BASE_HYPERPARAMS_YAML), str(model_yaml),
            "--seed", "0", "--log", "info", "--run_train", "-lv",
        ])
        val_f1, val_acc = parse_val_metrics(out)
        print(f"-> val_f1={val_f1:.4f} val_acc={val_acc:.4f}")
        rows.append({"window_epochs": w, "window_minutes": w * 0.5, "val_f1": val_f1, "val_acc": val_acc})

    return pd.DataFrame(rows)


TUNE_RANGES = """
epochs:
  type: value
  value: 5
lr:
  type: tune
  tune_type: float
  params:
    name: { type: value, value: lr }
    low: { type: value, value: 0.0001 }
    high: { type: value, value: 0.01 }
    log: { type: value, value: True }
gamma_dag:
  type: tune
  tune_type: categorical
  params:
    name: { type: value, value: gamma_dag }
    choices: { type: list, value: [{type: value, value: 0}, {type: value, value: 0.00001}, {type: value, value: 0.0001}, {type: value, value: 0.001}, {type: value, value: 0.01}] }
gamma_cbr:
  type: tune
  tune_type: categorical
  params:
    name: { type: value, value: gamma_cbr }
    choices: { type: list, value: [{type: value, value: 0}, {type: value, value: 0.00001}, {type: value, value: 0.0001}, {type: value, value: 0.001}, {type: value, value: 0.01}] }
gamma_batch_entropy:
  type: value
  value: 0
gamma_cp:
  type: tune
  tune_type: categorical
  params:
    name: { type: value, value: gamma_cp }
    choices: { type: list, value: [{type: value, value: 0}, {type: value, value: 0.00001}, {type: value, value: 0.0001}, {type: value, value: 0.001}, {type: value, value: 0.01}] }
gamma_sparsity:
  type: tune
  tune_type: categorical
  params:
    name: { type: value, value: gamma_sparsity }
    choices: { type: list, value: [{type: value, value: 0}, {type: value, value: 0.00001}, {type: value, value: 0.0001}, {type: value, value: 0.001}, {type: value, value: 0.01}] }
max_iters:
  type: tune
  tune_type: int
  params:
    name: { type: value, value: max_iters }
    low: { type: value, value: 5 }
    high: { type: value, value: 30 }
    step: { type: value, value: 1 }
    log: { type: value, value: False }
temperature:
  type: tune
  tune_type: float
  params:
    name: { type: value, value: temperature }
    low: { type: value, value: 10.0 }
    high: { type: value, value: 5000.0 }
    log: { type: value, value: True }
dag_alpha:
  type: value
  value: 0.001
cluster_size:
  type: tune
  tune_type: categorical
  params:
    name: { type: value, value: cluster_size }
    choices: { type: list, value: [{type: value, value: 5}, {type: value, value: 10}, {type: value, value: 15}, {type: value, value: 20}] }
batch_size:
  type: tune
  tune_type: categorical
  params:
    name: { type: value, value: batch_size }
    choices: { type: list, value: [{type: value, value: 128}, {type: value, value: 256}, {type: value, value: 512}] }
gradient_max_norm:
  type: tune
  tune_type: float
  params:
    name: { type: value, value: gradient_max_norm }
    low: { type: value, value: 0.1 }
    high: { type: value, value: 2.0 }
    step: { type: value, value: 0.1 }
    log: { type: value, value: False }
weight_decay:
  type: tune
  tune_type: float
  params:
    name: { type: value, value: weight_decay }
    low: { type: value, value: 0.00001 }
    high: { type: value, value: 0.01 }
    log: { type: value, value: True }
label_smoothing:
  type: tune
  tune_type: float
  params:
    name: { type: value, value: label_smoothing }
    low: { type: value, value: 0.0 }
    high: { type: value, value: 0.3 }
    step: { type: value, value: 0.05 }
    log: { type: value, value: False }
batch_norm:
  type: value
  value: False
rescale_edges:
  type: value
  value: false
t_norm_str:
  type: tune
  tune_type: categorical
  params:
    name: { type: value, value: t_norm_str }
    choices: { type: list, value: [{type: value, value: GodelTNorm}, {type: value, value: ProductTNorm}, {type: value, value: LukasiewiczTNorm}] }
first_layer:
  type: tune
  tune_type: categorical
  params:
    name: { type: value, value: first_layer }
    choices: { type: list, value: [{type: value, value: 16}, {type: value, value: 32}, {type: value, value: 48}] }
second_layer:
  type: tune
  tune_type: categorical
  params:
    name: { type: value, value: second_layer }
    choices: { type: list, value: [{type: value, value: 16}, {type: value, value: 24}, {type: value, value: 32}] }
third_layer:
  type: tune
  tune_type: categorical
  params:
    name: { type: value, value: third_layer }
    choices: { type: list, value: [{type: value, value: 8}, {type: value, value: 16}, {type: value, value: 24}] }
rand_weight:
  type: value
  value: 0
"""


def write_tune_hyperparameters_yaml() -> Path:
    SWEEP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = SWEEP_CONFIG_DIR / "tune_hyperparameters_brainwear.yaml"
    with open(path, "w") as f:
        f.write(TUNE_RANGES)
    return path


def tune_hyperparams(data_yaml: Path, model_yaml: Path, n_trials: int) -> tuple[float, dict[str, str]]:
    tune_yaml = write_tune_hyperparameters_yaml()
    run_cmd([
        sys.executable, "src/deeparguing/cli/run.py",
        "--config", str(data_yaml), str(tune_yaml), str(model_yaml),
        "--seed", "0", "--log", "info", "--run_train", "-lv",
        "--tuning", "--ht-obj", "f1", "-nt", str(n_trials),
    ])
    # "--- BEST TRIAL ---" is only ever written to summary.md (via
    # write_markdown_summary), never printed to stdout in that form.
    with open(find_output("summary.md")) as f:
        summary_text = f.read()
    return parse_best_trial(summary_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    parser.add_argument("--hparam-trials", type=int, default=DEFAULT_HPARAM_TRIALS)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--embedding-size", type=int, default=32)
    args = parser.parse_args()
    windows = [int(w) for w in args.windows.split(",")]

    window_results = sweep_windows(windows, args.hidden_size, args.embedding_size)
    window_results = window_results.sort_values("val_f1", ascending=False).reset_index(drop=True)
    best_window = int(window_results.iloc[0]["window_epochs"])
    print(f"\n{'=' * 80}\nBest window: {best_window} epochs ({best_window * 0.5:.1f} min)\n{window_results}\n{'=' * 80}")

    best_outdir = SWEEP_DATA_DIR / f"w{best_window}"
    best_data_yaml, best_model_yaml = make_variant_configs(
        best_window, best_outdir, f"lstm_w{best_window}.pt"
    )
    best_value, best_params = tune_hyperparams(best_data_yaml, best_model_yaml, args.hparam_trials)
    print(f"\nBest hyperparameters (val F1={best_value:.4f}):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    table_lines = ["| window_epochs | window_minutes | val_f1 | val_acc |", "|---|---|---|---|"]
    for _, row in window_results.iterrows():
        table_lines.append(
            f"| {int(row['window_epochs'])} | {row['window_minutes']:.1f} | "
            f"{row['val_f1']:.4f} | {row['val_acc']:.4f} |"
        )

    report_path = Path(output_path("brainwear_sweep_recommendation.md"))
    lines = [
        "# Brainwear window + hyperparameter sweep",
        "",
        "## Window size sweep (fixed stride=window, 1 seed, val F1)",
        "",
        *table_lines,
        "",
        f"**Winner: {best_window} epochs ({best_window * 0.5:.1f} min)**",
        "",
        f"## Hyperparameter sweep for window={best_window} (val F1={best_value:.4f}, {args.hparam_trials} trials)",
        "",
    ]
    for k, v in best_params.items():
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        "## Next steps",
        "1. Update `tuning/brainwear/data_brainwear.yaml`/`preprocess_brainwear.py` invocation to use "
        f"`--window-epochs {best_window}` as the permanent window size.",
        "2. Copy the hyperparameters above into `tuning/brainwear/hyperparameters_brainwear.yaml`.",
        "3. Re-run preprocessing + pretraining + the model stage with the winning window size.",
        "4. Delete this script, `data/brainwear_sweep/`, and `tuning/brainwear_sweep/`.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote recommendation to {report_path}")


if __name__ == "__main__":
    main()
