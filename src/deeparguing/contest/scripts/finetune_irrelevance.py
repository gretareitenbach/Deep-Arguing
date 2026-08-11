"""Fine-tune ``LearnedPartialOrder``'s ``feature_weights_1`` extractor
against the touched-pairs dataset ``build_irrelevance_finetune_dataset.py``
produces, per ``week7_checklist.md``'s Wednesday plan. Everything except
``feature_weights_1`` (frozen ResNet, comparison function, base score,
``model.A`` itself) stays untouched this week -- see
``deeparguing.casebase_edge_weights.finetune`` for the three-term loss
(correction + lambda * preservation + protect_lambda * protect) and why the
fine-tuned parameter is scoped that narrowly. ``protect`` is a margin hinge
over a held-out ``eval_split`` sample of currently-correct predictions
(``global_optimize.py``'s ``_build_protect_set``), an output-level backstop
for pairs ``preservation`` doesn't cover -- see ``protect_loss``'s docstring.

Hyperparameters and paths come from a YAML config file (default
``tuning/contest/finetune_irrelevance.yaml``); any CLI flag overrides the
corresponding config value -- same pattern as ``contest_all.py``/
``contest_all_irrelevance.py``.

Usage::

    python -m deeparguing.contest.scripts.finetune_irrelevance
    python -m deeparguing.contest.scripts.finetune_irrelevance --lr 1e-3 --lam 0.1 --steps 200
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from deeparguing.casebase_edge_weights.finetune import (
    IrrelevanceFinetuneLosses, assert_shares_partial_order, compute_losses,
    freeze_all_except_trainable, trainable_parameters)
from deeparguing.contest.core.contest import MARGIN
from deeparguing.contest.global_optimize import _build_protect_set
from deeparguing.contest.scripts.run_contest import load_fitted_model_and_data
from deeparguing.output_paths import (resolve_read_path, resolve_write_path,
                                       today_output_dir)
from deeparguing.md_log import write_markdown_log

DEFAULT_CONFIG_PATH = "tuning/contest/finetune_irrelevance.yaml"
DEFAULT_LR = 1e-3
DEFAULT_LAM = 1.0
DEFAULT_STEPS = 200
DEFAULT_BATCH_SIZE = 128
DEFAULT_CHUNK_SIZE = 128
DEFAULT_LOG_EVERY = 10
DEFAULT_EVAL_SPLIT = "val"
DEFAULT_PROTECT_MARGIN = MARGIN
DEFAULT_PROTECT_LAMBDA = 1.0
DEFAULT_PROTECT_SAMPLE_SIZE = 200
DEFAULT_SEED = 0


def _load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        print(f"Warning: config file {config_path} not found -- using CLI/defaults only.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolved(cli_value: Any, config: dict[str, Any], key: str, fallback: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config.get(key) is not None:
        return config[key]
    return fallback


def _required(cli_value: Any, config: dict[str, Any], key: str, config_path: str) -> Any:
    value = cli_value if cli_value is not None else config.get(key)
    if value is None:
        raise ValueError(
            f"'{key}' was not given on the command line and is not set in "
            f"{config_path} -- add it there or pass --{key.replace('_', '-')}."
        )
    return value


def _sample_batch(tensors: dict[str, torch.Tensor], batch_size: int | None) -> dict[str, torch.Tensor]:
    n = tensors["targets"].shape[0]
    if batch_size is None or batch_size >= n:
        return tensors
    idx = torch.randperm(n)[:batch_size]
    return {k: v[idx] for k, v in tensors.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None, help="Preservation-penalty weight.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Train pairs per step (default: full-batch).")
    parser.add_argument("--chunk-size", type=int, default=None, help="See correction_loss's chunk_size.")
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument(
        "--eval-split", default=None, choices=["val", "test"],
        help="Held-out split (from the checkpoint's config) used to build the protect set.",
    )
    parser.add_argument("--protect-margin", type=float, default=None, help="Protect-loss hinge margin.")
    parser.add_argument("--protect-lambda", type=float, default=None, help="Protect-loss weight.")
    parser.add_argument(
        "--protect-sample-size", type=int, default=None,
        help="Max number of currently-correct eval-split samples to protect.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for protect-set sampling.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--output-checkpoint-filename", default=None)
    parser.add_argument("--log-filename", default=None)
    args = parser.parse_args()

    config = _load_config(args.config)

    checkpoint = resolve_read_path(_required(args.checkpoint, config, "checkpoint", args.config))
    dataset_path = resolve_read_path(_required(args.dataset, config, "dataset", args.config))
    lr = _resolved(args.lr, config, "lr", DEFAULT_LR)
    lam = _resolved(args.lam, config, "lam", DEFAULT_LAM)
    steps = _resolved(args.steps, config, "steps", DEFAULT_STEPS)
    batch_size = _resolved(args.batch_size, config, "batch_size", DEFAULT_BATCH_SIZE)
    chunk_size = _resolved(args.chunk_size, config, "chunk_size", DEFAULT_CHUNK_SIZE)
    log_every = _resolved(args.log_every, config, "log_every", DEFAULT_LOG_EVERY)
    eval_split = _resolved(args.eval_split, config, "eval_split", DEFAULT_EVAL_SPLIT)
    if eval_split not in ("val", "test"):
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}.")
    protect_margin = _resolved(args.protect_margin, config, "protect_margin", DEFAULT_PROTECT_MARGIN)
    protect_lambda = _resolved(args.protect_lambda, config, "protect_lambda", DEFAULT_PROTECT_LAMBDA)
    protect_sample_size = _resolved(
        args.protect_sample_size, config, "protect_sample_size", DEFAULT_PROTECT_SAMPLE_SIZE
    )
    seed = _resolved(args.seed, config, "seed", DEFAULT_SEED)
    device = _resolved(args.device, config, "device", "cuda" if torch.cuda.is_available() else "cpu")
    log_dir_str = _resolved(args.log_dir, config, "log_dir", str(today_output_dir()))
    output_checkpoint_filename = _resolved(
        args.output_checkpoint_filename, config, "output_checkpoint_filename", "finetuned_checkpoint.pt"
    )
    log_filename = _resolved(args.log_filename, config, "log_filename", "finetune_irrelevance.md")

    log_dir = Path(log_dir_str)
    log_dir.mkdir(parents=True, exist_ok=True)
    md_path = str(log_dir / log_filename)

    print(f"Loading model from {checkpoint} ...")
    model, data_dict = load_fitted_model_and_data(checkpoint, device)
    assert_shares_partial_order(model)

    X_eval = data_dict[f"X_{eval_split}"]
    y_eval = data_dict[f"y_{eval_split}"]

    torch.manual_seed(seed)
    protect_samples, protect_target_classes = _build_protect_set(
        model, X_eval, y_eval, protect_sample_size
    )
    print(
        f"Protect set: {protect_samples.shape[0]} currently-correct {eval_split}-split "
        f"samples (of up to {protect_sample_size} requested)."
    )
    if protect_samples.shape[0] == 0 and protect_lambda > 0.0:
        print(
            "WARNING: protect set is empty but protect_lambda > 0 -- protect_loss "
            "will be 0 for the whole run (nothing to protect against, not an error)."
        )

    print(f"Loading dataset from {dataset_path} ...")
    dataset = torch.load(dataset_path, map_location=device, weights_only=False)
    train_tensors, val_tensors, manifest = dataset["train"], dataset["val"], dataset["manifest"]
    print(
        f"train: {manifest['num_train_pairs']} pairs from {manifest['num_train_samples']} samples; "
        f"val: {manifest['num_val_pairs']} pairs from {manifest['num_val_samples']} samples"
    )
    if train_tensors["targets"].shape[0] == 0:
        raise ValueError(
            f"{dataset_path} has 0 training pairs -- nothing to fine-tune against. "
            "Check contest_all_irrelevance.py's flip rate / touched_edges."
        )

    # Anchor for the preservation penalty: partial_order's raw output over
    # every casebase-internal pair, at the model's CURRENT (pre-finetune)
    # weights. Captured before freeze_all_except_trainable/training changes
    # anything.
    with torch.no_grad():
        frozen_raw_po = model.casebase_edge_weights(model.X_train, model.X_train).detach().clone()

    freeze_all_except_trainable(model)
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=lr)

    write_markdown_log(
        [
            "--- IRRELEVANCE FINE-TUNE RUN ---",
            f"checkpoint={checkpoint}, dataset={dataset_path}",
            f"lr={lr}, lam={lam}, steps={steps}, batch_size={batch_size}, "
            f"chunk_size={chunk_size}, device={device}",
            f"eval_split={eval_split}, protect_margin={protect_margin}, "
            f"protect_lambda={protect_lambda}, protect_sample_size={protect_sample_size}, "
            f"seed={seed}, num_protect_samples={protect_samples.shape[0]}",
            f"train pairs={manifest['num_train_pairs']}, val pairs={manifest['num_val_pairs']}",
        ],
        md_path,
        mode="w",
    )

    def _eval(tensors: dict[str, torch.Tensor]) -> IrrelevanceFinetuneLosses | None:
        if tensors["targets"].shape[0] == 0:
            return None
        with torch.no_grad():
            return compute_losses(
                model, tensors["new_cases"], tensors["casebase_items"], tensors["targets"],
                model.X_train, frozen_raw_po, lam,
                protect_samples, protect_target_classes, protect_margin, protect_lambda,
                chunk_size,
            )

    history = []
    progress = tqdm(range(1, steps + 1), desc="Fine-tuning irrelevance channel", unit="step")
    for step in progress:
        batch = _sample_batch(train_tensors, batch_size)
        train_losses = compute_losses(
            model, batch["new_cases"], batch["casebase_items"], batch["targets"],
            model.X_train, frozen_raw_po, lam,
            protect_samples, protect_target_classes, protect_margin, protect_lambda,
            chunk_size,
        )
        optimizer.zero_grad()
        train_losses.combined.backward()
        optimizer.step()

        progress.set_postfix(
            correction=f"{train_losses.correction.item():.4f}",
            preservation=f"{train_losses.preservation.item():.4f}",
            protect=f"{train_losses.protect.item():.4f}",
            combined=f"{train_losses.combined.item():.4f}",
        )

        if step % log_every == 0 or step == steps:
            val_losses = _eval(val_tensors)
            line = (
                f"step {step}/{steps}: train correction={train_losses.correction.item():.6f} "
                f"preservation={train_losses.preservation.item():.6f} "
                f"protect={train_losses.protect.item():.6f} "
                f"combined={train_losses.combined.item():.6f}"
            )
            if val_losses is not None:
                line += (
                    f" | val correction={val_losses.correction.item():.6f} "
                    f"preservation={val_losses.preservation.item():.6f} "
                    f"protect={val_losses.protect.item():.6f} "
                    f"combined={val_losses.combined.item():.6f}"
                )
            progress.write(line)
            history.append(line)
            write_markdown_log([line], md_path, mode="a")

    output_checkpoint_path = log_dir / output_checkpoint_filename
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config_paths": torch.load(checkpoint, map_location=device, weights_only=False)["config_paths"],
            "A": model.A,
            "X_train": model.X_train,
            "y_train": model.y_train,
            "default_indexes": model.default_indexes,
        },
        output_checkpoint_path,
    )
    print(f"Saved fine-tuned checkpoint (unchanged A, updated feature_weights_1) to {output_checkpoint_path}")
    write_markdown_log([f"Saved fine-tuned checkpoint to {output_checkpoint_path}"], md_path, mode="a")


if __name__ == "__main__":
    main()
