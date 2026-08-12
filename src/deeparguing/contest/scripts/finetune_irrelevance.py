"""Fine-tune ``LearnedPartialOrder``'s ``feature_weights_1`` extractor
against the touched-pairs dataset ``build_irrelevance_finetune_dataset.py``
produces, per ``week7_checklist.md``'s Wednesday plan. During training, only
``feature_weights_1`` moves -- the frozen ResNet, comparison function, base
score, and ``model.A`` are all untouched step-to-step -- see
``deeparguing.casebase_edge_weights.finetune`` for the two-term loss
(correction + protect_lambda * protect) and why the fine-tuned parameter is
scoped that narrowly. ``protect`` is a margin hinge over a held-out
``eval_split`` sample of currently-correct predictions (``global_optimize.py``'s
``_build_protect_set``) -- see ``protect_loss``'s docstring. (A third term,
``preservation``, was removed 2026-08-12 -- see updates.md and
``deeparguing.casebase_edge_weights.finetune``'s module docstring for why.)

Before each checkpoint is saved, ``model.A`` IS recomputed (via
``model.fit()`` on the unchanged casebase, see ``GradualAACBR.casebase_and_defaults``)
from whatever ``feature_weights_1`` weights are being saved -- new-case edges
already go through the live, fine-tuned network at prediction time (they're
computed fresh per call, see ``finetune.py``'s module docstring), so leaving
``model.A`` at its pre-finetune value would mean two different versions of
the relevance function coexist in the same argumentation graph.
2026-08-12's evaluation (see updates.md) found recomputing ``A`` this way is
accuracy-neutral on the full CIFAR10 test set relative to leaving it frozen,
while removing that inconsistency -- see
``deeparguing.contest.scripts.diff_finetune_edge_sparsity`` for how much the
recompute actually changed the graph's topology (not just edge weights).

Hyperparameters and paths come from a YAML config file (default
``tuning/contest/finetune_irrelevance.yaml``); any CLI flag overrides the
corresponding config value -- same pattern as ``contest_all.py``/
``contest_all_irrelevance.py``.

Checkpoint selection: val ``combined`` loss is tracked at every eval
(``log_every`` steps); ``output_checkpoint_filename`` (default
``finetuned_checkpoint.pt`` -- what downstream pipeline stages read) always
holds the lowest-val-combined-loss snapshot of ``feature_weights_1`` (with
``A`` recomputed from those weights), not whatever the last step happened to
land on, since 2026-08-11's 200-step run plateaued/overfit on val well
before the final step (see updates.md). The actual final-step weights (also
with ``A`` recomputed from them) are saved separately under
``final_checkpoint_filename`` (default ``finetuned_checkpoint_final.pt``)
for comparison. If no val eval ever ran (empty val split), the best
checkpoint falls back to the final one, with a warning.

Usage::

    python -m deeparguing.contest.scripts.finetune_irrelevance
    python -m deeparguing.contest.scripts.finetune_irrelevance --lr 1e-3 --protect-lambda 1 --steps 200
"""

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
import yaml

from deeparguing.casebase_edge_weights.finetune import (
    TRAINABLE_FEATURE_EXTRACTOR_INDEX, assert_shares_partial_order,
    freeze_all_except_trainable, run_finetune)
from deeparguing.contest.core.contest import MARGIN
from deeparguing.contest.global_optimize import _build_protect_set
from deeparguing.contest.scripts.run_contest import load_fitted_model_and_data
from deeparguing.output_paths import (resolve_read_path, resolve_write_path,
                                       today_output_dir)
from deeparguing.md_log import write_markdown_log

DEFAULT_CONFIG_PATH = "tuning/contest/finetune_irrelevance.yaml"
DEFAULT_LR = 1e-3
DEFAULT_STEPS = 200
DEFAULT_BATCH_SIZE = 128
DEFAULT_CHUNK_SIZE = 128
DEFAULT_LOG_EVERY = 10
DEFAULT_EVAL_SPLIT = "val"
DEFAULT_PROTECT_MARGIN = MARGIN
DEFAULT_PROTECT_LAMBDA = 1.0
DEFAULT_PROTECT_SAMPLE_SIZE = 200
DEFAULT_SEED = 0
DEFAULT_FINAL_CHECKPOINT_FILENAME = "finetuned_checkpoint_final.pt"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--lr", type=float, default=None)
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
    parser.add_argument(
        "--output-checkpoint-filename", default=None,
        help="Where the best-val-combined-loss checkpoint is saved (default: finetuned_checkpoint.pt).",
    )
    parser.add_argument(
        "--final-checkpoint-filename", default=None,
        help="Where the final-step checkpoint is saved, for comparison against the best one "
        f"(default: {DEFAULT_FINAL_CHECKPOINT_FILENAME}).",
    )
    parser.add_argument("--log-filename", default=None)
    args = parser.parse_args()

    config = _load_config(args.config)

    checkpoint = resolve_read_path(_required(args.checkpoint, config, "checkpoint", args.config))
    dataset_path = resolve_read_path(_required(args.dataset, config, "dataset", args.config))
    lr = _resolved(args.lr, config, "lr", DEFAULT_LR)
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
    final_checkpoint_filename = _resolved(
        args.final_checkpoint_filename, config, "final_checkpoint_filename", DEFAULT_FINAL_CHECKPOINT_FILENAME
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

    freeze_all_except_trainable(model)

    write_markdown_log(
        [
            "--- IRRELEVANCE FINE-TUNE RUN ---",
            f"checkpoint={checkpoint}, dataset={dataset_path}",
            f"lr={lr}, steps={steps}, batch_size={batch_size}, "
            f"chunk_size={chunk_size}, device={device}",
            f"eval_split={eval_split}, protect_margin={protect_margin}, "
            f"protect_lambda={protect_lambda}, protect_sample_size={protect_sample_size}, "
            f"seed={seed}, num_protect_samples={protect_samples.shape[0]}",
            f"train pairs={manifest['num_train_pairs']}, val pairs={manifest['num_val_pairs']}",
        ],
        md_path,
        mode="w",
    )

    def _log_eval(step: int, train_losses, val_losses) -> None:
        line = (
            f"step {step}/{steps}: train correction={train_losses.correction.item():.6f} "
            f"protect={train_losses.protect.item():.6f} "
            f"combined={train_losses.combined.item():.6f}"
        )
        if val_losses is not None:
            line += (
                f" | val correction={val_losses.correction.item():.6f} "
                f"protect={val_losses.protect.item():.6f} "
                f"combined={val_losses.combined.item():.6f}"
            )
        write_markdown_log([line], md_path, mode="a")

    result = run_finetune(
        model, train_tensors, val_tensors,
        lr, steps, batch_size, chunk_size,
        protect_samples, protect_target_classes, protect_margin, protect_lambda,
        log_every=log_every, on_eval=_log_eval,
    )

    trainable_extractor = model.casebase_edge_weights.feature_extractors[TRAINABLE_FEATURE_EXTRACTOR_INDEX]
    config_paths = torch.load(checkpoint, map_location=device, weights_only=False)["config_paths"]

    def _recompute_A() -> None:
        """Re-fit model.A from the casebase using model.casebase_edge_weights's
        CURRENT (live) feature_weights_1 -- see this module's docstring for why."""
        X_casebase, y_casebase, X_default, y_default = model.casebase_and_defaults()
        model.fit(X_casebase, y_casebase, X_default, y_default)

    def _save(state_dict: dict[str, Any], path: Path) -> None:
        torch.save(
            {
                "state_dict": state_dict,
                "config_paths": config_paths,
                "A": model.A,
                "X_train": model.X_train,
                "y_train": model.y_train,
                "default_indexes": model.default_indexes,
            },
            path,
        )

    # model's live weights are already the final-step ones -- run_finetune trains in place.
    _recompute_A()
    final_state_dict = copy.deepcopy(model.state_dict())
    final_checkpoint_path = log_dir / final_checkpoint_filename
    _save(final_state_dict, final_checkpoint_path)
    print(f"Saved final-step checkpoint (A recomputed from final weights) to {final_checkpoint_path}")

    output_checkpoint_path = log_dir / output_checkpoint_filename
    if result.best_extractor_state is not None:
        trainable_extractor.load_state_dict(result.best_extractor_state)
        _recompute_A()
        best_state_dict = model.state_dict()
        selection_line = (
            f"Best checkpoint: step {result.best_step}/{steps} "
            f"(val combined={result.best_val_losses.combined.item():.6f}), "
            f"selected over the final step's val combined loss. A recomputed from these weights."
        )
    else:
        best_state_dict = final_state_dict
        selection_line = (
            "WARNING: no val eval ever ran (empty val split) -- best checkpoint falls back "
            "to the final-step weights (A already recomputed above)."
        )
    _save(best_state_dict, output_checkpoint_path)
    print(selection_line)
    print(f"Saved best-val checkpoint (used by downstream pipeline stages) to {output_checkpoint_path}")
    write_markdown_log(
        [
            selection_line,
            f"Saved best-val checkpoint to {output_checkpoint_path}",
            f"Saved final-step checkpoint to {final_checkpoint_path}",
        ],
        md_path,
        mode="a",
    )


if __name__ == "__main__":
    main()
