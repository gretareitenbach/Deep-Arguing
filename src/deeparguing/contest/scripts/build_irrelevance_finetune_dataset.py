"""Serialize contest_all_irrelevance.json's touched
(sample, casebase_item, old_E, corrected_E) triples into
irrelevance_finetune_dataset.pt.

Usage::

    python -m deeparguing.contest.scripts.build_irrelevance_finetune_dataset
"""

import argparse
import json
import random
from pathlib import Path

import torch

from deeparguing.casebase_edge_weights.finetune import \
    corrected_E_to_partial_order_target
from deeparguing.contest.scripts.run_contest import load_all_samples
from deeparguing.md_log import write_markdown_log
from deeparguing.output_paths import (resolve_read_path, resolve_write_path,
                                       today_output_dir)

DEFAULT_VAL_FRAC = 0.2
DEFAULT_SEED = 0


def _gather_rows(contest_log: dict) -> list[dict]:
    """One row per (sample, casebase_item) pair, flipped samples only."""
    rows = []
    for sample in contest_log["samples"]:
        if not sample.get("flipped"):
            continue
        for edge in sample.get("touched_edges", []):
            rows.append(
                {
                    "sample_index": sample["index"],
                    "casebase_item": edge["casebase_item"],
                    "dim": edge["dim"],
                    "old_E": edge["old_E"],
                    "corrected_E": edge["corrected_E"],
                }
            )
    return rows


def _split_by_sample(
    sample_indices: list[int], val_frac: float, seed: int
) -> tuple[set[int], set[int]]:
    distinct = sorted(set(sample_indices))
    rng = random.Random(seed)
    rng.shuffle(distinct)
    n_val = round(len(distinct) * val_frac)
    val = set(distinct[:n_val])
    train = set(distinct[n_val:])
    return train, val


def _build_tensors(
    rows: list[dict], new_cases: torch.Tensor, X_train: torch.Tensor
) -> dict[str, torch.Tensor]:
    if not rows:
        return {
            "new_cases": new_cases[:0],
            "casebase_items": X_train[:0],
            "targets": torch.empty(0),
            "old_E": torch.empty(0),
            "corrected_E": torch.empty(0),
            "sample_index": torch.empty(0, dtype=torch.long),
            "casebase_item_index": torch.empty(0, dtype=torch.long),
            "dim_index": torch.empty(0, dtype=torch.long),
        }
    sample_idx = torch.tensor([r["sample_index"] for r in rows], dtype=torch.long)
    casebase_idx = torch.tensor([r["casebase_item"] for r in rows], dtype=torch.long)
    dim_idx = torch.tensor([r["dim"] for r in rows], dtype=torch.long)
    old_E = torch.tensor([r["old_E"] for r in rows], dtype=torch.float32)
    corrected_E = torch.tensor([r["corrected_E"] for r in rows], dtype=torch.float32)
    targets = corrected_E_to_partial_order_target(corrected_E)

    return {
        "new_cases": new_cases[sample_idx],
        "casebase_items": X_train[casebase_idx],
        "targets": targets,
        "old_E": old_E,
        "corrected_E": corrected_E,
        "sample_index": sample_idx,
        "casebase_item_index": casebase_idx,
        "dim_index": dim_idx,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contest-log", default="contest_all_irrelevance.json")
    parser.add_argument("--checkpoint", default="model_checkpoint.pt")
    parser.add_argument("--qbaf", default="misclassified_qbaf.json")
    parser.add_argument("--output", default="irrelevance_finetune_dataset.pt")
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    contest_log_path = resolve_read_path(args.contest_log)
    checkpoint_path = resolve_read_path(args.checkpoint)
    qbaf_path = resolve_read_path(args.qbaf)

    print(f"Loading contest log from {contest_log_path} ...")
    with open(contest_log_path, "r", encoding="utf-8") as f:
        contest_log = json.load(f)
    if "schema_version" not in contest_log.get("config", {}):
        raise ValueError(
            f"{contest_log_path} predates touched_edges logging (no "
            "config.schema_version)."
        )

    print(f"Loading qbaf from {qbaf_path} ...")
    with open(qbaf_path, "r", encoding="utf-8") as f:
        qbaf_data = json.load(f)
    new_cases, _ = load_all_samples(qbaf_data, args.device, num_samples=None)

    print(f"Loading X_train from {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    X_train = ckpt["X_train"]

    rows = _gather_rows(contest_log)
    print(f"{len(rows)} touched pairs from {sum(1 for s in contest_log['samples'] if s.get('flipped'))} flipped samples")

    sample_indices = [r["sample_index"] for r in rows]
    train_samples, val_samples = _split_by_sample(sample_indices, args.val_frac, args.seed)

    train_rows = [r for r in rows if r["sample_index"] in train_samples]
    val_rows = [r for r in rows if r["sample_index"] in val_samples]

    train_tensors = _build_tensors(train_rows, new_cases, X_train)
    val_tensors = _build_tensors(val_rows, new_cases, X_train)

    manifest = {
        "contest_log": contest_log_path,
        "checkpoint": checkpoint_path,
        "qbaf": qbaf_path,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "num_flipped_samples": sum(1 for s in contest_log["samples"] if s.get("flipped")),
        "num_touched_pairs": len(rows),
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "num_train_pairs": len(train_rows),
        "num_val_pairs": len(val_rows),
        "target_space_note": (
            "targets are corrected_E + 1 (partial_order raw-output target), "
            "NOT 1 - corrected_E -- see finetune.py's module docstring."
        ),
    }

    output_path = Path(resolve_write_path(args.output))
    torch.save(
        {"train": train_tensors, "val": val_tensors, "manifest": manifest},
        output_path,
    )
    print(f"Saved {output_path}")

    md_path = str(today_output_dir() / "irrelevance_finetune_dataset.md")
    old_E_all = torch.tensor([r["old_E"] for r in rows]) if rows else torch.empty(0)
    corrected_E_all = torch.tensor([r["corrected_E"] for r in rows]) if rows else torch.empty(0)
    lines = [
        "--- IRRELEVANCE FINE-TUNE DATASET ---",
        f"contest_log={contest_log_path}, checkpoint={checkpoint_path}, qbaf={qbaf_path}",
        f"flipped samples={manifest['num_flipped_samples']}, touched pairs={manifest['num_touched_pairs']}",
        f"train: {manifest['num_train_samples']} samples, {manifest['num_train_pairs']} pairs",
        f"val: {manifest['num_val_samples']} samples, {manifest['num_val_pairs']} pairs",
    ]
    if rows:
        lines.append(
            f"old_E: mean={old_E_all.mean():.4f} std={old_E_all.std():.4f}; "
            f"corrected_E: mean={corrected_E_all.mean():.4f} std={corrected_E_all.std():.4f}; "
            f"mean |delta|={(corrected_E_all - old_E_all).abs().mean():.4f}"
        )
    lines.append(manifest["target_space_note"])
    write_markdown_log(lines, md_path, mode="w")
    print(f"Saved data card to {md_path}")


if __name__ == "__main__":
    main()
