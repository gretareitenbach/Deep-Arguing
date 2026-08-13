"""Serialize a ``contest_all.py`` run's touched ``model.A`` edges
(``{source, target, dim, old_weight, new_weight}``, casebase-item-to-
casebase-item) into a training-ready ``casebase_finetune_dataset.pt``, the
casebase-internal counterpart to ``build_irrelevance_finetune_dataset.py``'s
new-case-to-casebase pairs. Consumed by ``contest.scripts.run_finetune``'s
``--casebase-dataset`` flag, which regresses ``model.A`` toward these values
via ``casebase_correction_loss`` -- see that function's docstring for why
this needs the checkpoint's ``X_train``/``y_train``/``default_indexes``
(to differentiably re-fit through), not just the touched-edge values
themselves.

Unlike ``contest_all_irrelevance.json``'s fixed filename,
``contest_all.py`` writes a timestamped ``contestation_<ts>.json`` -- pass
the exact path via ``--contest-log`` (or the ``contest_log`` config key).

Filtering (see ``contest.scripts.run_finetune``'s module docstring for the
mechanism this depends on): with ``defaults_not_attack=True`` (CIFAR10's
config), an edge whose source is a default case AND whose source/target
labels differ is forced to exactly 0 in the attacks channel, structurally,
regardless of ``feature_weights_1`` -- ``casebase_correction_loss`` could
never fit a nonzero target there. Such edges are dropped, with a warning
giving the count, rather than silently included as an unfittable residual.

Split is by *edge* (not by sample -- there's no "sample" grouping concept
for casebase-internal pairs; every touched edge is already the finest-grained
unit here).

Usage::

    python -m deeparguing.contest.scripts.build_casebase_finetune_dataset \\
        --contest-log outputs/12Aug2026/contestation_20260812T120000Z.json \\
        --checkpoint outputs/30Jul2026/model_checkpoint.pt
"""

import argparse
import json
import random
from pathlib import Path

import torch

from deeparguing.md_log import write_markdown_log
from deeparguing.output_paths import (resolve_read_path, resolve_write_path,
                                       today_output_dir)

DEFAULT_VAL_FRAC = 0.2
DEFAULT_SEED = 0


def _unreachable_mask(
    source_idx: torch.Tensor, target_idx: torch.Tensor, y_train: torch.Tensor, default_indexes: torch.Tensor,
) -> torch.Tensor:
    """True where an edge is structurally forced to 0 regardless of
    feature_weights_1: source is a default case, and source/target labels
    differ (``defaults_not_attack``'s effect on the attacks channel --
    supports aren't default-masked, so same-label default edges are fine).
    """
    is_default_source = torch.isin(source_idx, default_indexes)
    differing_labels = torch.any(y_train[source_idx] != y_train[target_idx], dim=-1)
    return is_default_source & differing_labels


def _split_by_edge(n: int, val_frac: float, seed: int) -> tuple[set[int], set[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = round(n * val_frac)
    val = set(indices[:n_val])
    train = set(indices[n_val:])
    return train, val


def _build_tensors(rows: list[dict]) -> dict[str, torch.Tensor]:
    if not rows:
        return {
            "source_idx": torch.empty(0, dtype=torch.long),
            "target_idx": torch.empty(0, dtype=torch.long),
            "dim_idx": torch.empty(0, dtype=torch.long),
            "old_weight": torch.empty(0),
            "targets": torch.empty(0),
        }
    return {
        "source_idx": torch.tensor([r["source"] for r in rows], dtype=torch.long),
        "target_idx": torch.tensor([r["target"] for r in rows], dtype=torch.long),
        "dim_idx": torch.tensor([r["dim"] for r in rows], dtype=torch.long),
        "old_weight": torch.tensor([r["old_weight"] for r in rows], dtype=torch.float32),
        "targets": torch.tensor([r["new_weight"] for r in rows], dtype=torch.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contest-log", required=True, help="Exact contest_all.py output path (timestamped, no default).")
    parser.add_argument("--checkpoint", default="model_checkpoint.pt")
    parser.add_argument("--output", default="casebase_finetune_dataset.pt")
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    contest_log_path = resolve_read_path(args.contest_log)
    checkpoint_path = resolve_read_path(args.checkpoint)

    print(f"Loading contest log from {contest_log_path} ...")
    with open(contest_log_path, "r", encoding="utf-8") as f:
        contest_log = json.load(f)
    touched_edges = contest_log.get("touched_edges", [])
    print(f"{len(touched_edges)} touched edges in log")

    print(f"Loading y_train/default_indexes from {checkpoint_path} ...")
    # Only reads saved tensors, not the full data pipeline (no dataset
    # download/access needed -- y_train/default_indexes are exactly what
    # _unreachable_mask needs, straight off the checkpoint).
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    y_train = checkpoint["y_train"]
    default_indexes = checkpoint["default_indexes"]

    if touched_edges:
        source_idx = torch.tensor([e["source"] for e in touched_edges], dtype=torch.long)
        target_idx = torch.tensor([e["target"] for e in touched_edges], dtype=torch.long)
        unreachable = _unreachable_mask(source_idx, target_idx, y_train, default_indexes)
        num_unreachable = int(unreachable.sum())
        rows = [e for e, drop in zip(touched_edges, unreachable.tolist()) if not drop]
    else:
        num_unreachable = 0
        rows = []

    if num_unreachable:
        print(
            f"WARNING: dropped {num_unreachable} touched edge(s) whose source is a default "
            "case with a differing-label target -- structurally forced to 0 regardless of "
            "feature_weights_1 (defaults_not_attack), can't be fit. See module docstring."
        )

    train_indices, val_indices = _split_by_edge(len(rows), args.val_frac, args.seed)
    train_rows = [rows[i] for i in sorted(train_indices)]
    val_rows = [rows[i] for i in sorted(val_indices)]

    train_edges = _build_tensors(train_rows)
    val_edges = _build_tensors(val_rows)

    manifest = {
        "contest_log": contest_log_path,
        "checkpoint": checkpoint_path,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "num_touched_edges_total": len(touched_edges),
        "num_unreachable_dropped": num_unreachable,
        "num_usable_edges": len(rows),
        "num_train_edges": len(train_rows),
        "num_val_edges": len(val_rows),
    }

    output_path = Path(resolve_write_path(args.output))
    torch.save({"train": train_edges, "val": val_edges, "manifest": manifest}, output_path)
    print(f"Saved {output_path}")

    md_path = str(today_output_dir() / "casebase_finetune_dataset.md")
    lines = [
        "--- CASEBASE FINE-TUNE DATASET ---",
        f"contest_log={contest_log_path}, checkpoint={checkpoint_path}",
        f"touched edges total={manifest['num_touched_edges_total']}, "
        f"dropped unreachable={manifest['num_unreachable_dropped']}, "
        f"usable={manifest['num_usable_edges']}",
        f"train: {manifest['num_train_edges']} edges",
        f"val: {manifest['num_val_edges']} edges",
    ]
    if rows:
        old_w = torch.tensor([r["old_weight"] for r in rows])
        new_w = torch.tensor([r["new_weight"] for r in rows])
        lines.append(
            f"old_weight: mean={old_w.mean():.4f} std={old_w.std():.4f}; "
            f"new_weight: mean={new_w.mean():.4f} std={new_w.std():.4f}; "
            f"mean |delta|={(new_w - old_w).abs().mean():.4f}"
        )
    write_markdown_log(lines, md_path, mode="w")
    print(f"Saved data card to {md_path}")


if __name__ == "__main__":
    main()
