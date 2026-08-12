"""Diff two checkpoints' casebase-internal adjacency matrices (``model.A``)
to see whether fine-tuning changed the argumentation graph's TOPOLOGY (which
edges exist / are blocked as non-minimal), not just their weights.

Motivation (2026-08-12, see updates.md): ``finetune_irrelevance.py`` now
recomputes ``A`` from the fine-tuned ``feature_weights_1`` before saving
(rather than leaving ``A`` frozen at its pre-finetune value), for
interpretability -- new-case edges already go through the live, fine-tuned
network (they're computed fresh per prediction, see
``deeparguing.casebase_edge_weights.finetune``'s module docstring), so a
frozen ``A`` meant two different versions of the relevance function
coexisted in the same argumentation graph. But recomputing ``A`` doesn't
just reweight edges -- ``fit()`` also redetermines minimality/blocking and
symmetric attacks, both of which depend on comparing edge weights against
each other combinatorially -- so it's possible for edges to appear or
disappear entirely, not just shift in magnitude. This script quantifies
that: how many edges were gained/lost, how many flipped sign among edges
live in both, and how big the magnitude shift is where nothing structural
changed.

Both checkpoints already have ``A`` saved, so this is a pure tensor diff --
no forward pass, no dataset (CIFAR10 or otherwise) needed, and it runs fast
regardless of casebase size.

Hyperparameters and paths come from a YAML config file (default
``tuning/contest/diff_finetune_edge_sparsity.yaml``); any CLI flag overrides
the corresponding config value -- same pattern as ``finetune_irrelevance.py``.

Usage::

    python -m deeparguing.contest.scripts.diff_finetune_edge_sparsity
    python -m deeparguing.contest.scripts.diff_finetune_edge_sparsity \\
        --before outputs/30Jul2026/model_checkpoint.pt \\
        --after outputs/12Aug2026/finetuned_checkpoint.pt
"""

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from deeparguing.contest.scripts.finetune_irrelevance import (_load_config,
                                                                _required,
                                                                _resolved)
from deeparguing.md_log import write_markdown_log
from deeparguing.output_paths import resolve_read_path, today_output_dir

DEFAULT_CONFIG_PATH = "tuning/contest/diff_finetune_edge_sparsity.yaml"
DEFAULT_OUTPUT_FILENAME = "diff_finetune_edge_sparsity.md"


def _load_A(checkpoint_path: str) -> Tensor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint["A"]


def sparsity_diff(before: Tensor, after: Tensor) -> dict[str, Any]:
    """Compare two same-shape adjacency matrices edge-for-edge.

    Parameters
    ----------
    before, after : Tensor
        Same shape ``(n, n, d)`` (or whatever ``model.A``'s shape is) --
        must line up 1:1, i.e. the same casebase in the same order.

    Returns
    -------
    dict[str, Any]
        Counts/stats -- see the keys below for what each measures.
    """
    if before.shape != after.shape:
        raise ValueError(
            f"A shapes differ: before={tuple(before.shape)}, after={tuple(after.shape)} -- "
            "can't diff edge-for-edge (did the casebase change size or order?)."
        )

    before_nonzero = before != 0
    after_nonzero = after != 0
    gained = (~before_nonzero) & after_nonzero  # edges that appeared
    lost = before_nonzero & (~after_nonzero)  # edges that disappeared
    both_nonzero = before_nonzero & after_nonzero
    sign_flips = both_nonzero & (before.sign() != after.sign())
    abs_diff = (before - after).abs()

    n_before_nonzero = int(before_nonzero.sum())
    n_both_nonzero = int(both_nonzero.sum())

    return {
        "n_total": before.numel(),
        "n_before_nonzero": n_before_nonzero,
        "n_after_nonzero": int(after_nonzero.sum()),
        "n_gained": int(gained.sum()),
        "n_lost": int(lost.sum()),
        "n_both_nonzero": n_both_nonzero,
        "n_sign_flips": int(sign_flips.sum()),
        "frac_lost_of_before": (int(lost.sum()) / n_before_nonzero) if n_before_nonzero else 0.0,
        "frac_gained_of_before": (int(gained.sum()) / n_before_nonzero) if n_before_nonzero else 0.0,
        "mean_abs_diff_both_nonzero": (
            float(abs_diff[both_nonzero].mean()) if n_both_nonzero else 0.0
        ),
        "max_abs_diff_both_nonzero": (
            float(abs_diff[both_nonzero].max()) if n_both_nonzero else 0.0
        ),
        "mean_abs_diff_overall": float(abs_diff.mean()),
        "max_abs_diff_overall": float(abs_diff.max()),
    }


def _report(before_path: str, after_path: str, diff: dict[str, Any]) -> str:
    lines = [
        f"Before: {before_path}",
        f"After: {after_path}",
        "",
        "## Topology change",
        "",
        f"- Total possible edges: {diff['n_total']}",
        f"- Live (nonzero) edges before: {diff['n_before_nonzero']}",
        f"- Live (nonzero) edges after: {diff['n_after_nonzero']}",
        f"- Edges gained (zero -> nonzero): {diff['n_gained']} "
        f"({diff['frac_gained_of_before']:.1%} of before's live edges)",
        f"- Edges lost (nonzero -> zero): {diff['n_lost']} "
        f"({diff['frac_lost_of_before']:.1%} of before's live edges)",
        f"- Edges live in both: {diff['n_both_nonzero']}",
        f"- Sign flips among edges live in both: {diff['n_sign_flips']}",
        "",
        "## Magnitude change",
        "",
        f"- Mean |before - after|, edges live in both: {diff['mean_abs_diff_both_nonzero']:.6f}",
        f"- Max |before - after|, edges live in both: {diff['max_abs_diff_both_nonzero']:.6f}",
        f"- Mean |before - after|, all edges: {diff['mean_abs_diff_overall']:.6f}",
        f"- Max |before - after|, all edges: {diff['max_abs_diff_overall']:.6f}",
        "",
        "## Reading this",
        "",
        "If gained/lost/sign-flip counts are all 0, fine-tuning only reweighted "
        "edges that already existed -- the argumentation graph's topology is "
        "unchanged, just its magnitudes. Nonzero gained/lost/sign-flip counts "
        "mean the graph's structure itself changed: some cases now attack "
        "different cases than before (or stopped attacking each other "
        "entirely), which changes which cases would show up in an explanation, "
        "not just how strongly.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--before", default=None, help="Checkpoint holding the 'before' A (e.g. the pre-finetune baseline).")
    parser.add_argument("--after", default=None, help="Checkpoint holding the 'after' A (e.g. the fine-tuned checkpoint).")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--output-filename", default=None)
    args = parser.parse_args()

    config = _load_config(args.config)

    before_path = resolve_read_path(_required(args.before, config, "before_checkpoint", args.config))
    after_path = resolve_read_path(_required(args.after, config, "after_checkpoint", args.config))
    log_dir_str = _resolved(args.log_dir, config, "log_dir", str(today_output_dir()))
    output_filename = _resolved(args.output_filename, config, "output_filename", DEFAULT_OUTPUT_FILENAME)

    log_dir = Path(log_dir_str)
    log_dir.mkdir(parents=True, exist_ok=True)
    md_path = str(log_dir / output_filename)

    print(f"Loading A from {before_path} ...")
    before_A = _load_A(before_path)
    print(f"Loading A from {after_path} ...")
    after_A = _load_A(after_path)

    diff = sparsity_diff(before_A, after_A)
    print(
        f"gained={diff['n_gained']} lost={diff['n_lost']} sign_flips={diff['n_sign_flips']} "
        f"(of {diff['n_before_nonzero']} live edges before) | "
        f"mean_abs_diff(both nonzero)={diff['mean_abs_diff_both_nonzero']:.6f}"
    )

    report = _report(before_path, after_path, diff)
    write_markdown_log(["--- EDGE SPARSITY DIFF ---", report], md_path, mode="w")
    print(f"Wrote report to {md_path}")


if __name__ == "__main__":
    main()
