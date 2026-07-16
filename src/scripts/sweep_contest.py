"""Grid-search batch_contest hyperparameters to find the best config for
this optimization, against the same starting adjacency matrix every time.

Loads the model and misclassified samples once, then for each config
resets ``model.A`` back to the original (unperturbed) adjacency before
running ``batch_contest`` -- so every config starts from the same place and
results are directly comparable, rather than each config building on the
edits the previous config already made.

Usage::

    python src/scripts/sweep_contest.py \\
        --checkpoint outputs/checkpoints/model_checkpoint.pt \\
        --qbaf outputs/qbaf/misclassified_qbaf.json

The sweep runs in three parts:

  1. A grid over ``K_GRID`` x ``MARGIN_GRID`` -- the two hyperparameters an
     earlier hand-picked sweep (see git history) showed actually move the
     result. ``batch_size`` and ``divergence_bound`` were tested there too
     and ruled out: batch_size=20 consistently underperformed full-batch,
     and divergence_bound made zero difference at any k tested (raising it
     to 1e4 gave byte-identical results to the default of 100), so neither
     is swept here.
  2. A margin=0 "ceiling" diagnostic -- not a real candidate (see TIE_EPS:
     most of what it credits are near/exact ties that real argmax
     tie-breaking wouldn't reliably resolve to the target class), just
     context for how much headroom exists if ties didn't matter. The
     ranking below naturally scores it low despite its high raw clear
     count, since almost all of those clears get subtracted back out as
     risky ties.
  3. Two refinement checks applied specifically to the grid's best (k,
     margin) combo: alpha_init=10 (a much bigger line-search starting
     step) and batch_size=20 (mini-batching) -- re-testing hypotheses that
     didn't pan out at arbitrary k/margin combos, now anchored at the
     actual best one found.
  4. A finer margin sweep (``MARGIN_FINE_GRID``) at the grid's winning k,
     to bracket the boundary between the winning margin and margin=0 and
     see if any safe (non-tied) headroom was skipped over by the coarser
     grid.

Configs are ranked by "robust clears" (cleared minus risky/tied clears --
see TIE_EPS), tie-broken by fewer edges changed then fewer iterations --
not raw cleared count, since that's easy to inflate with ties that don't
generalize to real inference. Prints a full ranked table plus a highlighted
best-config recommendation (with the ``contest_all.py`` command to
reproduce it), and writes each config's full per-sample JSON log to
``--log-dir``, same schema as ``contest_all.py``'s.
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor

from deeparguing.counterfactuals.contest import DEFAULT_K, MARGIN, MAX_ITERS, THRESHOLD
from deeparguing.counterfactuals.batch_contest import (ALPHA_INIT,
                                                         DIVERGENCE_BOUND,
                                                         MAX_BACKTRACKS, TOL,
                                                         BatchContestResult,
                                                         batch_contest)
from deeparguing.counterfactuals.run_contest import load_model
from deeparguing.gradual_aacbr import GradualAACBR
from deeparguing.md_log import write_markdown_log
from misclassified_grae_by_class import load_all_samples

SWEEP_LOG_PATH = "outputs/logs/sweep_contest.md"

# A "cleared" sample whose margin achieved (target - rival) is under this is
# a near/exact tie, not a robust win -- real inference (evals.py) breaks
# ties via plain argmax, which picks whichever class index is lower, not
# necessarily the target. Counted separately so a config's reported success
# rate can't be inflated by ties that don't generalize.
TIE_EPS = 1e-4

# The two axes shown to actually move the result. Edit these to widen or
# narrow the search.
K_GRID = [3, 5, 10, 15, 20, 30]
MARGIN_GRID = [0.01, 0.005, 0.001]

# Finer margins to bracket the boundary between "margin=0.001" (0 risky
# ties, 19/100 in the last run) and "margin=0" (49 risky ties out of 50
# clears) -- tried only at the grid's winning k, as a fourth refinement
# phase, to see if there's a bit more safe (non-tied) headroom in between.
MARGIN_FINE_GRID = [0.0001, 0.0002, 0.0003, 0.0005, 0.0007]


@dataclass
class SweepConfig:
    label: str
    k: int = DEFAULT_K
    threshold: float = THRESHOLD
    margin: float = MARGIN
    max_iters: int = MAX_ITERS
    tol: float = TOL
    max_edits: int | None = None
    batch_size: int | None = None
    divergence_bound: float = DIVERGENCE_BOUND
    alpha_init: float = ALPHA_INIT
    max_backtracks: int = MAX_BACKTRACKS


def _grid_configs() -> list[SweepConfig]:
    return [
        SweepConfig(f"k={k}, margin={margin}", k=k, margin=margin)
        for k in K_GRID
        for margin in MARGIN_GRID
    ]


DIAGNOSTICS: list[SweepConfig] = [
    SweepConfig("margin=0 (ceiling diagnostic, not a real candidate)", margin=0.0),
]


def _safe_filename(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label).strip("_")


def _robust_score(result: BatchContestResult, risky_ties: int) -> tuple[int, int, int]:
    """Sort key (ascending): most robust clears first, then fewer edges
    changed, then fewer iterations."""
    robust_cleared = result.num_cleared - risky_ties
    return (-robust_cleared, result.num_edges_changed, result.iterations)


def _run_one(
    model: GradualAACBR,
    samples: Tensor,
    true_classes: list[int],
    original_A: Tensor,
    cfg: SweepConfig,
    log_dir: Path,
    run_stamp: str,
    args: argparse.Namespace,
) -> tuple[SweepConfig, BatchContestResult, int]:
    model.A = original_A.clone()
    print(f"\n--- {cfg.label} ---")
    result = batch_contest(
        model,
        samples,
        true_classes,
        k=cfg.k,
        threshold=cfg.threshold,
        margin=cfg.margin,
        max_iters=cfg.max_iters,
        tol=cfg.tol,
        max_edits=cfg.max_edits,
        batch_size=cfg.batch_size,
        divergence_bound=cfg.divergence_bound,
        alpha_init=cfg.alpha_init,
        max_backtracks=cfg.max_backtracks,
    )
    margin_achieved = result.final_target_strengths - result.final_rival_strengths
    risky_ties = int(((margin_achieved < TIE_EPS) & result.cleared).sum().item())
    result_line = (
        f"cleared {result.num_cleared}/{result.num_total} "
        f"({risky_ties} of those are near/exact ties < {TIE_EPS}), "
        f"{result.num_edges_changed} edges changed, "
        f"{result.iterations} iterations used"
    )
    print(result_line)
    write_markdown_log([f"--- {cfg.label} ---", result_line], SWEEP_LOG_PATH)

    log_path = log_dir / f"sweep_{run_stamp}_{_safe_filename(cfg.label)}.json"
    log = {
        "config": {
            "label": cfg.label,
            "checkpoint": args.checkpoint,
            "qbaf": args.qbaf,
            "num_samples": args.num_samples,
            "k": cfg.k,
            "threshold": cfg.threshold,
            "margin": cfg.margin,
            "max_iters": cfg.max_iters,
            "tol": cfg.tol,
            "max_edits": cfg.max_edits,
            "batch_size": cfg.batch_size,
            "divergence_bound": cfg.divergence_bound,
            "alpha_init": cfg.alpha_init,
            "max_backtracks": cfg.max_backtracks,
        },
        "summary": {
            "num_total": result.num_total,
            "num_cleared": result.num_cleared,
            "success_rate": result.num_cleared / max(1, result.num_total),
            "risky_ties_among_cleared": risky_ties,
            "num_edges_changed": result.num_edges_changed,
            "iterations": result.iterations,
        },
        "samples": [
            {
                "index": i,
                "true_class": true_classes[i],
                "cleared": bool(result.cleared[i]),
                "final_target_strength": result.final_target_strengths[i].item(),
                "final_rival_class": result.final_rival_classes[i],
                "final_rival_strength": result.final_rival_strengths[i].item(),
            }
            for i in range(result.num_total)
        ],
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    return cfg, result, risky_ties


def _print_table(
    rows: list[tuple[SweepConfig, BatchContestResult, int]], best_label: str
) -> None:
    header = (
        f"{'config':<40} {'cleared':>10} {'rate':>7} {'ties':>6} "
        f"{'edges':>7} {'iters':>7}"
    )
    table_lines = [header, "-" * len(header)]
    for cfg, result, risky_ties in rows:
        rate = result.num_cleared / max(1, result.num_total)
        marker = "  <-- BEST" if cfg.label == best_label else ""
        table_lines.append(
            f"{cfg.label:<40} {result.num_cleared:>4}/{result.num_total:<5} "
            f"{rate:>6.1%} {risky_ties:>6} "
            f"{result.num_edges_changed:>7} {result.iterations:>7}{marker}"
        )

    print("\n" + "\n".join(table_lines))
    write_markdown_log(
        ["--- FULL RANKED TABLE ---", "```\n" + "\n".join(table_lines) + "\n```"],
        SWEEP_LOG_PATH,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/model_checkpoint.pt")
    parser.add_argument("--qbaf", default="outputs/qbaf/misclassified_qbaf.json")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit the run to the first N misclassified samples (default: all).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--log-dir",
        default="outputs/contestation",
        help="Directory to write each config's JSON log to.",
    )
    args = parser.parse_args()

    with open(args.qbaf, "r", encoding="utf-8") as f:
        qbaf = json.load(f)

    model = load_model(args.checkpoint, args.device)
    assert model.A is not None, "checkpoint's model was never fit()"
    samples, true_classes = load_all_samples(qbaf, args.device, args.num_samples)
    original_A = model.A.detach().clone()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    write_markdown_log(["--- SWEEP RUN ---", f"run_stamp: {run_stamp}"], SWEEP_LOG_PATH, mode="w")

    grid = _grid_configs()
    print(
        f"Sweeping {len(grid)} grid configs ({len(K_GRID)} k values x "
        f"{len(MARGIN_GRID)} margins) + {len(DIAGNOSTICS)} diagnostic(s) "
        f"over {samples.shape[0]} misclassified samples..."
    )

    rows: list[tuple[SweepConfig, BatchContestResult, int]] = [
        _run_one(model, samples, true_classes, original_A, cfg, log_dir, run_stamp, args)
        for cfg in grid + DIAGNOSTICS
    ]

    rows.sort(key=lambda row: _robust_score(row[1], row[2]))
    best_cfg = rows[0][0]
    print(f"\nBest of the grid so far: {best_cfg.label}")
    write_markdown_log([f"Best of the grid so far: {best_cfg.label}"], SWEEP_LOG_PATH)

    # Re-test the two previously-inconclusive knobs specifically at the
    # best (k, margin) found, instead of at arbitrary combos.
    refinements: list[SweepConfig] = []
    if best_cfg.alpha_init == ALPHA_INIT:
        refinements.append(
            SweepConfig(
                f"{best_cfg.label}, alpha_init=10",
                k=best_cfg.k, margin=best_cfg.margin, alpha_init=10.0,
            )
        )
    if best_cfg.batch_size is None:
        refinements.append(
            SweepConfig(
                f"{best_cfg.label}, batch=20",
                k=best_cfg.k, margin=best_cfg.margin, batch_size=20, max_iters=100,
            )
        )

    rows += [
        _run_one(model, samples, true_classes, original_A, cfg, log_dir, run_stamp, args)
        for cfg in refinements
    ]

    # Fourth refinement: bracket the margin boundary at the grid's winning
    # k, to see if there's safe (non-tied) headroom between the winning
    # margin and 0 that MARGIN_GRID's coarser steps skipped over.
    fine_margins = [m for m in MARGIN_FINE_GRID if m not in MARGIN_GRID and m != best_cfg.margin]
    fine_refinements = [
        SweepConfig(f"k={best_cfg.k}, margin={m}", k=best_cfg.k, margin=m)
        for m in fine_margins
    ]
    rows += [
        _run_one(model, samples, true_classes, original_A, cfg, log_dir, run_stamp, args)
        for cfg in fine_refinements
    ]

    rows.sort(key=lambda row: _robust_score(row[1], row[2]))
    best_cfg, best_result, best_ties = rows[0]

    model.A = original_A  # leave the model in its original state once the sweep is done

    _print_table(rows, best_cfg.label)

    cleared_list: list[bool] = best_result.cleared.tolist()
    uncleared_rivals = Counter(
        rc for rc, cleared in zip(best_result.final_rival_classes, cleared_list)
        if not cleared
    )
    rival_note = ""
    if uncleared_rivals:
        top_rivals = ", ".join(f"class {c}: {n}" for c, n in uncleared_rivals.most_common(3))
        rival_note = f"\nTop rival classes among its uncleared samples: {top_rivals}"

    reproduce_cmd = (
        f"python src/scripts/contest_all.py --checkpoint {args.checkpoint} "
        f"--qbaf {args.qbaf} --k {best_cfg.k} --margin {best_cfg.margin} "
        f"--alpha-init {best_cfg.alpha_init}"
    )
    if best_cfg.batch_size is not None:
        reproduce_cmd += f" --batch-size {best_cfg.batch_size} --max-iters {best_cfg.max_iters}"

    best_config_block = (
        f"Best config: {best_cfg.label}\n"
        f"  cleared {best_result.num_cleared}/{best_result.num_total} "
        f"({best_ties} risky ties), {best_result.num_edges_changed} edges changed, "
        f"{best_result.iterations} iterations"
        f"{rival_note}\n"
        f"\nReproduce with:\n  {reproduce_cmd}"
    )
    print("\n" + best_config_block)
    write_markdown_log(["--- BEST CONFIG ---", best_config_block], SWEEP_LOG_PATH)


if __name__ == "__main__":
    main()
