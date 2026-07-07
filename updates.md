# Updates

Everything committed by Greta Reitenbach since forking the repo from Adam Gould's
`Deep-Arguing` (last upstream commit: `8eedb57`, "Final experiments completed",
2026-05-07). Organized by date; commit hashes refer to `git log`.

## 2026-07-02

- **Fixed bugs causing tests to fail** (`69328d1`)
  - `GradualAACBR.forward`: `return_all_strengths` path squeezes the trailing
    dimension when `dimensions == 1`, matching the shape the non-`return_all_strengths`
    path already returned.
  - `CurriculumTrainer`: made `validation_log_strategy` optional (defaults to a new
    `CurriculumValidationLog()`), fixed several methods that were missing an
    explicit `self` parameter, extracted the epochs-vs-curriculum-length check
    into `_validate_epochs`, added a `_reset_optimizer_state` helper, and made
    the calls to `criterion`/`regulariser` tolerant of both `(model, preds, y)`
    and `(preds, y)` signatures via a `try`/`except TypeError` fallback.
  - `train/strategies.py`: same criterion/regulariser signature fallback applied
    in `CurriculumValidationLog`.

- **Edited evals to be compatible with older numpy** (`f326272`)
  - `evals.py`: replaced `np.concat` (newer numpy only) with `np.concatenate`,
    and swapped the `assert type(x) == float` checks for explicit `float(x)`
    casts so the metrics don't depend on the exact numpy scalar type returned.

- **Added misclassification download to CLI run** (`1236cae`)
  - `cli/run.py`: after test-set evaluation, isolate misclassified test samples
    and export them (with their QBAF tensors) to `graphs/misclassified_qbaf.json`
    via `model.export_to_json`. (Later gated behind a flag — see 2026-07-06.)

## 2026-07-03

- **Fixed visualizer cytoscape extension** (`802c4a3`)
  - `visualizer/index.html`: one-line fix to the Cytoscape extension setup.

- **Merge branch 'main' into counterfactual** (`60cea82`)
  - Brought the `counterfactual` branch up to date with the fixes above.

## 2026-07-06

- **Gate misclassified QBAF export behind `--misclassified_log` flag** (`3b595aa`)
  - `cli/parse_command_line.py` / `cli/run.py`: the misclassified-sample export
    added on 07-02 was running unconditionally whenever `--run_test` was set;
    added `--misclassified_log`/`-ml` so it's opt-in instead.

- **Debugging** (`26017f1`)
  - Vendored `torchviz`'s `make_dot` into `src/deeparguing/evals/compute_graph.py`,
    since `torchviz` is unmaintained and fails to import on Python >= 3.12
    (depends on the removed `distutils` module).
  - `requirements.txt`: dropped `torchviz`, pinned previously-implicit
    dependencies (`torchvision`, `graphviz`, `matplotlib`, `optuna`, `PyYAML`,
    `wandb`).
  - `evals.py` / `parse_yaml.py`: small follow-on fixes for the above.

- **Created tests for evals.py** (`894e126`)
  - Added `tests/evals_test.py` (146 lines) covering `evaluate_model`.
  - `evals.py`: renamed a variable (`predictions` -> `final_strengths`/
    `batch_predictions`) for clarity while batching predictions.
  - `compute_graph.py`: added a `pyright` ignore-comment for the vendored,
    untyped `torchviz` code.

## 2026-07-07

- **Created grae module** (`ef43211`)
  - New `src/deeparguing/counterfactuals/grae.py` module: implements
    Gradient-based Relation Attribution Explanations (G-RAEs, Definition 13
    of the Contestability paper). `compute_grae` detaches `model.A` and
    `model.new_cases_attacks_adjacency` into fresh leaf tensors, replays the
    aggregation -> influence -> `gradual_semantics` computation on those
    leaves, and reads `.grad` off both after a single `.backward()` (with an
    optional per-sample mode for the casebase-edge gradient).
    `finite_difference_grae` provides a perturbation-based cross-check of
    the analytic gradients (Algorithm 1 of the paper).
  - CLI wiring: added `--grae_log`/`-gl` to `parse_command_line.py`; `run.py`
    now computes G-RAEs for the exported misclassified samples (when
    `--misclassified_log` is also set) and saves them to
    `graphs/misclassified_grae.pt`.
  - Added `tests/grae_test.py` (25 tests) covering the new module:
    - Cross-checks `compute_grae`'s analytic gradients against
      `finite_difference_grae` (Algorithm 1) across new cases/iteration
      counts, checks the finite-difference error shrinks with `epsilon`, and
      checks the batched/aggregate code path sums the per-sample Algorithm 1
      estimates correctly.
    - Property-based tests for Propositions 4-6 (Direct/Indirect Influence,
      Irrelevance) on a hand-built EW-QBAF that bypasses `fit()` by setting
      `model.A`/`X_train`/`default_indexes` directly, confirming G-RAEs have
      the expected sign for direct, indirect (odd/even downstream-attack
      parity), and independent edges.
    - Input-validation tests (unfitted model, `target_indices` length
      mismatch, batch size > 1 for the finite-difference path) and
      result-shape/detachment checks.
