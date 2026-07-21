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

- **Updated visualizer for edge perturbation** (`6b7f8ea`)
  - `gradual_aacbr.py`: `export_to_json` gained optional `grae_casebase_edges`/
    `grae_new_case_edges`/`grae_target_indices` params, serialized under a
    `"grae"` key, so a G-RAE result can ride along in the same JSON the
    visualizer already reads instead of a separate file.
  - `cli/run.py`: reordered the `--grae_log` block to run before the JSON
    export (it previously only wrote `misclassified_grae.pt`), and now passes
    the computed G-RAEs into `export_to_json` too. The `.pt` export is
    unchanged/still written, since it's more compact for large casebases and
    carries `selected_indices`, which the JSON export doesn't.
  - `visualizer/app.js` / `index.html`: added a "G-RAE Perturbation" sidebar
    panel. Clicking an edge (while a new case is selected) selects it for
    perturbation; a Δ slider live-computes `predicted = original_strength +
    gradient * Δ` (a linear/first-order approximation) and pulses the
    differentiated default node's opacity/border accordingly.
  - Extended the panel to handle multi-target exports (`target_indices` as a
    list-of-lists instead of one index per sample, with a matching extra
    axis on `casebase_edges`/`new_case_edges`): renders a live "competition"
    table of every candidate class's predicted strength, bolds the current
    winner, and computes the exact crossing Δ (closed-form, since predicted
    strength is linear in Δ) at which another class would overtake it.
    `run.py`'s real export still only produces the single-target shape;
    multi-target is currently only exercised by hand-built demo fixtures.
  - Fixed a latent bug in the upload handler: the actual precompute/render
    work ran inside a `setTimeout` nested outside the surrounding
    `try`/`catch`, so any exception there (regardless of cause) was silently
    swallowed — no alert, and `hideLoading()` never ran, leaving the loading
    overlay stuck indefinitely. Wrapped that block (and the `initCytoscape`
    step) in their own `try`/`catch` so failures now surface an alert and
    clear the overlay.

## 2026-07-09

- **Added summary markdown for CLI** (`d4b0028`)
  - `cli/run.py`: new `write_markdown_summary` helper renders the CLI's
    per-run summary lines (validation/train/test results, plus the new
    GRAE breakdown below) to `graphs/summary.md` -- `--- X ---` lines become
    `## X` headings, everything else becomes a bullet. Hyperparameter-tuning
    runs append a `--- BEST TRIAL ---` block after the final trial's summary.
  - Added a G-RAE magnitude breakdown by edge type: buckets each sample's
    `compute_grae` output by the sign of the underlying adjacency entry
    (`model.A` for casebase edges, `model.new_cases_attacks_adjacency` for
    new-case edges) into attack (negative) vs. support (positive), and
    averages `|gradient|` within each bucket via a new `_mean_abs_grae`
    helper. Logged per-seed and averaged into the summary's new
    `--- GRAE RESULTS ---` block. Since `new_cases_attacks_adjacency` is
    always `<= 0` by construction (`RegularIrrelevance` returns a value in
    `[0, 1]`, then negated -- new cases can only attack, never support), the
    new-case support bucket is expected to come out `nan` (empty mask, no
    positive entries to average).

## 2026-07-10

- **Created contest.py and test suite** (`a8cd256`)
  - New `src/deeparguing/counterfactuals/contest.py` module: implements the
    Week 3 heuristic contestability algorithm sketched in `grae.py`'s module
    docstring. Each outer iteration computes G-RAEs for the current sample
    against `target_class` (via `grae.compute_grae`), picks the top-`k`
    entries of `model.A` by `|G-RAE|` (`select_top_k`), then runs a
    backtracking line search (`backtracking_line_search`) that shrinks the
    step size from `ALPHA_MAX` until the trial perturbation crosses
    `threshold + margin` (or backtracks are exhausted, in which case the
    smallest-alpha trial is taken as a fallback partial step). Accepted
    steps are committed directly onto `model.A` -- the casebase-internal
    adjacency, so a successful contest has a persistent, global effect on
    the model, not just the queried sample. Returns a `ContestResult`
    (`success`, `iterations`, `max_weight_delta`, `edge_trace`,
    `final_strength`).
  - The file started as pseudocode (`sample.edge_weights`,
    `model.semantics_forward`, `NotImplementedError` stubs) that didn't
    match this repo's actual API; rewired it to the real one -- gradients
    come from the already-tested `grae.compute_grae`, and trial strengths
    come from real forward passes (`model(sample)`) with `model.A`
    temporarily swapped for a trial tensor. Also fixed a latent bug in the
    original sketch where hitting `max_iters` always reported the constant
    `MAX_ITERS` as the iteration count rather than how many iterations
    actually ran.
  - Exported `ContestResult`/`contest` from `counterfactuals/__init__.py`,
    mirroring how `grae.py`'s API is already exposed there.
  - Added `tests/contest_test.py` (10 tests) reusing the same synthetic
    EW-QBAF fixture as `tests/grae_test.py`: unit tests for `select_top_k`
    and `backtracking_line_search` (finds a crossing alpha, doesn't mutate
    `model.A` itself, returns `None` when `max_backtracks=0`), and
    end-to-end tests for `contest` (commits the accepted step to `model.A`,
    touches only the traced edge indices, reports `success=False` with
    `iterations == max_iters` when the threshold is unreachable in time,
    and raises on an unfitted model / batch size > 1).

## 2026-07-13

- **Changed threshold criteria for contestation** (`960e8ba`)
  - `contest.py`: the original stopping rule (`target_strength >= threshold +
    margin`, `THRESHOLD = 0.5`) assumed a single-argument acceptance boundary,
    but `GradualAACBR.forward`'s per-class strengths are neither normalized
    nor mutually exclusive (no softmax -- see `forward`'s `strengths @ W`
    combination), so crossing 0.5 says nothing about whether `target_class`
    actually wins the model's `argmax` prediction (the criterion `evals.py`/
    `cli/run.py` actually use to call a sample "misclassified"). Measured
    against the real `outputs/misclassified_qbaf.json` checkpoint (100
    CIFAR-10 samples): 5% already had `target_class` strength `>= 0.5` while
    still losing the `argmax`, and of the 95 that didn't, 87% still had the
    real rival class beating `target_class` by ~1.5 in strength even after
    hypothetically crossing `0.5 + margin` -- the fixed threshold was
    effectively decoupled from the thing the algorithm is supposed to fix.
  - Replaced it with an argmax-relative check: `target_class`'s strength
    must beat the best *other* class's strength (`_target_and_rival`) by
    `margin`. Every forward pass already computes every class's strength at
    once, so tracking the rival costs no extra passes. If `target_class` is
    the model's only default argument (no competing class -- the
    single-topic-argument setting from the original Contestability paper,
    e.g. `tests/contest_test.py`'s synthetic fixture), there is no real
    rival: `_target_and_rival` falls back to a fixed virtual competitor at
    `threshold`, recovering the original absolute criterion for that case.
  - `EdgeTraceStep`/`ContestResult` gained `*_rival_class`/`*_rival_strength`
    fields (renamed `old_strength`/`new_strength`/`final_strength` to
    `*_target_strength` for clarity) so the trace shows whether a step's
    rival changed identity, not just whether the target moved.
  - `run_contest.py`: `--threshold` documented as the single-topic-argument
    fallback only; logging now reports the rival class/strength and the
    achieved vs. required margin per step.

- **Changed to bracket-and-bisect search** (`fe0e607`)
  - `contest.py`: plain backtracking (shrink `alpha` from `alpha_max` until
    a trial crosses the margin, return the first one that does) optimizes
    for "cheap," not "minimal" -- confirmed live on sample 80 of the
    checkpoint above, where `alpha_max=1.0` crossed on the very first trial,
    overshooting the required margin by ~2.2 and moving edge weights by
    `>3` in one step. Renamed `backtracking_line_search` to
    `bisection_line_search`: phase 1 (bracket) is unchanged geometric
    backtracking, used only to find *a* crossing alpha; phase 2 (bisect)
    then binary-searches inside `[last-failing-alpha, first-crossing-alpha]`
    for up to `MAX_BISECTIONS` trials (or until the bracket narrows below
    `BISECT_TOL`) to converge toward the minimal crossing step instead of
    accepting the first one found. Re-run on sample 80: `max_weight_delta`
    dropped from `3.292147` to `0.061085` and the margin overshoot from
    `~2.2` to `0.0012`, for the same `success=True` outcome.
  - Each bisection trial is one more `torch.no_grad()` forward pass (no
    backward pass), so the added cost is linear in `MAX_BISECTIONS` and
    skipped entirely for dead-gradient samples (phase 1 never finds a
    crossing, so phase 2 never runs -- confirmed on sample 91, a live
    example of the kind of sample `sweep_dead_gradients.py` flags: `alpha`
    shrinks through all `MAX_BACKTRACKS` trials with an all-zero effective
    direction, `max_weight_delta=0.0` for all 50 `max_iters`, since `contest`
    has no early-exit for a dead direction and just repeats the same no-op
    step until `max_iters` is exhausted).
  - `tests/contest_test.py`: renamed the `backtracking_line_search` tests to
    `bisection_line_search`, updated for the new 5-tuple return shape and
    `final_target_strength`/`final_rival_*` fields, and added a test
    confirming bisection refines below `alpha_max` when it overshoots on the
    first trial.

- **Changed bisection thresholds** (`fad4ef3`)
  - `contest.py`: re-running sample 47 (initial rival gap only `0.0167`)
    showed phase 1 crossing at `alpha_max=1.0` and then *every* one of the
    10 phase-2 bisection trials also crossing -- `alpha_hi` halved all the
    way down to `0.5^10` without a single failing trial to pull `alpha_lo`
    up, so the loop exhausted its budget (not `BISECT_TOL`) with the margin
    still overshot by 13.5x (`0.1351` against a required `0.01`). Bumped
    `MAX_BISECTIONS` 10 -> 30 and tightened `BISECT_TOL` 1e-4 -> 1e-6; each
    extra step is one cheap forward pass and halves the bracket, so the
    added resolution is exponential (`~1e-9` relative to `alpha_max=1.0`)
    for ~20 extra passes, paid only by samples that reach phase 2 (dead-
    gradient samples like 91 above never do). Re-run on sample 47: margin
    overshoot dropped from `0.1351` to `0.0001` and `max_weight_delta` from
    `0.006209` to `0.001085`.

- **Added dead-gradient bottleneck escape** (uncommitted)
  - New `src/deeparguing/counterfactuals/bottleneck.py`: handles the case
    where `_casebase_grae` comes back uniformly ~0 for a sample, instead of
    letting `contest()` spin uselessly on a zero direction until `max_iters`
    (as sample 91 did before). Reused
    `GradualAACBR.forward(..., return_all_strengths=True)`, which already
    exposes every casebase node's own converged strength per sample -- no
    changes to `gradual_aacbr.py` needed. `find_bottleneck` greedily walks
    the influence graph from the sample towards `target_class`'s default
    argument (following the single strongest edge per hop) and returns the
    first node pinned at exactly 0. Confirmed by working through the
    aggregation math (and a synthetic test that initially assumed
    otherwise, then failed): because `aggregation_func` sums densely over
    *every* source node each iteration (not just structurally-existing
    edges), a *uniformly* zero `_casebase_grae` requires `target_class`'s
    *own* final strength to be pinned at 0 too, not merely some upstream
    node -- otherwise at least the direct one-hop edges into the target
    would show nonzero gradient from whichever source nodes have nonzero
    strength. `select_bottleneck_edges` ranks the bottleneck's incoming
    edges by `|source node's own strength|` -- exactly the local partial
    derivative of its aggregation, read directly off the forward pass
    already computed (the aggregation step is linear, so no backward pass
    needed). `expanding_step_search` is `bisection_line_search`'s mirror
    image: grows `alpha` geometrically from a small `alpha_init` instead of
    shrinking from `alpha_max`, stopping once the bottleneck node's own
    strength is no longer pinned at 0 (a different, weaker condition than
    crossing the classification margin, hence a separate function).
    `find_and_escape_bottleneck` orchestrates the three and falls back to
    `grae_vector` for direction only if the local leverage is itself all
    zero (every incoming source also saturated); if that's also zero,
    returns `None` -- structurally hopeless.
  - `contest.py`: added `LIVE_GRAD_THRESHOLD` (`1e-9` -- **not** re-validated
    here against the real checkpoint's max|grad| distribution per the day's
    compute constraints; confirm the live/dead split is still cleanly
    bimodal before trusting this value on a new checkpoint/dataset). Each
    outer iteration now checks
    `grae_vector`'s magnitude and routes to `find_and_escape_bottleneck`
    instead of `select_top_k`/`bisection_line_search` when dead; both paths
    feed one shared trace-recording/commit block (unified into a common
    6-tuple: `edge_indices, alpha, new_A, new_target_strength,
    new_rival_class, new_rival_strength`), so a dead-then-escaped sample
    falls through to ordinary gradient steps on the next iteration once
    `_casebase_grae` is recomputed and live again. `contest()` imports
    `find_and_escape_bottleneck` lazily (inside the function body) since
    `bottleneck.py` imports several private helpers back from `contest.py`
    -- an eager top-level import would be circular.
  - Deleted `sweep_dead_gradients.py`: its original purpose was to pre-filter
    "dead" samples out before running `contest()`, so the expensive search
    wasn't wasted on ones it couldn't move -- but `contest()` now handles
    that itself (escaping when possible, failing after 1 iteration instead
    of `max_iters` when not), so a sample being "dead" no longer implies it
    should be skipped, and the script's own docstring advice to that effect
    was now actively wrong. No remaining callers depend on it (`run_contest.py`
    never imported it; the dependency ran the other way, the script reusing
    `run_contest.py`'s `load_model`/`load_sample`).
  - Added `tests/bottleneck_test.py` (6 tests) on a hand-built 4-node ReLU
    EW-QBAF (bypassing `fit()`, same approach as `grae_test.py`'s
    property-based fixtures): `find_bottleneck` locates the saturated node,
    `select_bottleneck_edges` prefers the higher-leverage source,
    `find_and_escape_bottleneck` picks that edge and returns `None` when
    every attacker is itself pinned at 0 (structurally hopeless), and two
    `contest()` end-to-end cases -- starts dead, escapes, then finishes via
    ordinary bisection steps; and the structurally-hopeless case reports
    `success=False` after exactly 1 iteration rather than looping to
    `max_iters`.
  - `contest.py`: re-running sample 47 (initial rival gap only `0.0167`)
    showed phase 1 crossing at `alpha_max=1.0` and then *every* one of the
    10 phase-2 bisection trials also crossing -- `alpha_hi` halved all the
    way down to `0.5^10` without a single failing trial to pull `alpha_lo`
    up, so the loop exhausted its budget (not `BISECT_TOL`) with the margin
    still overshot by 13.5x (`0.1351` against a required `0.01`). Bumped
    `MAX_BISECTIONS` 10 -> 30 and tightened `BISECT_TOL` 1e-4 -> 1e-6; each
    extra step is one cheap forward pass and halves the bracket, so the
    added resolution is exponential (`~1e-9` relative to `alpha_max=1.0`)
    for ~20 extra passes, paid only by samples that reach phase 2 (dead-
    gradient samples like 91 above never do). Re-run on sample 47: margin
    overshoot dropped from `0.1351` to `0.0001` and `max_weight_delta` from
    `0.006209` to `0.001085`.

## 2026-07-14

- **Created script to average G-RAEs by true class** (`cfba437`)
  - New `src/scripts/misclassified_grae_by_class.py`: for every misclassified
    sample exported by `--misclassified_log`, computes each sample's
    per-edge casebase G-RAE via `compute_grae(..., target_indices=true_classes,
    per_sample=True)` (differentiating each sample's own true-class strength,
    not the model's wrong prediction), then averages those per-sample
    `(n, n, d)` gradients within each true class (`average_grae_by_true_class`)
    into one `(num_classes, n, n, d)` tensor -- answering e.g. "what's the
    average gradient of edge 56 -> 72 for samples whose true class is 7."
    Classes with zero misclassified samples are left as `NaN` rather than
    silently `0`. Saved to `outputs/misclassified_grae_by_class.pt`.

- **Added visualizer support for the averaged edges** (`e4f0e69`)
  - `misclassified_grae_by_class.py`: also writes a `--viz-output` JSON --
    the original QBAF export (same `adjacency_matrix`/`X_train`/etc. the
    visualizer already reads) plus a new `grae_by_class` field, with
    `NaN`-filled classes zeroed out first since JSON has no `NaN` literal.
  - `visualizer/app.js` / `index.html`: new "Avg G-RAE by True Class"
    sidebar panel, shown automatically when the uploaded JSON has
    `grae_by_class`. Toggling it on and picking a class overrides edge
    width/color -- normally driven by the raw adjacency weight -- with that
    class's averaged gradient, normalized by the class's own max
    `|gradient|` so thickness spans the full visible range per class (not
    globally comparable across classes). The edge-click tooltip also
    surfaces the raw averaged value for the selected class.

- **Added top-10-edges-per-class printout** (`f3a1e24`)
  - `misclassified_grae_by_class.py`: new `print_top_edges_by_class`, gated
    behind `--top-k` (default 10, `0` disables). Ranks each class's
    `(n, n, d)` grid by `|avg G-RAE|` and prints the winning
    `(source, target, dim)` triples.

- **Handled the all-zero ("dead") class case** (`dd98af3`)
  - `misclassified_grae_by_class.py`: a class whose averaged gradient grid
    is exactly zero (e.g. class 0 and 6 on the real CIFAR-10 checkpoint --
    every misclassified sample of that class has a uniformly-zero G-RAE,
    the same saturated-node condition `contest.py`'s `LIVE_GRAD_THRESHOLD`
    routing was built for) previously printed a misleading top-10 list of
    arbitrarily-tied zeros (`topk`'s tie-break order over identical values
    isn't meaningful). Now prints `"Class N: all gradients are zero
    (dead)..."` instead.

- **Added conflict-aware multi-sample contestation** (`b49e24b`)
  - Motivation: running `contest()` sequentially over many misclassified
    samples against the same shared `model.A` means edits accumulate -- one
    class's fix can be partially undone by the next class's fix landing on
    the same edge with the opposite sign. The real `grae_by_class` data
    showed this concretely: edge `55 -> 20` has opposite-sign average
    gradient for class 8 (`+0.32`) vs. classes 1/7 (`-0.61`/`-0.51`), so
    naive sequential contestation oscillates on it.
  - `contest.py`: added `select_top_k_conflict_aware`, an alternative to
    `select_top_k` that discounts (via a `conflict_lambda`-weighted penalty)
    edges that other classes' averaged G-RAE wants moved in the opposite
    direction, instead of ranking purely by the current sample's own
    `|G-RAE|`. `contest()` gained optional `grae_by_class`/`conflict_lambda`
    params that route through it; both default to the original
    conflict-unaware behavior (`grae_by_class=None`), so no existing
    caller/test needed updating.
  - New `src/scripts/contest_all.py`: runs `contest()` over every
    misclassified sample in sequence against one shared, mutating `model.A`
    (mirroring how it'd be run by hand), with `--conflict-lambda`
    controlling the new selection strategy. `--compare-baseline` runs a
    plain (`conflict_lambda=0`) pass on a separately-loaded model first for
    a side-by-side comparison. Tracks, across the whole run, which edges
    got touched by how many distinct samples and how many times a later
    sample reversed an earlier sample's sign on the same edge ("edge
    reversals") -- a direct measure of the conflict this was built to
    reduce.

- **Switched CIFAR-10 model to `QuadraticEnergySemantics`** (`ffb9be5`)
  - Diagnosis: running `contest_all.py`'s baseline pass on the real
    checkpoint showed target-class strengths exploding across samples (22
    -> 66 -> 432 -> ... -> 386796), even without conflict-aware selection.
    Root cause: the checkpoint's `ReluSemantics.influence_func`
    (`relu(relu(base_scores) + aggregations)`) has no upper bound, and
    `forward_till_convergence` iterates it `max_iters` times per forward
    pass -- structurally a power iteration against a nonnegative gain
    matrix (`model.A`'s support entries) that amplifies geometrically once
    support edges accumulate. Since every `contest()` call commits new
    support edges into the same shared `model.A`, each sample's fix primes
    the graph to amplify further for the next one.
  - `tuning/cifar10/resnet/model_cifar10_image.yaml`: `semantics.class_name`
    changed from `ReluSemantics` to `QuadraticEnergySemantics`
    (`damping=1.0`, `conservativeness=1.0`, untuned defaults). QE's
    `influence_func` squashes every update into `base_scores +
    h*(1-base_scores)` with `h = x^2/(1+x^2)` bounded in `[0, 1)`, so
    strengths stay within `[0, 1]` regardless of `model.A`'s magnitude.
    Valid without further changes since `base_score`'s `LearnedBaseScore`
    activation is already `sigmoid` (bounded `(0, 1)`), matching QE's
    assumption. Requires retraining from scratch -- the base-score/
    edge-weight networks were optimized under `ReluSemantics` specifically
    -- which invalidates the existing `outputs/model_checkpoint.pt`/
    `misclassified_qbaf.json` and everything computed from them so far.

## 2026-07-16

- **Finished CLI for batch contesting** (`f3445c5`)
  - Renamed `counterfactuals/joint_contest.py` -> `batch_contest.py` (and
    `JointContestResult`/`joint_contest` -> `BatchContestResult`/`batch_contest`)
    across `__init__.py`/`contest.py`/`grae.py`'s docstrings and
    `tests/joint_contest_test.py` -> `tests/batch_contest_test.py`, for naming
    consistency with the rest of the module -- no behavior change.
  - `src/scripts/contest_all.py`: switched from all-CLI-flag configuration to
    a YAML config file (default `tuning/contest/contest.yaml`), with any CLI
    flag overriding the corresponding config value for a one-off run
    (`_load_config`/`_resolved`/`_required` helpers). Added
    `alpha_init`/`max_backtracks` passthroughs to `batch_contest`, and now
    records which edges were touched (`touched_edges`, unraveled into
    `(source, target, dim)` plus before/after weight) in the run's JSON log,
    so a later global-accuracy investigation can correlate a specific edge
    with whichever samples depend on it.
  - New `src/scripts/sweep_contest.py` (since removed, see the next entry):
    grid-searched `batch_contest`'s `k`/`margin` (the two hyperparameters an
    earlier hand-picked sweep showed actually move the result), plus a
    margin=0 "ceiling" diagnostic and two refinement checks (`alpha_init=10`,
    `batch_size=20`) anchored at the grid's winner. Ranked configs by
    "robust clears" (cleared minus near/exact-tie clears -- real inference's
    plain-argmax tie-break doesn't reliably favor the target class) rather
    than raw clear count.
  - New `tuning/contest/contest.yaml`: holds the sweep's winning config --
    `k=3` (the algorithm's default), `margin=0.001` -- clearing 19/100
    misclassified samples with zero risky ties.

- **Organized outputs/** (`fa052f4`)
  - Restructured the flat `outputs/` directory into subfolders grouped by
    artifact type rather than by producing script (several scripts read/write
    the same artifact type as `--checkpoint`/`--qbaf` inputs, so this keeps
    those cross-script dependencies obvious): `checkpoints/` (`model_checkpoint.pt`,
    `contested_checkpoint.pt`, and the pretrain scripts' `.pt` files, moved in
    from the repo root), `qbaf/` (`pre_training_*.json`/`post_training_*.json`/
    `misclassified_qbaf.json`/the viz-ready G-RAE-by-class export), `grae/`
    (`misclassified_grae.pt`/`misclassified_grae_by_class.pt`), `contestation/`
    (unchanged -- `contest_all.py`/`sweep_contest.py` already wrote here), and
    `logs/` (`summary.md`). Updated every default path/docstring/help text this
    touched: `cli/run.py`, `cli/parse_command_line.py`,
    `counterfactuals/run_contest.py`, `scripts/misclassified_grae_by_class.py`,
    `scripts/sweep_contest.py`, `tuning/contest/contest.yaml`, and the
    `weights_path` entries in the CIFAR-10/MNIST/FashionMNIST tuning configs
    that pointed at the now-relocated pretrain checkpoints.
    `misclassified_grae_by_class.py` also gained `mkdir` calls for its own
    output paths, which it had previously never created itself (relying on
    `outputs/` already existing from an earlier `run.py` invocation). Four
    files with no producing script left in the repo (`global_eval_results.csv`,
    `top10_overlap.json`, `manifest.json`, `contest_samples_0-99.log` --
    traced via stale `.pyc` caches to scripts since deleted:
    `contest_all_and_evaluate.py`, `global_evaluation.py`,
    `sweep_dead_gradients.py`) were moved to `outputs/_archive/` rather than
    force-fit into the new structure.
  - New `src/deeparguing/md_log.py`: a `write_markdown_log` helper that mirrors
    a batch of lines to a markdown file (`"--- X ---"` -> `## X` heading, a
    line already containing a newline is passed through as-is for pre-rendered
    code blocks, everything else -> a bullet), so results/metrics/diagnostics
    printed to the terminal survive after the scrollback is gone. Wired into
    every substantive (non-progress-marker) print found across the CLI and
    standalone scripts: `cli/run.py`'s per-seed VALIDATION/TEST/TRAIN results
    (via a new optional `log_path` param on `evals.print_results`, which now
    also renders the confusion matrix as a fenced code block) plus running
    avg-F1 and per-seed G-RAE-magnitude lines, all appended to
    `outputs/logs/summary.md` (reset once per trial, not per run of
    `objective()`, so `--tuning`'s per-trial detail no longer needs to survive
    only in console scrollback); `simple_trainer.py`'s NaN-loss warning and
    `curriculum_trainer.py`'s class-advance events (same file);
    `sweep_contest.py`'s per-config result line, full ranked comparison table,
    and best-config/reproduce-command block ->
    `outputs/logs/sweep_contest.md`; `misclassified_grae_by_class.py`'s
    top-edges-per-class printout and `--edge` report ->
    `outputs/logs/misclassified_grae_by_class.md`; `verify_resnet.py`'s run
    metadata and both tests' metrics/pass-fail ->
    `outputs/logs/verify_resnet.md`; `tune_pretrain_cnn.py`'s new-best-model
    and per-run-accuracy lines -> `outputs/logs/tune_pretrain_cnn.md`;
    `generate_defeasible_data.py`'s class-distribution printout ->
    `outputs/logs/generate_defeasible_data.md`; and `weights.py`'s entire
    output (its whole purpose being terminal output meant for copy-pasting
    into a YAML config) -> `outputs/logs/weights.md`. Left alone: pure
    progress markers (trial/seed start banners, tqdm bars, "Finished
    Training"/"Training complete.", curriculum start/complete lines, "Saved
    ... to ..." confirmations) and `pretrain_cnn.py`/`pretrain_resnet.py`,
    whose final metrics only ever went to `wandb.log` and were never printed
    in the first place.

- **Refactored** (`ab3009a`)
  - Moved `src/scripts/contest_all.py` -> `src/deeparguing/counterfactuals/contest_all.py`
    (now run as `python -m deeparguing.counterfactuals.contest_all`), and moved
    its `load_all_samples` dependency (previously imported from
    `misclassified_grae_by_class.py`) into `run_contest.py`, so `contest_all.py`
    no longer reaches outside the `counterfactuals` package for it.
  - Deleted `src/scripts/misclassified_grae_by_class.py` and
    `src/scripts/sweep_contest.py` (both leftover now that `batch_contest` and
    the YAML config exist) and the empty `src/scripts/mean_std.py` stub.
    `sweep_contest.py`'s already-found winning config lives on in
    `tuning/contest/contest.yaml`; nothing else in the repo depended on either
    deleted script.

## 2026-07-17

- **Minor cleanup** (`e26104e`)
  - Cherry-picked from `TheAdamG/Deep-Arguing@b17822f`. Removed stale/dead
    files predating the counterfactuals work: `data/defeasible/defeasible.csv`,
    the `examples/` notebook and toy-example scripts, `optimisations.txt`,
    `tmp/pytorch_cnn.py`, `src/scripts/generate_defeasible_data.py`, and
    `test_seed_reproducibility.py` -- none had a producing script or active
    caller left in the repo.

- **Sync setup.py install_requires with requirements.txt** (`7692b99`)
  - `setup.py`: added `itables`, `graphviz`, `PyYAML`, `ucimlrepo`, `wandb`
    and replaced `torchviz` with `torchvision` in `install_requires`, matching
    `requirements.txt` -- which had picked up `torchvision` during the
    `multi_contest` merge and dropped `torchviz` back on 2026-07-06 when its
    `make_dot` was vendored in, while `setup.py` had drifted out of sync with
    both changes.

## 2026-07-20

- **Updated logging** (`45f7cc2`)
  - `cli/run.py`: `write_markdown_summary`'s `--- RUN LOG ---` header now also
    records the run's start timestamp and the configured semantics class name
    (read off `model_config["semantics"]["class_name"]`, `"N/A"` if no
    semantics config is present), so a `summary.md` from a past run can be
    identified without cross-referencing which config file produced it.

- **Added global contestation tests** (`7886d75`)
  - New `src/deeparguing/evals/global_contest_eval.py`:
    `compute_baseline_metrics`/`evaluate_contested_model` wrap `evaluate_model`
    to report global (full-split) accuracy/precision/recall/F1 for a
    *contested* `model.A` relative to a fixed baseline, without ever calling
    `model.fit()` (which would rebuild `model.A` from the casebase edge-weight
    functions and silently discard whatever edits produced the contested
    adjacency) -- deliberately agnostic to whether that adjacency came from
    `contest()`, `batch_contest`, or a hand-crafted edit, since it only ever
    swaps a same-shaped tensor onto an already-fitted model.
  - `evals.py`: `evaluate_model` gained a `refit: bool = True` param -- when
    `False`, skips `fit()` entirely (raising if `model.A` is still `None`) and
    evaluates the model exactly as currently configured, with
    `X_casebase`/`y_casebase`/`X_default`/`y_default` now optional and ignored
    in that case. Needed so `global_contest_eval.py` can evaluate an
    already-fitted model with its `A` swapped, without `evaluate_model`
    re-fitting over that swap.
  - New `src/deeparguing/counterfactuals/run_global_contest_eval.py`:
    standalone driver that rebuilds the baseline model + held-out split from
    `outputs/checkpoints/model_checkpoint.pt`'s own config
    (`load_model_and_split`, same rebuild approach as `run_contest.load_model`),
    evaluates it, swaps in `outputs/contestation/contested_checkpoint.pt`'s
    `A`, evaluates again, and logs the delta.
  - Added `tests/global_contest_eval_test.py`, extending the shared small-QBAF
    fixture with two default arguments (one per class) and two more "new case"
    nodes acting as a 2-sample test set, so `compute_baseline_metrics`/
    `evaluate_contested_model` have a non-trivial argmax to score. Also
    updated `evals/__init__.py`'s exports for the new module.

- **Updated eval logging** (`f0695d2`)
  - `run_global_contest_eval.py`: results are now also appended as a markdown
    table to `outputs/logs/global_contest_eval.md` by default (mirroring
    `md_log`'s convention used elsewhere, e.g. `summary.md`) -- baseline vs.
    contested vs. delta for all four metrics (`_metrics_table`) plus both
    runs' confusion matrices as fenced code blocks (`_confusion_matrix_block`).
    New `--log-path` flag, empty string to skip.

## 2026-07-21

- **Refactored for simplicity** (`7277355`)
  - Consolidated the small synthetic EW-QBAF fixture that `grae_test.py`,
    `contest_test.py`, `batch_contest_test.py`, and
    `gradual-aacbr_test.py::test_semantics` had each hand-copied (edge
    weights, base scores, the three casebase-edge-weight/base-score/
    irrelevance closures, and `_make_fitted_model`) into a new shared
    `tests/qbaf_fixtures.py`, imported by all four instead of redefined --
    removes ~150 duplicated lines with no behavior change (all 140 tests
    still pass).
  - `run_contest.py`: extracted the checkpoint-rebuild logic (`torch.load` ->
    `parse_model_config` -> restore `state_dict`/`A`/`X_train`/`y_train`/
    `default_indexes` -> `.eval()`) that `load_model` and
    `run_global_contest_eval.load_model_and_split` had each duplicated into a
    new shared `load_fitted_model_and_data`, which also returns the config's
    `data_dict` (needed by `load_model_and_split` for its `X_<split>`/
    `y_<split>` pair, but not by plain `load_model`). Both functions are now
    thin wrappers over it.

- **Created sweep for global eval** (`c0e9a12`)
  - New `src/deeparguing/counterfactuals/sweep_global_contest_eval.py`: pure
    orchestration script, no new algorithm -- sweeps `contest_all.py`'s
    first-N-misclassified-samples truncation over `N in {0, 1, 5, 10, 25, 50,
    100}` (`--ns` to override) and writes one CSV row per N to
    `outputs/contestation/global_eval_sweep.csv`. Per N: resets `model.A` to a
    fresh clone of the baseline (so each N is contested independently, not
    cumulatively), calls `batch_contest` on the first N samples from
    `load_all_samples`, then scores the resulting adjacency via
    `compute_baseline_metrics`/`evaluate_contested_model`
    (`global_contest_eval.py`) against a baseline computed once up front.
    Columns: `samples_flipped` (`result.num_cleared`), `global_acc`/`acc_drop`
    (vs. the once-computed baseline), `mean_weight_delta`/`max_weight_delta`
    (`|new - old|` over the edges `batch_contest` touched, baseline-vs-final --
    same lookup `contest_all.py` already does for its own JSON log, reused
    here), `max_strength` (peak final target strength across that N's
    contested samples, a divergence-bound sanity check), and `edge_reversals`
    (touched edges where `old * new < 0`, i.e. attack<->support sign flips).
    Fixed seed (`--seed`, default 0) via `torch.manual_seed`, though nothing
    in the default full-batch path is actually stochastic. Hyperparameters
    (`k`/`margin`/etc.) read from `tuning/contest/contest.yaml`, same as
    `contest_all.py`.

- **Added plotting to eval sweep** (`0dbb993`)
  - `sweep_global_contest_eval.py`: CSV values rounded to 3 decimal places
    (`CSV_DECIMALS`) before writing. Also renders a line chart of `acc_drop`
    vs. `N` (`_plot_acc_drop_vs_n`), each point labeled with that N's
    `samples_flipped` count, saved alongside the CSV as a same-named `.png`.
    Styled per the dataviz skill's reference palette (single blue series,
    thin 2px line, recessive gridlines, no legend needed for one series).
