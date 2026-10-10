# Migration plan

Design date: 2026-10-09. Companion to `docs/framework_architecture.md`. This plan is the order of work. It does not perform that work.

No stage modifies a research algorithm, a circuit, a loss, a recall schedule, a corruption rule, or an archived file under `results/`. No stage runs an experiment `main`. The 40 tests that currently pass cover imports and paths. A green default pytest after a stage is necessary and is not a claim that archived metrics were reproduced.

## Frozen for every stage

These paths are read, not edited, unless a stage below names a specific non-scientific edit:

- `src/hopfield.py`, `src/qam.py`, `src/data_generation.py`, `src/metrics.py`
- `experiments/*.py` scientific bodies, including both `QAM` classes and `build_qam` / `train_qam` / the local `qam_recall`
- `results/**`
- `baseline/configs/original_experiments.json` (it is an input to the loader, not a generated file)

`src/paths.py` stays the archive locator. New runs use `benchmark/output.py`.

Commands that are out of scope for implementation work, including "just to check":

```bash
python -m experiments.noise_robustness_sweep
python -m experiments.storage_capacity
python -m experiments.circuit_depth
python -m experiments.convergence
python -m experiments.scaling
python -m experiments.statistical_significance
python -m experiments.noise_robustness_inference
python -m scripts.run_all_experiments --run
python -m scripts.run_all_experiments --only <name>
python -m scripts.plot_critical_noise_threshold
python -m scripts.cliffs_delta
python -m scripts.plot_inference_paper_figures
```

The plot commands above rewrite CSVs inside `results/`. Other plot scripts rewrite PNGs there. They are not used as a test harness.

## Stage 0 — Design record

**Scope.** Add the architecture and this plan. No Python changes.

**Files.**

- `docs/framework_architecture.md` (created)
- `docs/migration_plan.md` (created)

**Acceptance.**

- Both documents name the eight procedures, the two corruption rules, the noise-sweep reseed to 42, and `max(1, int(n * p))`.
- Both documents state that archive equivalence has not been tested.
- `git status` shows no changes under `src/`, `experiments/`, `scripts/`, `tests/`, or `results/`.

**Tests.** None. Do not run pytest as a substitute for a scientific check. The existing suite may be run if a later reader wants the import baseline; it is not an acceptance gate for this stage.

**Git checkpoint.** Message: `Add benchmark framework architecture and migration plan.` Commit only when requested. Do not add `results/`.

## Stage 1 — Characterization tests

**Scope.** Lock current helper behaviour before any call site moves. Tests call the existing functions. They do not add a `benchmark/` package and they do not change production code.

**Files.**

- `tests/test_hopfield_recall.py` (new)
- `tests/test_corruption.py` (new)
- `tests/test_metrics_definitions.py` (new)
- `tests/test_rng_coupling.py` (new)
- `tests/fixtures/` only if a snapshot array is stored as a file; prefer literals inside the test module so the fixture cannot drift from an unreviewed binary

**Acceptance.**

- Synchronous recall, sequential async recall (including energy), and one seeded random-order recall match the assertions in the architecture's testing section.
- Bernoulli and fixed-count tests pass an explicit `numpy.random.RandomState` and cover fixed-count's `max(1, int(n * p))` at `n=4`, `p=0.15` (one flip) and `p=0` (still one flip).
- `mean_std_ci` is locked to population `std` (`ddof=0`).
- The RNG test restores global NumPy state and fails if the post-`seed(42)` draw equals the draw that would have followed the trial seed without that reseed.
- No file under `results/` changes (`git status` clean there).
- No test calls a `main`.

**Tests.**

```bash
./venv/bin/python -m pytest -q tests/test_hopfield_recall.py tests/test_corruption.py tests/test_metrics_definitions.py tests/test_rng_coupling.py tests/test_imports.py tests/test_paths.py
```

`test_paths.py` creates and removes directories under `results/`. Run it only as part of this suite, and confirm the probe directories are gone afterward.

**Git checkpoint.** `Add characterization tests for recall, corruption, metrics, and RNG coupling.`

## Stage 2 — Output safety

**Scope.** A writer that cannot target `results/`, and a launcher guard so a batch run is not one forgotten flag away from the archive. Scientific modules are not retargeted. `results_subdir` keeps creating `results/<name>/`.

**Human decision before editing the launcher.** `scripts/run_all_experiments.py --only` currently starts a subprocess that overwrites `results/<name>/`. This stage makes `--run` and `--only` exit without launching unless `--overwrite-archived-results` is present. Confirm that CLI change before the edit. If it is declined, ship `benchmark/output.py` anyway and leave the launcher documented as archive-clobbering. Do not silently change it.

**Files.**

- `benchmark/__init__.py` (new)
- `benchmark/output.py` (new)
- `tests/test_output_safety.py` (new)
- `pyproject.toml` — add `benchmark` to `[tool.setuptools.packages.find] include`
- `.gitignore` — add `runs/`
- `scripts/run_all_experiments.py` — only if the launcher decision above is accepted. The edit is the flag check. The experiment list and the subprocess command stay.

**Acceptance.**

- `create_run_dir` rejects a resolved path that is `results/` or inside it, including a symlink-style trick of passing `results/noise_robustness_sweep` as the output root. Test this with `tmp_path` and a monkeypatched repository root, not by writing the real archive.
- A non-empty run directory is rejected.
- A new directory under a temporary root receives a file.
- `results_subdir("noise_robustness_sweep", "summary")` still resolves under `<repo>/results/noise_robustness_sweep/summary`. The existing path test covers the real root; the new test uses `tmp_path` for the refusal cases.
- Editable install still imports `benchmark`. After the `pyproject.toml` change, reinstall with `pip install -e ".[dev]"` from the repository root before the import test.

**Tests.**

```bash
./venv/bin/python -m pytest -q tests/test_output_safety.py tests/test_paths.py tests/test_imports.py
```

**Git checkpoint.** `Add a run directory writer that refuses the archived results tree.`

## Stage 3 — Config loader

**Scope.** Read `baseline/configs/original_experiments.json` and compare its numeric grids with the constants already defined on the experiment modules. Importing those modules is already what `tests/test_imports.py` does. Do not call `main`. Do not `eval` formula strings.

**Files.**

- `benchmark/config_loader.py` (new)
- `tests/test_config_loader.py` (new)

**Acceptance.**

- Every key under `experiments` in the JSON corresponds to one module that imports.
- These values match the module constants exactly: noise-sweep `n_bits_list`, `trials`, `corruption_rates`, `capacity_ratio`; capacity `n_bits_list`, `trials_per_m`, `recall_noise`, `train_steps`, `bootstrap_samples`; depth cases and `depths`; convergence `CONFIGS`, `TRAIN_STEPS`, `SEEDS`; scaling `n_bits_list`, `trials`, `RANDOM_SEED_BASE`; statistical-significance grids, `DEPTH`, `EPOCHS`, `LR`, `BOOTSTRAP_SAMPLES`; inference `N_BITS_LIST`, `CAPACITY_RATIOS`, `RANDOM_SEED_BASE`, `TRIALS_PER_CONFIG`.
- `device` in the JSON is the string `default.qubit`.
- The loader returns formula fields as strings. A test asserts the noise-sweep `init_reseed` text still mentions seed 42, and the fixed-count description still mentions `max(1, int(...))`.
- A mutated copy of the JSON (in `tmp_path`) with `trials: 31` for the noise sweep raises. The real JSON file is not rewritten.

**Tests.**

```bash
./venv/bin/python -m pytest -q tests/test_config_loader.py
```

**Git checkpoint.** `Load transcribed experiment grids and check them against module constants.`

## Stage 4 — Thin adapters

**Scope.** Add the eight adapter modules from the architecture. Each module imports the source symbol and returns it. No second circuit, no second loss, no unified `fit`.

Delegation map:

| Adapter module | Imports |
|---|---|
| `benchmark/adapters/hopfield_synchronous.py` | `src.hopfield.Hopfield` |
| `benchmark/adapters/hopfield_async_sequential.py` | `hopfield_hebbian_weights`, `hopfield_async_sequential_recall` |
| `benchmark/adapters/hopfield_async_random.py` | `HopfieldAsyncRandom` |
| `benchmark/adapters/qam_ry_linear.py` | `train_ry_linear_qam`, `train_ry_linear_qam_record_loss`, `qam_recall` |
| `benchmark/adapters/qam_rot_fullbatch.py` | `experiments.noise_robustness_sweep.QAM` |
| `benchmark/adapters/qam_rot_minibatch.py` | `experiments.scaling.QAM` |
| `benchmark/adapters/qam_paulix.py` | `build_qam`, `train_qam`, and local `qam_recall` aliased |
| `benchmark/adapters/hamming_nearest.py` | `nearest_neighbor_recall` |

**Files.**

- `benchmark/identity.py` (new)
- `benchmark/adapters/*.py` (new)
- `tests/test_adapters_delegate.py` (new)

**Acceptance.**

- `describe()["procedure_id"]` is the id in the architecture table, and `describe()["source"]` is the dotted path of the symbol that runs.
- `hopfield_synchronous.build(4)` returns an instance of `src.hopfield.Hopfield` with `steps == 10`.
- `hamming_nearest` has no `train` attribute.
- `qam_rot_fullbatch.build` is the experiment class. A test may construct `QAM(2)` only if it restores global RNG state, because `__init__` calls `seed(42)`. Prefer an identity test (`adapter.build.__wrapped__` or `adapter.implementation is QAM`) that does not construct, plus the stage 1 reseed test. Do not train.
- `qam_rot_minibatch` records that the scaling call passes `depth=min(6, n_bits)`, `steps=200`, `lr=0.05`, and that the class default depth of 3 is not the protocol value.
- `qam_paulix`'s recall function is the statistical-significance function, and `src.qam.qam_recall` is a different object.
- `qam_ry_linear` exposes `return_tuple` as a parameter on the train passthrough. The test inspects the signature. It does not run Adam.
- Importing adapters does not write under `results/`.

**Tests.**

```bash
./venv/bin/python -m pytest -q tests/test_adapters_delegate.py tests/test_hopfield_recall.py tests/test_imports.py
```

**Git checkpoint.** `Add thin adapters that delegate to the existing Hopfield and QAM procedures.`

## Stage 5 — Smoke protocol runners

**Scope.** `benchmark/runner.py` and `benchmark/protocols/` call adapters in the historical order for a single smoke cell. Output goes through `benchmark/output.py`. `comparable_to_archive` is false. No `--full-grid` execution in tests. No `main`.

Implement runners in this order. Stop if a runner would need a copied loss or a copied circuit; that means the adapter is wrong.

1. Inference-shaped smoke: `n_bits=8`, one capacity ratio, one noise level, `trial_id=0`, sequential Hopfield and Hamming nearest neighbour, inline Bernoulli mask, `compute_metrics`. No pool, no 216,000 tasks. Write per-trial rows to `tmp_path`.
2. Synchronous Hopfield smoke: one pattern batch, `corrupt_bernoulli` with the global stream only if the test saved and restored it; otherwise pass a `RandomState` and mark the row `rng: explicit-test`, which is not the historical noise-sweep stream.
3. Historical noise-sweep order, still without the 300-step train in the default suite: seed, pattern `choice`, Hopfield Bernoulli, then a **construction-only** check that `QAM.__init__` reseeds. The default test deletes the QAM object after init or skips init and relies on stage 1. The optional `slow` test calls `run_trial((4, 0.0, 0))` and checks return keys only.
4. RY-linear, fixed-count, and PauliX smokes stay behind `slow` because they train. Default acceptance does not run them. Their protocol modules may exist and must refuse to launch when `grid` is `full` unless `--full-grid` is passed.

**Files.**

- `benchmark/runner.py` (new)
- `benchmark/scoring.py` (new) — the dispatch table from the architecture, calling `src.metrics` helpers
- `benchmark/protocols/noise_robustness_sweep.py` (new)
- `benchmark/protocols/noise_robustness_inference.py` (new)
- `benchmark/protocols/storage_capacity.py`, `circuit_depth.py`, `convergence.py`, `scaling.py`, `statistical_significance.py` (new stubs that load config and raise `NotImplementedError` on `--full-grid`, or smoke implementations if they only call already-tested helpers)
- `benchmark/data/patterns.py`, `benchmark/data/corruption.py` (new wrappers)
- `tests/test_smoke_runner.py` (new)

**Acceptance.**

- The inference-shaped smoke writes a CSV under `tmp_path` with procedure ids `hopfield_async_sequential` and `hamming_nearest`. If the row also has `model` equal to `hopfield` and `qam` in order to mirror the archive schema, the manifest says `model=qam` means `hamming_nearest`.
- The smoke does not create `results/noise_robustness_inference/` entries beyond what `test_paths.py` already cleans up. Assert the real raw CSV's size is unchanged (50,153,034 bytes at audit time) if the test touches the repository root at all. Prefer not to stat it from a test that could be brittle; prefer `git status -- results` clean.
- `scoring.py` calls `compute_metrics` for that smoke and `mean_std_ci` only for the noise-sweep aggregate path. It does not call `parametric_ci`.
- A requested full inference grid is rejected in tests. The rejection is an exception, not a truncated run.
- `slow` tests are excluded from the default pytest addopts (`-q` has no `-m` filter today). Register the marker in `pyproject.toml` and document `pytest -q -m 'not slow'` as the checkpoint command so a future strict marker setting cannot surprise anyone. If `addopts` stays `-q`, do not mark default tests as `slow`.

**Tests.**

```bash
./venv/bin/python -m pytest -q tests/test_smoke_runner.py tests/test_output_safety.py tests/test_corruption.py tests/test_metrics_definitions.py
```

**Git checkpoint.** `Run smoke protocol cells through adapters into an isolated output directory.`

## Stage 6 — Manifests

**Scope.** Every successful `benchmark` run writes `manifest.json` before tables. Fields: protocol name, `procedure_id` per arm, config object loaded in stage 3, Python version, versions of numpy, pandas, pennylane, scipy, matplotlib via `importlib.metadata`, device string `default.qubit`, seed formula as text, RNG contract, `grid` (`smoke` or `full`), output paths, git commit if `git rev-parse HEAD` succeeds, and `comparable_to_archive: false`.

There is no code path that sets `comparable_to_archive` to true. A future human comparison can add a separate note file. The boolean in the manifest stays false until a written comparison exists, and even then the change belongs in a follow-up design, not in this stage.

**Files.**

- `benchmark/manifest.py` (new)
- `benchmark/runner.py` (call the manifest writer)
- `tests/test_manifest.py` (new)

**Acceptance.**

- A smoke run's manifest parses as JSON, names `default.qubit`, names the procedure ids, and has `comparable_to_archive` false.
- The manifest does not contain a copied metric from `results/`.
- Package versions are read from the environment, not hard-coded to the pins, so a mismatch between the pin and the venv is visible. The test can assert the keys exist.

**Tests.**

```bash
./venv/bin/python -m pytest -q tests/test_manifest.py tests/test_smoke_runner.py
```

**Git checkpoint.** `Record procedure identity, device, and package versions on each benchmark run.`

## Stage 7 — Explicit random state (optional, gated)

**Scope.** Do not start this stage with the others. Start it only after stages 1–6 are checkpointed and a human asks for a second protocol version that threads a legacy `RandomState`.

The work, when it happens:

- Add draw-sequence tests that compare the global-seed path to the threaded path for one cell of one protocol.
- Candidate mechanism: `numpy.random.RandomState(seed)`, not `numpy.random.default_rng`.
- Helpers that already take `rng` (`corrupt_bernoulli`, `corrupt_fixed_count`, `random_bipolar_patterns`, `bootstrap_mean_ci`) are the only functions that receive it.
- `QAM.__init__` in the noise sweep still executes `pnp.random.seed(42)`. Reproducing that cell means the threaded path must still perform that global reseed, or the test is expected to fail and the threaded path is rejected.
- Scaling stays on the PennyLane NumPy global stream until a sequence test includes permutations and both minibatch draws.
- Reference modules in `experiments/` are not switched over in this stage. The new path lives under `benchmark/protocols/` as a named variant, for example `rng_mode: legacy-global` (default) and `rng_mode: random-state`. Default remains `legacy-global`.

**Files.** Only after the gate: `tests/test_rng_sequence.py`, and the protocol module under test. No edits to `src/` or `experiments/`.

**Acceptance.** The sequence test matches stage 1 fixtures for the chosen cell. If it does not, the variant is not merged. Absence of this stage does not block use of the smoke runner.

**Tests.** The new sequence test plus the stage 1 RNG test.

**Git checkpoint.** `Add an opt-in RandomState path that matches the recorded global draw sequence.` Omit the checkpoint entirely if the sequence does not match.

## Cross-stage rules

- One stage, one checkpoint. Do not combine stage 4's adapters with a drive-by edit to a loss.
- `benchmark/` may import `experiments.*`. `experiments/` does not import `benchmark/`. The reference scripts stay runnable without the framework package's runner, aside from the optional launcher flag in stage 2.
- Multiprocessing pools are copied only when a full-grid runner is intentionally implemented. Smoke tests are single-process.
- Wall-clock assertions are absent. A test may check that a time key exists. It does not check the value against an archived train-time column.
- Wilcoxon sentinels stay `None` (capacity) and `1.0` (statistical significance). `scoring.py` exposes two wrappers if both protocols grow runners. It does not coerce one to the other.
- Dead helpers stay dead: `parametric_ci`, storage capacity's `confidence_interval`, scaling's `bootstrap_ci`, and the unused `ansatz_type` constant. Do not wire them up because they are nearby.

## After the last required stage

Stages 0–6 are the framework. Stage 7 is optional. At that point the repository has:

- reference experiment modules that still overwrite `results/` if someone runs them
- a runner that writes smokes under `runs/` or a chosen directory outside `results/`
- tests that lock recall, corruption, metrics, and the seed-42 coupling
- manifests that identify which procedure a row came from

A comparison of one archived cell is a separate request. The smallest cell that includes both the synchronous Hopfield arm and the Rot full-batch arm is one noise-sweep trial, not the 20-row summary. Comparing that trial with `results/noise_robustness_sweep/summary/noise_robustness_sweep.csv` is not valid: the CSV is a 30-trial aggregate and has no per-trial rows. Matching the aggregate requires all 30 seeds and still leaves train-time columns machine-dependent. Do that only under an explicit request, write it under `runs/`, and hash the archived CSV first.

## Suggested checkpoint log

| Commit | After |
|---|---|
| Architecture and migration plan | Stage 0 |
| Characterization tests | Stage 1 |
| Safe run directories | Stage 2 |
| Config loader | Stage 3 |
| Adapters | Stage 4 |
| Smoke runners | Stage 5 |
| Manifests | Stage 6 |
| Opt-in RandomState path | Stage 7, only if the sequence test passes |
