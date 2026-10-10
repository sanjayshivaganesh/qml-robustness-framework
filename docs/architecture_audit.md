# Architecture audit

Audit date: 2026-10-09. Scope is the working tree of this repository. No research algorithms were changed for this audit.

The repository is already a set of experiment scripts that compare a classical Hopfield memory with several different recall procedures. There is no single application entry point, no configuration schema, and no shared training loop. Shared code covers Hebbian Hopfield recall, one RY-linear variational circuit, pattern corruption, metrics, and result paths. Three other quantum circuits, plus a classical nearest-neighbor procedure, live inside the experiment that uses them.

Uncommitted edits relative to commit `2f179a8` (`update README.md`) are docstrings, comments, and a few print strings. Spot-checked files: `src/hopfield.py`, `src/qam.py`, `src/data_generation.py`, `src/paths.py`, and the experiment modules. `src/metrics.py` matches HEAD. The numeric experiment logic described below is the code on disk.

## Repository structure

Paths that carry the research implementation and its archived outputs:

```
pyproject.toml
requirements.txt
README.md
docs/methodology.md
src/
  hopfield.py
  qam.py
  data_generation.py
  metrics.py
  paths.py
experiments/
  noise_robustness_sweep.py
  storage_capacity.py
  circuit_depth.py
  convergence.py
  scaling.py
  statistical_significance.py
  noise_robustness_inference.py
scripts/
  run_all_experiments.py
  plot_noise_robustness_sweep.py
  plot_noise_robustness_inference.py
  plot_scaling.py
  plot_inference_paper_figures.py
  plot_critical_noise_threshold.py
  plot_bootstrap_distribution.py
  cliffs_delta.py
tests/
  test_imports.py
  test_paths.py
results/
  <experiment>/{raw,summary,plots,paper_figures,*.log}
```

`results/` is tracked (71 files). Patterns are not stored. There is no dataset directory, no saved weight matrices, and no saved circuit parameters.

`src/paths.py` sets `PROJECT_ROOT` from the location of that file. `experiment_dir` and `results_subdir` create `results/<experiment>/...` on use. Output location does not depend on the process cwd.

## Component mapping

| Responsibility | Existing implementation | Current status | Recommended action | Dependencies or risks |
|---|---|---|---|---|
| Hopfield, synchronous recall | `src/hopfield.py`: `Hopfield.train`, `Hopfield.recall`, `Hopfield.recall_timed` | Used by the noise sweep, storage capacity, and statistical significance. 10 synchronous `sign(Wx)` steps; zeros become `+1`. Hebbian `W`, zero diagonal, divided by `m`. | Reuse the class directly. | Changing the step count, the zero rule, or the call order changes archived accuracies. `recall_timed` divides wall time by `n_bits`. |
| Hopfield, sequential async recall | `hopfield_hebbian_weights`, `hopfield_async_sequential_recall` | Used only by `experiments/noise_robustness_inference.py`. Index order, 20 sweeps, energy `E = -1/2 x^T W x`. | Reuse the functions. Keep them separate from `Hopfield`. | A shared "Hopfield recall" adapter that picks one schedule would mix three different dynamics. |
| Hopfield, random-order async recall | `HopfieldAsyncRandom` | Used only by `experiments/scaling.py`, imported under the name `Hopfield`. Permutations use PennyLane's NumPy RNG. | Reuse the class for the scaling protocol only. | It shares the PennyLane/NumPy RNG stream with pattern generation. Inserting another random call changes both patterns and update order. |
| RY-linear variational QAM | `src/qam.py`: `qam_ansatz`, `build_ry_linear_circuit`, `train_ry_linear_qam`, `train_ry_linear_qam_record_loss`, `qam_recall` | Used by storage capacity, circuit depth, and convergence. RY encoding, RY + linear CNOT, Adam, Pauli-Z readout, `sign` at recall. There is no `QAM` class in this module. | Reuse these functions as the capacity/depth/convergence model. | `return_tuple=True` (capacity, convergence) and `False` (depth) are both part of the current behavior. Depth's recall is inlined and does not call `qam_recall`. |
| Rot-ansatz QAM, full-batch | `experiments/noise_robustness_sweep.py`: class `QAM` | Local to that experiment. RX and RY encoding, `Rot`, depth `min(6, n)`, 300 Adam steps, lr `0.05`. Loss is a sum of squared errors, not the per-bit mean in `src.qam`. | Keep the class in place and wrap it. Do not fold it into `src.qam`. | `__init__` calls `pnp.random.seed(42)` before `np.random.randn`. Verified in this audit: that seed call resets the global NumPy RNG. |
| Rot-ansatz QAM, minibatch | `experiments/scaling.py`: class `QAM` | Local to scaling. RY encoding, `Rot`, depth `min(6, n)`, 200 steps, minibatch 2, early stop after 15 steps with loss change below `1e-4`. | Keep it local and wrap it. | `loss` draws a new minibatch on every call, and `opt.step` calls `loss` again. Early stopping compares two different batches. |
| PauliX-encoding QAM | `experiments/statistical_significance.py`: `build_qam`, `train_qam`, `qam_recall` | Local. PauliX where the bit is `-1`, RY + linear CNOT, depth 10, 100 epochs, lr `0.02`, init `N(0, 0.1)`. | Keep it local and wrap it. | The name `qam_recall` shadows `src.qam.qam_recall` inside that module. |
| Nearest-neighbor recall | `src.qam.nearest_neighbor_recall` | Called by the inference experiment. Output rows use `model="qam"`. No circuit runs. | Reuse the function. Record the column as a classical baseline whenever new code writes that label. | Pooling this column with variational QAM rows is a scientific error. The function lives in `src/qam.py` beside the circuit code. |
| Pattern generation | `src.data_generation.random_bipolar_patterns` | Implemented. Experiment loops do not call it. They call `np.random.choice([-1, 1], ...)` or PennyLane's equivalent. | Point new runners at the existing helper only after the RNG stream is pinned. Switching call sites now would change draws. | Global RNG. Patterns are not saved, so a later run cannot reload the original memories. |
| Bernoulli corruption | `corrupt_bernoulli` | Used by the noise sweep and statistical significance. Inference and scaling inline the same mask (`rand < p`, then multiply by `-1`). | Reuse the helper for those two callers. Leave the inlined copies until a seeded generator is threaded through. | Sweep corruption after `QAM.__init__` is drawn from the RNG state that follows `seed(42)`, so QAM queries are not tied to the trial seed in the same way Hopfield queries are. |
| Fixed-count corruption | `corrupt_fixed_count` | Used by storage capacity and circuit depth at `p=0.15`. Flip count is `max(1, int(n * p))`. | Reuse as-is for those experiments. | At `p=0` the helper still flips one bit. Those experiments never pass `p=0`. |
| Bit and full-pattern accuracy | `src.metrics`: `bit_accuracy`, `full_accuracy`, `compute_metrics`, `full_and_bitwise_accuracy`, `pattern_accuracy_pair`, `hamming_error` | Used, but not by every experiment. The noise sweep and scaling compute the same quantities inline. | Reuse the helpers. Add tests that lock the current definitions before any caller is switched over. | Full accuracy is 1 only when every bit matches. Bit accuracy is the fraction of matches. |
| Gaussian CI | `mean_std_ci` | Noise sweep. `1.96 * np.std / sqrt(n)` with population standard deviation (`ddof=0`). | Reuse for that protocol. | Pandas `std` defaults to `ddof=1`. Mixing the two changes the interval. |
| Student-t CI | `parametric_ci` in `src/metrics.py`; `confidence_interval` in `storage_capacity.py` | Defined. No caller in the current tree. Storage capacity uses bootstrap instead. | Leave unused until a protocol actually needs them. | `confidence_interval` builds a PennyLane NumPy array and calls SciPy. Dead code is easy to mistake for the reported interval. |
| Bootstrap CI | `bootstrap_mean_ci` (capacity, 2000, Python loop); local `bootstrap_ci` in inference (200, vectorized, unseeded); `bootstrap_diff` in statistical significance (5000, paired QAM − Hopfield); unused `bootstrap_ci` in `scaling.py` | Three live implementations. | Keep each implementation attached to the experiment that produced the archived CI. | Inference aggregation does not seed NumPy. Rebuilding the summary from the same raw rows can change the percentile columns. Scaling's `bootstrap_ci` is unused; the scaling plots use `1.96 * std / sqrt(trials)`. |
| Wilcoxon and Cliff's delta | `scipy.stats.wilcoxon` in capacity and statistical significance; `src.metrics.cliffs_delta` via `scripts/cliffs_delta.py` | Capacity stores the p-value on both model rows and writes `None` on failure. Statistical significance writes `1.0` on failure. Cliff's delta is a post-pass over `statistical_rows.csv`. Positive delta means Hopfield scores tend to exceed QAM scores. | Reuse. Keep the two failure values distinct, because the archived CSVs differ. | Capacity's log shows a SciPy divide warning when the paired samples match (perfect recall). |
| Plotting and reporting | Experiment `main` functions write some figures while they run. `scripts/plot_*.py` and `scripts/cliffs_delta.py` read CSVs and write other figures. | Both styles are in use. Some archived PNGs have no writer in the current tree. | Keep plot scripts as consumers of CSVs. New benchmark runs should write tables first and plot second. | Re-running an experiment `main` overwrites that experiment's CSV. Several plot scripts also overwrite CSVs (`critical_noise_thresholds.csv`, `cliffs_delta_table.csv`). |
| Experiment entry | `python -m experiments.<name>`, or `python -m scripts.run_all_experiments --only <name>` | Seven `main` functions. `run_all_experiments` lists them and, with `--run` or `--only`, launches each as a subprocess from `PROJECT_ROOT`. | Treat each module as a protocol, not as a plugin with a common config object. | `--run` executes every protocol, including the 216,000-trial inference grid, and overwrites archived CSVs. |
| Tests | `tests/test_imports.py`, `tests/test_paths.py` | Import, export, and path checks. No numeric assertion on recall, loss, or metrics. | Keep these tests. Add behavior tests around the unchanged helpers before moving call sites. | `test_paths.py` creates and then removes directories under `results/`. |
| Dependencies and paths | `pyproject.toml`, `requirements.txt`, `src/paths.py` | Python `>=3.10`. Pinned NumPy, pandas, matplotlib, PennyLane, PennyLane-Lightning, SciPy, seaborn, tqdm, psutil. This machine's venv is Python 3.13.7 and imports those pins. Experiments request `default.qubit`. | Keep the pins. Record `default.qubit` in every new run manifest. | Lightning is installed and unused by the experiment code. A device default change would be a different simulator. |

## Original experiment flow

The tree does not contain one original script. The direct paired comparison of synchronous Hopfield and a trained variational circuit is `experiments/noise_robustness_sweep.py`. The other six modules are separate protocols. Their "QAM" rows are different objects. `docs/methodology.md` already records that split; the call graph below is what the current functions do.

### Primary protocol: noise robustness sweep

`main` writes only after every `(n_bits, corruption)` cell finishes. A cell's trials run in a `multiprocessing.Pool` of size `max(1, cpu_count() - 1)`.

```
main
  results_subdir("noise_robustness_sweep", "summary" | "plots")
  for n_bits in [4, 6, 8, 10]:
    for corruption in [0.0, 0.1, 0.2, 0.3, 0.4]:
      Pool.map(run_trial, (n_bits, corruption, seed) for seed in range(30))
      mean_std_ci on the 30 trial scores
  DataFrame -> summary/noise_robustness_sweep.csv
  error-bar figures -> plots/noise_full_acc_n{n}.png, plots/noise_bit_acc_n{n}.png
```

`run_trial`:

1. `np.random.seed(seed)` and `pnp.random.seed(seed)`. Seeds are `0..29`.
2. `m = max(1, int(0.3 * n_bits))`. Patterns come from `np.random.choice([-1, 1], size=(m, n_bits))`.
3. `Hopfield(n_bits)` uses the default 10 steps. `train` builds `W`.
4. For every stored pattern, `corrupt_bernoulli(pattern, corruption)` flips bits on the global NumPy RNG. `recall_timed` returns the recalled pattern and the wall time divided by `n_bits`.
5. Full-pattern accuracy is the fraction of stored patterns recovered exactly. Bit accuracy is matching bits over `m * n_bits`.
6. `QAM(n_bits)` builds a `default.qubit` device, reseeds with `pnp.random.seed(42)`, and draws `0.01 * np.random.randn(depth, n_bits, 3)`. Depth is `min(6, n_bits)`.
7. `QAM.train` runs 300 Adam steps at lr `0.05`. The loss averages, over patterns, the sum of squared Pauli-Z errors.
8. The same stored patterns are corrupted again and passed to `QAM.recall`, which applies `sign` and maps zeros to `+1`.

Because step 6 resets the global NumPy RNG, the random draws inside QAM initialisation and the later QAM corruption masks do not continue the trial seed's stream. Hopfield corruption happens before that reset. This was checked by confirming that `pennylane.numpy.random.seed` changes the next value returned by `numpy.random.rand` (the two `seed` functions are different objects; the stream they control is shared).

The sweep does not write per-trial rows. The archived summary has 20 data rows (4 lengths × 5 noise levels).

### Sibling protocols

**Storage capacity** (`experiments/storage_capacity.py`). For each `n_bits` in `{8, 12, 16}` and each `m` in `1..n` (for `n=16`, `m` is `1..10, 12, 14, 16`), five seeds `0..4`:

`numpy.random.seed(seed)` → `choice` patterns → `Hopfield.train` / `Hopfield.recall` on `corrupt_fixed_count(..., 0.15)` → `train_ry_linear_qam(..., depth=min(8, n), steps=100, lr=0.05, return_tuple=True)` → `qam_recall` on a freshly corrupted copy. Aggregation is `bootstrap_mean_ci` with 2000 resamples on the global NumPy RNG, then Wilcoxon. Critical capacity is the first ratio whose mean full accuracy is below `0.75`.

**Circuit depth** (`experiments/circuit_depth.py`). Cases `(n, m) = (8, 4), (12, 8), (16, 12)`, depths `{1, 2, 4, 6, 8, 10, 12}`, five seeds. Same trainer as capacity with `return_tuple=False`. Recall is inlined: `sign` of the circuit output, zeros to `+1`, then `full_and_bitwise_accuracy`. No Hopfield arm.

**Convergence** (`experiments/convergence.py`). Same `(n, m)` pairs, seeds `{0, 1, 2}`, `train_ry_linear_qam_record_loss` for 300 steps. The CSV has 301 rows: loss before each step and one evaluation after the last update. Summary columns include loss at steps 0, 50, 100, 200, and the final step, plus `post100_improvement_pct`.

**Scaling** (`experiments/scaling.py`). `n_bits` in `{4, 6, 8, 12, 16}`, `m = min(max(1, n // 2), 6)`, 30 trials, seed `123 + trial_id + n_bits * 100` applied to PennyLane NumPy. Patterns and `HopfieldAsyncRandom` permutations share that stream. The scaling `QAM` is trained once per trial, then both models are scored at noise `{0.0, 0.1, 0.2, 0.3}` with an inlined Bernoulli mask. Raw file: `results/scaling/raw/scaling_sweep.csv` (1200 rows = 5 lengths × 30 trials × 4 noise levels × 2 models).

**Statistical significance** (`experiments/statistical_significance.py`). Grid `n ∈ {8, 12, 16}` × capacity `{0.2, 0.3, 0.4, 0.5}` × noise `{0.1, 0.2}` × seeds `0..29` (720 tasks). Each task seeds both NumPy and PennyLane NumPy, trains synchronous Hopfield and the PauliX QAM, and returns mean full and bit accuracy. `bootstrap_diff` resamples the paired difference QAM minus Hopfield 5000 times. Wilcoxon failure is stored as `1.0`.

**Inference** (`experiments/noise_robustness_inference.py`). Grid `n ∈ {8, 12, 16, 24, 32, 48, 64, 96, 128}` × capacity `{0.1, 0.25, 0.5, 0.75, 1.0}` × 16 noise levels from `linspace(0, 0.5, 16)` × 300 trials = 216,000 tasks. Seed formula:

```
42 + trial_id + int(n_bits * 1000) + int(capacity_ratio * 100) + int(noise_p * 1000)
```

Each trial stores `m = max(1, int(n * capacity))` patterns, picks one target, applies an inlined Bernoulli mask, recalls with `hopfield_async_sequential_recall(..., steps=20)`, and recalls with `nearest_neighbor_recall`. Metrics from `compute_metrics` are bit accuracy, full accuracy, and Hamming error. Recovery gain is input Hamming error minus output Hamming error. The pool uses the `spawn` context and `min(12, cpu_count())` workers. The raw CSV has 432,000 rows. Summary bootstrap (200 resamples) is unseeded.

Two later scripts define different critical-noise quantities from that summary:

- `scripts/plot_critical_noise_threshold.py`: last noise with mean full accuracy still at least `0.50`. Archived table: 90 data rows.
- `scripts/plot_inference_paper_figures.py`: first noise with mean full accuracy below `0.90`. Groups that never cross that floor are omitted. Archived table: 87 data rows. The three keys present only in the 0.50 table are `(qam, n=8, capacity=0.1)`, `(qam, n=12, capacity=0.1)`, and `(qam, n=16, capacity=0.1)`.

## Tests

Command run in this audit, from the repository root, using the project venv:

```bash
./venv/bin/python -m pytest -q
```

Result: exit code 0, 40 tests passed.

| Test | What it checks |
|---|---|
| `test_src_exports_exist` | Named callables exist on the five `src` modules. |
| `test_src_qam_has_no_qam_class` | `src.qam` has neither `QAM` nor `QuantumAssociativeMemory`. |
| `test_entry_point_imports` | Each experiment and script module imports and defines `main`. |
| `test_no_sys_path_hacks` | Those modules do not call `sys.path.insert` or `sys.path.append`. |
| `test_project_root_is_repository_root` | `PROJECT_ROOT` contains `pyproject.toml`, `src/paths.py`, and `experiments/`. |
| `test_results_subdir_is_under_project_root` | A subdirectory is created under `results/` even if cwd is elsewhere, then removed. |
| `test_experiment_dir_does_not_use_cwd` | `experiment_dir` is rooted at `PROJECT_ROOT / "results"`. |
| `test_existing_result_csv_is_findable` | `results/noise_robustness_sweep/summary/noise_robustness_sweep.csv` exists. |

No test trains a circuit, recalls a pattern, or checks a metric value.

## Dependencies and environment

From `pyproject.toml` / `requirements.txt`, and confirmed importable in `venv` during this audit:

| Package | Pin | Role in this tree |
|---|---|---|
| numpy | 2.4.4 | Patterns, Hopfield, metrics |
| pandas | 3.0.2 | CSV tables |
| matplotlib | 3.10.8 | Figures |
| pennylane | 0.44.1 | Circuits, Adam, and the NumPy RNG used by scaling |
| pennylane-lightning | 0.44.0 | Installed. Experiment code requests `default.qubit`. |
| scipy | 1.17.1 | Wilcoxon, and the unused Student-t helpers |
| seaborn | 0.13.2 | Scaling plots |
| tqdm | 4.67.3 | Storage-capacity progress |
| psutil | 7.2.2 | Noise-sweep system printout |
| pytest | `>=8.0` extra `dev` | The venv has pytest 9.1.1 |

Python requirement is `>=3.10`. The venv interpreter used here is Python 3.13.7. Package install is `pip install -e .` from the repository root (`hopfield-qam-research` 0.1.0). No GPU, QPU, or credential is required. The simulator is CPU `default.qubit`.

Importing matplotlib in a restricted environment can fail to write the default font cache. This audit set `MPLCONFIGDIR` to the repository `.mplconfig` directory, which is gitignored.

## Reuse opportunities

- `Hopfield`, `hopfield_hebbian_weights`, `hopfield_async_sequential_recall`, and `HopfieldAsyncRandom` already separate the three recall schedules. A benchmark runner can call them without rewriting Hebbian storage.
- `train_ry_linear_qam` and `qam_recall` already are the capacity, depth, and convergence model.
- `corrupt_bernoulli`, `corrupt_fixed_count`, and the metric helpers already match the definitions used by the callers that import them.
- `results_subdir` is the right place for new artefacts, provided new runs use a new experiment name or an explicit output directory so `results/` stays intact.
- Archived CSVs under `results/` are the numerical record. Plot scripts already know how to read them.
- `baseline/configs/original_experiments.json` transcribes the in-code constants so a later runner can load them without retyping the grids.

## Architectural problems

1. **Several models share the label QAM.** Rot full-batch, RY-linear, Rot minibatch, PauliX, and Hamming nearest neighbor are different procedures. Inference writes the nearest neighbor as `model="qam"`.
2. **Protocols own data generation, training, scoring, aggregation, and plotting.** There is no place to change the output directory, the grid, or the seed without editing the scientific module.
3. **Randomness is global.** Seeds are process-wide `numpy.random.seed` / PennyLane seed calls. Call order is part of the result. The noise-sweep reseed to 42 is one concrete case.
4. **Raw trials are missing for three protocols.** The noise sweep, storage capacity, and circuit depth persist aggregates only. The original patterns and per-pattern predictions are not recoverable from those CSVs.
5. **The same quantity is implemented more than once.** Bernoulli masks, accuracy, and bootstrap intervals differ by file, including seeded versus unseeded resampling and `ddof`.
6. **Running a study overwrites its archive.** Every experiment `main` writes into `results/<name>/`. `scripts/run_all_experiments` documents that.
7. **Figures and tables have drifted from the current writers.** Archived files with no current `savefig` or `to_csv` call: `results/noise_robustness_sweep/plots/bitwise_noise.png`, `results/noise_robustness_sweep/plots/full_pattern_noise.png`. Scaling figures on disk match `scripts/plot_scaling.py`. The names written by `experiments/scaling.py` (`full_acc_vs_n_bits.png`, `bit_acc_vs_n_bits.png`, `accuracy_gap.png`, `train_time_vs_n_bits.png`) are absent. `scripts/plot_noise_robustness_inference.py` would write `bit_accuracy_n*.png`, `full_accuracy_n*.png`, `runtime_scaling.png`, `bit_acc_mean_capacity_effect.png`, `full_acc_mean_capacity_effect.png`, `hopfield_heatmap.png`, and `qam_heatmap.png`; those files are absent. `results/noise_robustness_inference/plots/` contains only `critical_noise_threshold_vs_capacity_ratio.png`.
8. **Logs record an older layout.** `results/storage_capacity/storage_capacity_sweep.log` reports paths `results_storage_capacity/...` and the completion line that HEAD prints (`Publication-quality storage capacity experiment complete.`). `results/noise_robustness_inference/experiment.log` reports `noise_robustness_inference_results/...` and 216,000 completed tasks. `results/noise_robustness_sweep/experiment.log` is a failed launch against `/Users/sanjay_shiva/Downloads/Quantum/ResearchCode/...`, not a successful run log. The CSVs now live under `results/<experiment>/`.
9. **Timing is wall-clock and not comparable.** The noise sweep reports inference time per bit. Capacity and scaling report time per pattern. Inference reports time for the whole recall. Training times depend on the machine.
10. **Dead helpers look like the analysis.** `parametric_ci`, storage capacity's `confidence_interval`, scaling's `bootstrap_ci`, and the unused constant `ansatz_type` are easy to wire up later and silently change the reported numbers.

## Reproducibility risks

- Archived metrics were not regenerated in this audit. Matching constants and CSV schemas support treating those files as the baseline record. A byte-level claim that today's `python -m experiments.<name>` produced those exact files is not established: logs still name the pre-reorganization directories.
- Patterns, weights, and trained parameters are not stored. Reproduction requires the same library versions, the same global RNG consumption order, and the same device (`default.qubit`).
- Inference summary confidence intervals are an unseeded bootstrap. The raw trial rows are the reproducible artefact; the summary percentile columns are not pinned by `RANDOM_SEED_BASE`.
- Noise-sweep QAM initialisation and QAM corruption share a stream that is reset to seed 42 inside `QAM.__init__`. Trial seeds `0..29` still affect pattern draws and Hopfield corruption, because those happen first.
- Capacity and depth reuse seeds `0..4` for every `(n, m)` or `(n, depth)` cell. That is what the code does. It is not 5 independent global seeds across the whole sweep.
- Wilcoxon failure handling differs (`None` versus `1.0`). A later unification would rewrite archived p-values.
- Multiprocessing is part of the noise sweep, the inference study, and the statistical-significance study. Result values should not depend on worker count if the per-trial seed is applied inside the worker. Wall-clock columns will.
- `tests/test_paths.py` creates directories under `results/`. The probes were removed when this audit's pytest run finished (`path_audit_probe_subdir` and `path_audit_probe_dir` were absent afterward).

## Recommended minimal refactoring sequence

Do this only after this audit. Leave the numerical bodies of the models alone until characterization tests exist.

1. **Leave `results/` intact.** New runs need an output directory that is not an existing experiment folder. Copy or hash the archived CSVs before any `main` is launched.
2. **Lock current helper behavior with tests.** Small deterministic checks for `Hopfield.recall`, both corrupters, and the metric helpers. Use explicit `numpy.random.Generator` instances in the tests so they do not depend on global state.
3. **Load grids from the transcribed config.** `baseline/configs/original_experiments.json` already copies the constants. A runner should pass those values into the existing functions rather than re-declaring them.
4. **Add thin adapters around the existing callables.** One adapter per procedure that already exists: three Hopfield recalls, four variational circuits, one nearest neighbor. Adapters call the current functions. They do not merge circuits.
5. **Record a manifest per run.** Protocol name, config, package versions, device, seed formula, and output paths. Persist patterns and per-trial predictions for any new run.
6. **Separate aggregation and plotting from training.** The current plot scripts can stay the consumers. Experiment `main` functions remain the reference implementations until a runner reproduces one small archived cell.
7. **Thread an explicit RNG only when a characterization test shows the draw sequence.** The noise-sweep reseed and the scaling shared stream are behavior, not accidents to clean up on the first refactor.

The first code change worth making is the characterization test plus a non-clobbering output path. Unifying the QAM classes before that would change the scientific record.
