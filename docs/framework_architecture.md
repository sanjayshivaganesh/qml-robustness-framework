# Benchmark framework architecture

Design date: 2026-10-09. This document specifies a framework. It does not add the framework, and it does not change any research algorithm.

The numerical record remains the code in `src/` and `experiments/` plus the archived files under `results/`. `docs/architecture_audit.md`, `docs/baseline_experiments.md`, and `baseline/configs/original_experiments.json` are the inspection record this design uses. No experiment `main` was run for this design. Equality between today's `python -m experiments.<name>` and the archived CSV bytes is not established.

## What the framework is

A thin runner package, `benchmark/`, that calls the implementations already in the tree. Each existing experiment stays a protocol with its own model, corruption rule, seed order, metric, and output schema. The framework's job is to name those pieces, call them in the historical order, and write new runs somewhere other than `results/`.

`src/` and `experiments/` stay the reference implementations. Framework code does not receive a second copy of a circuit, a loss, a recall schedule, or a corruption rule.

## Proposed package

Nothing below exists yet. Paths are relative to the repository root.

```
benchmark/
  __init__.py
  identity.py          # procedure id, RNG contract, describe()
  output.py            # run directories outside results/
  manifest.py          # versions, device, seed formula, procedure ids
  config_loader.py     # read-only load of baseline/configs/original_experiments.json
  runner.py            # stage dispatcher used by protocol modules
  scoring.py           # which existing metric helper each protocol calls
  adapters/
    __init__.py
    hopfield_synchronous.py
    hopfield_async_sequential.py
    hopfield_async_random.py
    qam_ry_linear.py
    qam_rot_fullbatch.py
    qam_rot_minibatch.py
    qam_paulix.py
    hamming_nearest.py
  data/
    __init__.py
    patterns.py
    corruption.py
  protocols/
    __init__.py
    noise_robustness_sweep.py
    storage_capacity.py
    circuit_depth.py
    convergence.py
    scaling.py
    statistical_significance.py
    noise_robustness_inference.py
```

`pyproject.toml` `[tool.setuptools.packages.find]` currently includes `src`, `experiments`, and `scripts` only. The stage that adds `benchmark/` also adds that package to `include`. `runs/` is the default output root and belongs in `.gitignore`. `results/` stays tracked and is not a write target for `benchmark/`.

Tests for the framework are new files under `tests/`. The existing import and path tests stay as they are.

## A. Model adapters

### Common interface

Every adapter exposes three facts and nothing else as a shared contract:

| Member | Responsibility |
|---|---|
| `identity` | Stable `procedure_id`, family (`hopfield`, `variational_qam`, or `hamming_nearest`), and the dotted source symbol that actually runs. |
| `rng_contract` | Which stream the procedure draws from, and whether construction reseeds. |
| `describe()` | JSON-serializable identity, hyperparameters passed through to the source, and the RNG contract. Written into the run manifest. |

There is no shared `fit` / `predict` method, no shared loss, and no shared step count. A protocol runner calls the methods that procedure already has. An adapter that does not train does not grow a `train` method so that it matches one that does.

Adapters do not corrupt inputs, do not compute accuracy, and do not write files. They return the object or the tuple the source function already returns. Extra scientific fields (energy, per-bit time) stay on that return value. They are not filled in with placeholders by a base class.

`describe()` records labels. The CSV column `model="qam"` in the inference archive is a schema fact of that protocol, handled at persistence time. It is not the adapter's `procedure_id`.

### Procedures

Eight procedures. The word QAM in an archived CSV is not an identifier.

| `procedure_id` | Source that runs | Used by | Adapter |
|---|---|---|---|
| `hopfield_synchronous` | `src.hopfield.Hopfield` | Noise sweep, storage capacity, statistical significance | Direct. `build(n_bits, steps=10)` returns a real `Hopfield`. `train`, `recall`, and `recall_timed` are the methods on that instance. 10 synchronous `sign(Wx)` steps. Zeros become `+1`. `recall_timed` returns `(pattern, wall_time / n_bits)`. |
| `hopfield_async_sequential` | `hopfield_hebbian_weights`, `hopfield_async_sequential_recall` | Inference | Direct. `store(patterns)` returns `W`. `recall(W, query, steps=20)` returns `(state, final_energy, energy_drop)`. Index order. Energy is `E = -1/2 x^T W x`. A zero local field becomes `+1`. |
| `hopfield_async_random` | `src.hopfield.HopfieldAsyncRandom` | Scaling only | Direct. The scaling protocol imports this class under the local name `Hopfield`; the adapter imports `HopfieldAsyncRandom` and does not reuse that alias. 10 sweeps. Permutations come from PennyLane NumPy. Update rule stays `+1` if the local field is `>= 0`, else `-1`. The adapter does not accept an injected generator. |
| `qam_ry_linear` | `src.qam.build_ry_linear_circuit`, `train_ry_linear_qam`, `train_ry_linear_qam_record_loss`, `qam_recall` | Storage capacity, circuit depth, convergence | Direct function passthrough. RY encoding `RY(x·π)`, RY + linear CNOT, parameter shape `(depth, n)`, init `0.01 * randn`, Adam, per-bit mean squared error averaged over patterns. Capacity passes `return_tuple=True`, depth `min(8, n)`, 100 steps, lr `0.05`, then `qam_recall`. Convergence calls `train_ry_linear_qam_record_loss` (300 steps, same depth policy) and does not recall. Depth passes `return_tuple=False` and recalls with the sign block written in `experiments/circuit_depth.py` (`sign`, zeros to `+1`). The depth protocol keeps that block. Switching depth onto `qam_recall` waits on a test that both map one fixed expectation vector the same way, and even then the switch is only inside `benchmark/`, not inside `experiments/circuit_depth.py`. |
| `qam_rot_fullbatch` | `experiments.noise_robustness_sweep.QAM` | Noise sweep | Thin wrapper. The class stays in that module. `build(n_bits)` calls `QAM(n_bits)`, which is depth `min(6, n)`, 300 Adam steps, lr `0.05`, RX and RY encoding, `Rot` + linear CNOT, parameter shape `(depth, n, 3)`. Loss is the mean over patterns of the **sum** of squared Pauli-Z errors. `__init__` calls `pnp.random.seed(42)` before `np.random.randn`. The adapter does not seed again and does not skip that call. `recall` remains the method that returns `(pattern, wall_time / n_bits)`. |
| `qam_rot_minibatch` | `experiments.scaling.QAM` | Scaling | Thin wrapper. The class stays in that module. The protocol calls `QAM(n_bits, depth=min(6, n_bits), steps=200, lr=0.05)`. The class default `depth=3` is not the scaling protocol. RY encoding, `Rot` + linear CNOT. Each `loss` draws a minibatch of `min(2, m)`. `train` evaluates `loss` and then `opt.step(loss, ...)`, so each step draws two batches. Early stop is 15 consecutive steps whose absolute loss change is below `1e-4`. The adapter does not cache one batch or turn the second evaluation off. |
| `qam_paulix` | `build_qam`, `train_qam`, `qam_recall` in `experiments/statistical_significance.py` | Statistical significance | Thin wrapper. Those three functions stay in that module. The adapter imports the local `qam_recall` under an alias so it does not bind `src.qam.qam_recall`. PauliX where the bit is `-1`, RY + linear CNOT, depth 10, 100 epochs, lr `0.02`, init `normal(0, 0.1)`, per-bit mean squared error. |
| `hamming_nearest` | `src.qam.nearest_neighbor_recall` | Inference rows with `model="qam"` | Direct. `recall(patterns, query)` returns the stored row with the smallest Hamming distance. Distance in the source is `sum(patterns != query) / n_bits`, then `argmin`. There is no circuit and no training method. New manifests record `procedure_id = hamming_nearest`. A reproduction of the inference schema still writes the column value `qam`, and the manifest says what that column means. |

Hebbian storage is the same formula in `Hopfield.train`, `hopfield_hebbian_weights`, and `HopfieldAsyncRandom.train` (outer products, zero diagonal, divide by `m`). Each adapter calls the function its protocol already calls. A later test may compare the three weight matrices on one ±1 batch. Agreement would be a finding. It would not be a reason to collapse the three call paths.

### What is reused directly

Call the existing symbol. The adapter module is an import path plus `describe()`:

- `Hopfield`, including `train`, `recall`, and `recall_timed`
- `hopfield_hebbian_weights` and `hopfield_async_sequential_recall`
- `HopfieldAsyncRandom`
- `train_ry_linear_qam`, `train_ry_linear_qam_record_loss`, `qam_recall`, `build_ry_linear_circuit`
- `nearest_neighbor_recall`

### What needs a thin experiment-specific adapter

The body stays in the experiment module. The adapter constructs or calls it and refuses to reimplement it:

- `experiments.noise_robustness_sweep.QAM`
- `experiments.scaling.QAM`
- `experiments.statistical_significance.build_qam`, `train_qam`, and the module-local `qam_recall`

Circuit-depth recall stays a three-line sign map inside `benchmark/protocols/circuit_depth.py`, copied only as a call to the same operations the experiment file performs, until the vector test above passes. The experiment file itself is not edited to call `qam_recall`.

## B. Dataset and perturbation interfaces

### Patterns

`src.data_generation.random_bipolar_patterns` draws `rng.choice([-1, 1], size=(n_patterns, n_bits))`. Experiment loops do not call it. They call `choice` on whichever NumPy they imported:

| Protocol | Draw |
|---|---|
| Noise sweep, statistical significance, inference | `numpy.random.choice` after `numpy.random.seed` |
| Storage capacity, circuit depth, convergence | `numpy.random.choice` via the `onp` alias, after `onp.random.seed` |
| Scaling | `pennylane.numpy.random.choice` after `pennylane.numpy.random.seed` |

`benchmark/data/patterns.py` exposes the existing helper for tests and for new runs that pass an explicit `rng`. Historical protocol runners keep the `choice` call on the same module the experiment uses, in the same position in the seed stream. Pointing those runners at `random_bipolar_patterns` is allowed only after a draw-sequence test shows the same array and the same subsequent RNG state. `numpy.random.Generator` (`default_rng`) is a different generator family from the legacy global `RandomState`. A matching seed integer does not make those sequences equal. Historical runners do not switch to `default_rng`.

A pattern batch recorded by a new run carries the array, `n_bits`, `m`, the generator name (`numpy.random` or `pennylane.numpy.random`), and the seed integer that was applied before the draw. The archived CSVs do not contain patterns. New runs can store them under `runs/`. That storage is additional. It is not a rewrite of the archive.

Inference also draws one target index with `numpy.random.randint(0, n_patterns)` after the pattern draw. That draw stays inside the inference protocol, immediately after the pattern `choice`, because it consumes the same stream as the corruption mask that follows.

### Two corruption procedures

These stay separate functions. A numeric level such as `0.15` does not select the procedure.

| `procedure_id` | Source | Rule | Protocols that already call the helper | Protocols that inline the same mask |
|---|---|---|---|---|
| `bernoulli` | `corrupt_bernoulli` | Each bit flips independently when `rng.rand(n) < rate`. Rate `0` flips nothing. | Noise sweep, statistical significance | Scaling and inference: `rand(n) < p`, then `*= -1` |
| `fixed_count` | `corrupt_fixed_count` | Flip `max(1, int(n * noise_level))` distinct indices via `rng.choice(..., replace=False)`. At `noise_level == 0` the count is still 1. Capacity and depth always pass `0.15`, so that edge is unused by them and remains the helper's behavior. | Storage capacity, circuit depth | None |

`benchmark/data/corruption.py` exposes two callables:

- `apply_bernoulli(pattern, rate, rng=None)` → `corrupt_bernoulli`
- `apply_fixed_count(pattern, noise_level, rng=None)` → `corrupt_fixed_count`

`rng=None` keeps today's fallback to the global `numpy.random`. Historical runners that already call the helper keep that fallback so the global stream is consumed in the same place. Scaling and inference runners keep the inline mask in the protocol module until a test shows that `corrupt_bernoulli` consumes one `rand(n)` draw and returns the same bits on a shared `RandomState`. After that test, a `benchmark/` runner may call the helper. `experiments/scaling.py` and `experiments/noise_robustness_inference.py` stay as written.

Queries in a new run record `procedure_id`, the level, the target, the corrupted vector, and the flip count. Bernoulli's flip count is a result of the mask. Fixed-count's flip count is `max(1, int(n * level))`.

### Who is corrupted, and when

The schedule is part of the protocol, not part of the corruption function.

- Noise sweep: every stored pattern, once for Hopfield, then again for the Rot full-batch QAM after `QAM.__init__` has reseeded to 42. The two masks are different draws.
- Storage capacity and circuit depth: every stored pattern, a fresh fixed-count draw per recall. Hopfield's draws happen before RY-linear parameter init. QAM's draws happen after training.
- Scaling: both models are trained once per trial. For each noise level, every pattern is masked for Hopfield (and random-order recall consumes more of the stream), then every pattern is masked for QAM.
- Statistical significance: every stored pattern, Bernoulli, Hopfield first, then PauliX corruption after parameter init.
- Inference: one chosen pattern, one Bernoulli mask, then both recalls see that same query.
- Convergence: no corruption and no recall metric.

## C. Metrics

`benchmark/scoring.py` maps each protocol to the helpers it already uses. It does not define a new accuracy or a new interval. Callers pass the arrays the helper already expects.

### Reused as the definition

| Helper | Definition to preserve | Who uses it now |
|---|---|---|
| `bit_accuracy` | Fraction of matching bits, `float`. | Via `compute_metrics`. |
| `full_accuracy` | `1.0` only when every bit matches, else `0.0`. | Via `compute_metrics`. |
| `hamming_error` | Fraction of mismatches. | Via `compute_metrics`. |
| `compute_metrics` | `(bit_accuracy, full_accuracy, hamming_error)`. | Inference. |
| `full_and_bitwise_accuracy` | Mean of per-pattern exact matches, and mean of per-pattern bit fractions. | Storage capacity, circuit depth. |
| `pattern_accuracy_pair` | `(exact_match_as_int, bit_accuracy)` for one pair. | Statistical significance. |
| `mean_std_ci` | Mean, population standard deviation (`numpy.std`, `ddof=0`), and `1.96 * std / sqrt(n)`. | Noise sweep. |
| `bootstrap_mean_ci` | Mean and percentile interval. Python loop of `rng.choice` then `mean`, default 2000 resamples. Default `rng` is the global NumPy stream. | Storage capacity, through the local `bootstrap_ci` wrapper that passes `onp.random`. |
| `cliffs_delta` | `(n_{x>y} - n_{x<y}) / (n_x * n_y)`. | `scripts/cliffs_delta.py`, with Hopfield scores as `x` and the QAM column as `y`. Positive means Hopfield tends to be larger. |
| `cliffs_delta_magnitude` | Bands at `0.147`, `0.33`, `0.474`. | Same script. |

The noise sweep and scaling compute full-pattern accuracy as the fraction of exact pattern matches and bit accuracy as matching bits over `m * n_bits` (the sweep) or the mean of per-pattern bit fractions (scaling). Those are the same quantities as `full_and_bitwise_accuracy` on ±1 arrays. A characterization test locks that equality on a fixed batch. After it passes, new runners may call the helper. The experiment modules keep their inline reductions until a separate edit, which this design does not schedule.

### Kept attached to one protocol

A framework-level `ci()` that picked one of these would change archived columns. Each stays named and local.

| Quantity | Where it lives | Contract |
|---|---|---|
| Inference percentile bootstrap | `aggregate_results` local `bootstrap_ci`, 200 resamples, vectorized `numpy.random.choice` | Unseeded. Rebuilding the summary from the same raw rows can change `bit_ci_*` and `full_ci_*`. |
| Inference Gaussian columns | `1.96 * groupby std / sqrt(TRIALS_PER_CONFIG)` | Pandas `std` defaults to `ddof=1`. This is not `mean_std_ci`. |
| Paired bootstrap | `statistical_significance.bootstrap_diff` | 5000 resamples of the paired difference QAM minus Hopfield, via `numpy.random.randint` in a Python loop. |
| Scaling `bootstrap_ci` | `experiments/scaling.py` | Defined and unused. Scaling plots use `1.96 * std / sqrt(trials)` in `scripts/plot_scaling.py`. The runner does not start calling the unused helper. |
| `parametric_ci` and storage capacity's `confidence_interval` | `src/metrics.py`, `experiments/storage_capacity.py` | No caller. Runners do not route a reported interval through them. |
| Wilcoxon | `scipy.stats.wilcoxon` in capacity and in statistical significance | Capacity stores `None` on failure. Statistical significance stores `1.0`. Both values stay. |
| Recovery gain | Inference only | `input_hamming_error - output_hamming_error`. |
| Convergence summary | `experiments/convergence.py` | Mean loss at steps 0, 50, 100, 200, and the last recorded point. `post100_improvement_pct = (loss_100 - loss_final) / max(abs(loss_100), 1e-12) * 100`. `std_loss` is NumPy's population `std`. The recorded series has `steps + 1` points. |
| Critical capacity | Storage capacity | First `m/n` whose mean full accuracy is below `0.75`. |
| Critical noise `0.50` | `scripts/plot_critical_noise_threshold.py` | Last noise whose mean full accuracy is still at least `0.50`. |
| Critical noise `0.90` | `scripts/plot_inference_paper_figures.py` | First noise whose mean full accuracy is below `0.90`. Groups that never cross are omitted. |

### Timing

Wall-clock figures are machine-dependent. They are recorded with the placement the protocol already uses, and they are not acceptance criteria for a scientific comparison.

| Protocol | What is timed |
|---|---|
| Noise sweep | Train time separate from recall. Recall time is divided by `n_bits` inside `Hopfield.recall_timed` and `QAM.recall`. |
| Storage capacity, scaling | Train time separate. Inference time is the loop over `m` patterns, divided by `m`. |
| Inference | One clock around weight construction plus sequential recall. Nearest-neighbour time is only the lookup. Energy columns for the nearest-neighbour row are `NaN` because that procedure does not compute energy. |

## D. Experiment orchestration

`benchmark/runner.py` runs a list of stages supplied by one protocol module. It does not require every stage. Plotting is not a stage of training.

| Stage | Owner | Behaviour |
|---|---|---|
| Configuration | `config_loader.py` plus the protocol module | Load the matching object from `baseline/configs/original_experiments.json`. Numeric grids in that file are data. Formula strings (`max(1, int(...))`, seed formulas) stay documentation; the protocol module keeps the Python that implements them. The loader does not `eval` those strings. The runner checks that imported constants (`n_bits_list`, `trials`, and the other module-level grids) still match the JSON. A mismatch stops the run. |
| Dataset creation | Protocol, calling `benchmark/data/patterns.py` only when its RNG test allows | Same `choice` site and seed call as the experiment. |
| Model construction and training | The one adapter named by the protocol | Hyperparameters come from the protocol, which read them from the JSON where the JSON stores a number. Construction order matches the experiment, including the noise-sweep reseed inside `QAM.__init__`. |
| Corruption | `apply_bernoulli` or `apply_fixed_count`, or the protocol's inline mask | The procedure id is fixed by the protocol. |
| Recall / inference | Adapter method that already exists for that procedure | Convergence skips this stage. |
| Metric calculation | `benchmark/scoring.py` dispatch for that protocol | Helpers from section C. No new reduction. |
| Result persistence | `benchmark/output.py` | Tables and a manifest under the run directory. Patterns and per-trial predictions are written for new runs when the protocol produces them. The noise sweep, storage capacity, and circuit depth archives store aggregates only; a new run of those protocols still writes per-trial rows under `runs/` so the trial record exists. Those extra files do not replace the archived summary schema. A second table can mirror the archived column names for later comparison. |
| Reporting | Existing `scripts/plot_*.py` as readers, or a copy that takes an output directory | Training does not call `savefig`. Plot scripts that also call `to_csv` (`plot_critical_noise_threshold`, `cliffs_delta`, `plot_inference_paper_figures`) are not invoked on `results/` by the runner. |

Stage lists for the seven protocols:

| Protocol | Stages that run | Stages that do not |
|---|---|---|
| `noise_robustness_sweep` | Config, patterns, synchronous Hopfield train, Bernoulli on every pattern, timed recall, **then** Rot full-batch construct (reseed 42) and train, Bernoulli again, timed recall, inline full/bit means, `mean_std_ci` across trials | Energy, nearest neighbour |
| `storage_capacity` | Config, patterns, synchronous Hopfield, fixed-count, RY-linear `return_tuple=True`, fixed-count again, `full_and_bitwise_accuracy`, `bootstrap_mean_ci` (2000, global NumPy), Wilcoxon with `None` on failure, critical ratio `< 0.75` | A second QAM circuit |
| `circuit_depth` | Config, patterns, RY-linear `return_tuple=False`, fixed-count, inlined sign recall, `full_and_bitwise_accuracy` | Hopfield |
| `convergence` | Config, patterns, `train_ry_linear_qam_record_loss`, loss summary | Corruption, recall, accuracy |
| `scaling` | Config, PennyLane-NumPy patterns, `HopfieldAsyncRandom` train, Rot minibatch train once, then for each noise level the inline Bernoulli mask and both recalls | A reseed to 42. The shared stream is the protocol. |
| `statistical_significance` | Config, patterns, synchronous Hopfield, Bernoulli, PauliX train, Bernoulli, `pattern_accuracy_pair`, `bootstrap_diff` (5000), Wilcoxon with `1.0` on failure | The RY-linear trainer |
| `noise_robustness_inference` | Config, patterns, one target index, one inline Bernoulli mask, sequential async recall, nearest-neighbour recall, `compute_metrics`, recovery gain | Any variational circuit. Summary bootstrap stays unseeded and is reported as unseeded. |

Multiprocessing stays inside the protocol that already uses it, with the same context the module uses today:

- Noise sweep: `multiprocessing.Pool(max(1, cpu_count() - 1))`, `pool.map`.
- Statistical significance: `Pool(cpu_count())`, `pool.map`.
- Inference: `spawn` context, `min(12, cpu_count())` workers, `imap_unordered` with `chunksize=50`.

The trial seed is applied inside the worker in all three, so trial values are meant to be independent of worker count. Wall-clock columns are not. Inference raw row order follows completion order. Grouped means do not. Byte-identical raw CSVs are not the comparison criterion for that protocol.

`python -m experiments.<name>` and `scripts.run_all_experiments` remain the reference entry points. They still write under `results/<name>/`. The framework entry point is separate, for example `python -m benchmark.runner --protocol <name> --run-id <id>`, added in the migration. It never calls an experiment `main`.

Default invocations of the new runner are a declared smoke: one seed and the smallest `n_bits` in that protocol's grid, written to `runs/`, with the manifest field `comparable_to_archive: false`. The full grid requires an explicit `--full-grid`. The inference full grid is 216,000 tasks and is not an acceptance test. Storage capacity at `n=16` is hours of circuit training. Neither is launched to "see if the framework works."

## E. Randomness and reproducibility

### Current coupling

`numpy.random.seed` and `pennylane.numpy.random.seed` are different function objects. They control one shared legacy stream. `docs/baseline_experiments.md` records a check: after either seed call with `123`, the next `numpy.random.rand()` was `0.6964691855978616`.

What that means for each protocol:

| Protocol | Seed | What consumes the stream, in order |
|---|---|---|
| Noise sweep | Inside `run_trial`, both seed functions set to `0..29` | Pattern `choice`. Bernoulli masks for every Hopfield query. `QAM.__init__` calls `pnp.random.seed(42)`, then `np.random.randn` for parameters. Bernoulli masks for every QAM query continue from seed 42, not from the trial seed. Trial seeds still change patterns and Hopfield queries because those happen first. |
| Storage capacity, circuit depth | `onp.random.seed(seed)` with `seed in 0..4`, reapplied for every `(n, m)` or `(n, depth)` cell | Pattern `choice`, then that cell's Hopfield fixed-count draws (capacity) or the QAM fixed-count draws (depth). RY-linear init is `pennylane.numpy.random.randn` on the same stream, after the Hopfield corruption in capacity and after the pattern draw in depth. The same five seeds are reused across cells. They are not five seeds for the whole sweep. |
| Convergence | `onp.random.seed` for seeds `0, 1, 2` | Pattern `choice`, then `randn` inside `train_ry_linear_qam_record_loss`. |
| Scaling | PennyLane NumPy seed `123 + trial_id + n_bits * 100` | Pattern `choice`, Rot-minibatch `randn`, two minibatch `choice` draws per training step, then per noise level the Bernoulli masks and `HopfieldAsyncRandom` permutations (`pennylane.numpy.random.permutation`). One extra random call changes patterns, batches, masks, and update order together. |
| Statistical significance | Both seed functions set to `0..29` inside the worker | Pattern `choice`, Hopfield Bernoulli masks, `normal(0, 0.1)` parameter init, QAM Bernoulli masks. No reseed to 42. |
| Inference | `numpy.random.seed` only, formula `42 + trial_id + int(n*1000) + int(capacity*100) + int(noise*1000)` | Pattern `choice`, target `randint`, Bernoulli `rand`. No PennyLane draws. `aggregate_results` then draws an unseeded bootstrap on the parent process. |

Adam's parameter update is not documented here as RNG-free. The characterization tests snapshot draws at the seed, `choice`, corruption, and `randn` boundaries. They do not assume the optimizer is silent.

### Migration strategy

Historical behaviour is the global seed calls, in the order above, including the noise-sweep constructor's reset to 42.

1. Protocol runners in `benchmark/protocols/` call those seed functions in that order and then call the existing constructors. They do not pass a `Generator` into `Hopfield`, `QAM`, or `train_ry_linear_qam`.
2. Tests of pure helpers (`corrupt_*` with an explicit `rng`, metric helpers, synchronous recall on a hand-built pattern) use a `numpy.random.RandomState` or an array literal so they do not depend on test order.
3. Tests that must touch the global stream save `numpy.random.get_state()` and restore it in a `finally` block. The PennyLane seed function is the same stream; restoring the NumPy state is the cleanup. Those tests do not run in parallel with other global-stream tests.
4. An explicit random-state API is a later protocol version. It is added only after a test prints the draw sequence for one historical cell (patterns, masks, and the next RNG state) and the new path matches that sequence. `default_rng(seed)` is not that path. The first candidate, if a human asks for it, is a legacy `RandomState` threaded through helpers that already accept `rng`, without removing the global seed calls from the reference experiment modules.
5. The noise-sweep reseed and the scaling shared stream are behaviour to preserve. A runner that "fixes" QAM initialisation onto the trial seed, or that gives `HopfieldAsyncRandom` its own generator, is a different experiment and must not be compared with `results/`.

No stage in the migration plan claims that a new run reproduces an archived CSV. Train-time columns cannot match across machines. Inference summary percentiles are unseeded. Logs under `results/` still name `results_storage_capacity/` and `noise_robustness_inference_results/`.

## F. Result safety

### Archive

`results/` holds 71 tracked files, including the raw inference table (432,000 rows) and the summaries cited in `docs/baseline_experiments.md`. `src.paths.results_subdir` creates `results/<experiment>/...` and the experiment `main` functions then `to_csv` and `savefig` onto fixed names. That overwrites. The following scripts also overwrite derived tables:

- `scripts/plot_critical_noise_threshold.py` → `results/noise_robustness_inference/summary/critical_noise_thresholds.csv`
- `scripts/plot_inference_paper_figures.py` → `results/noise_robustness_inference/paper_figures/critical_noise_thresholds.csv`
- `scripts/cliffs_delta.py` → `results/statistical_significance/summary/cliffs_delta_table.csv`

`tests/test_paths.py` creates a directory under `results/` and deletes it. New tests follow that cleanup rule or, preferably, point `PROJECT_ROOT` at a `tmp_path` when they only need to see path logic. They do not add files next to archived CSVs.

### Safe writer

`benchmark/output.py` is the only writer the new runner uses.

- Default root: `<repo>/runs/<run_id>/`.
- Resolve the path. Refuse if it is `results/`, a parent of `results/`, or a child of `results/`.
- Refuse if the run directory already exists and is non-empty.
- Create the directory only after those checks. Write `manifest.json` first, then tables. Use a temporary file and `os.replace` for each table so a killed process does not leave a truncated CSV under a finished name.
- `results_subdir` is unchanged. Legacy `main` functions keep their current paths. The framework does not call them.

`runs/` is gitignored so a local smoke cannot be committed as if it were the archive.

### Legacy entry points

`python -m experiments.<name>` remains able to overwrite `results/<name>/`, because that is the reference command documented in `docs/baseline_experiments.md`. The migration adds one guard on the launcher only: `scripts/run_all_experiments.py --run` and `--only` return without launching a subprocess unless `--overwrite-archived-results` is also passed. That is a behaviour change to the launcher, not to any scientific module. It is called out as a human decision in `docs/migration_plan.md` before that edit lands.

Before any reference `main` is launched for a comparison a human has approved, hash the archived CSVs for that protocol and store the hashes outside `results/` (for example under `runs/<id>/archive_hashes.txt`). This design does not run those mains.

### Expensive protocols

The runner's default smoke does not build the inference task list of 216,000 items and does not build the full storage-capacity or statistical-significance grids. `--full-grid` is parsed and then still writes only under `runs/`. Acceptance tests never pass `--full-grid`. A smoke manifest records `grid: smoke` and the single cell it ran.

## G. Testing strategy

The 40 tests currently collected by `./venv/bin/python -m pytest -q` check imports, the absence of a `QAM` class on `src.qam`, `PROJECT_ROOT`, and that `results_subdir` lands under the repository. They do not train a circuit, recall a pattern, or check a metric. They stay. They are not evidence that the archived numbers match the code.

New tests are deterministic, write only under `tmp_path` or a monkeypatched root, and do not call any experiment `main`. Global RNG tests restore state. Markers:

- default `pytest -q`: characterization tests below, no 100-step or 300-step Adam run
- `slow`: optional, not part of the default acceptance command, one real `run_trial` or trainer call with the protocol's own settings for a single small cell, still not `main`

### Recall

Hand-specified ±1 patterns, no global seed:

- Synchronous `Hopfield`: after `train`, a clean pattern recalls to itself for a small separable set; a zero in `sign(Wx)` becomes `+1`; `steps` on the instance is the default 10 unless the test passes another value into `recall`.
- Sequential async: updates follow index order `0..n-1`; energy matches `-0.5 * x @ W @ x` on the input and on the final state; `energy_drop` is the difference the function returns; a zero local field writes `+1`.
- Random-order async: one pinned global seed, a snapshot of the permutation and the final state, state restored afterward. This test is the lock on "permutations use PennyLane NumPy." It does not reimplement the permutation.
- Hamming nearest neighbour: a query closer to row 1 than row 0 returns row 1; a tie returns the first `argmin` row.
- Sign map: a fixed expectation vector containing a `0` maps that coordinate to `+1`. Applied to `qam_recall` and, separately, to the three-line depth block.

### Corruption

Using an explicit `RandomState`, not the global stream:

- Bernoulli rate `0` changes no bits; rate `1` flips every bit; two calls consume two `rand(n)` draws.
- Fixed-count on `n=4`, level `0.15`, flips exactly one bit (`max(1, int(4 * 0.15))`); level `0` still flips one bit; indices are sampled without replacement.
- A paired test: from the same `RandomState` position, `corrupt_bernoulli` and the three-line inline mask return the same vector and leave the same RNG state. This authorizes a `benchmark/` runner to call the helper. It does not edit `experiments/scaling.py` or `experiments/noise_robustness_inference.py`.

### Metrics

- `bit_accuracy + hamming_error == 1` on a ±1 pair; `full_accuracy` is `1` only for an exact pair.
- `full_and_bitwise_accuracy` matches a hand-computed batch mean.
- `mean_std_ci` matches `np.std` with `ddof=0`, and the half-width differs from `1.96 * pandas.Series.std() / sqrt(n)` on a sample whose `ddof=1` std differs.
- `bootstrap_mean_ci(..., rng=RandomState(0))` is repeatable and stays inside the data min/max.
- `cliffs_delta([2, 2], [1, 0])` is positive; the swapped call is the negation.
- Wilcoxon failure values are not unified by a test that expects one sentinel.

### Small end-to-end

Default suite: build a `Hopfield(4)`, train two literal patterns, corrupt with `corrupt_bernoulli` on a `RandomState`, score with `compute_metrics`, write a CSV via `benchmark/output.py` into `tmp_path`. Assert the CSV exists and that a path under `results/` was refused.

`slow` suite, not required for the default checkpoint: call `experiments.noise_robustness_sweep.run_trial((4, 0.0, 0))` once. That exercises the real Rot full-batch class, the reseed to 42, and 300 Adam steps, and it does not write `results/`. The assertion is "it returns the eight keys `run_trial` already returns," plus a snapshot of the pattern array under that seed if the characterization test has recorded one. It is not an assertion against `noise_robustness_sweep.csv`, because that file is a 30-trial aggregate and this call is one trial. A later human review can compare a full cell. This design does not treat a passing `slow` test as archive equivalence.

### RNG characterization

One test, global state saved and restored:

1. `numpy.random.seed(0)` and `pennylane.numpy.random.seed(0)`.
2. Draw the noise-sweep pattern `choice` for `n_bits=4` (`m = max(1, int(0.3 * 4))`).
3. Draw the Hopfield Bernoulli mask at a fixed rate.
4. Call `pennylane.numpy.random.seed(42)` and draw `numpy.random.randn` of the parameter shape `QAM` uses.
5. Draw one more Bernoulli mask.
6. Store those arrays as expected fixtures beside the test.

That locks the reseed without training. A comment in the test states that `seed(42)` is the behaviour under test.

## H. Migration sequence

Full scope, files, acceptance criteria, tests, and checkpoint names are in `docs/migration_plan.md`. The order is fixed by dependency and risk:

| Stage | Scope | Depends on |
|---|---|---|
| 0 | These two design documents. | The audit and the baseline config already in the tree. |
| 1 | Characterization tests for recall, both corruption rules, metrics, and the noise-sweep reseed snapshot. No production edits. | Stage 0. |
| 2 | `benchmark/output.py` and the launcher overwrite flag. Legacy scientific modules untouched. | Stage 1, so path tests exist before any writer. |
| 3 | Read-only loader for `baseline/configs/original_experiments.json`, checked against module constants. No runs. | Stage 2's package skeleton if the loader lives in `benchmark/`. |
| 4 | Thin adapters that delegate to the symbols in section A. No training loops copied. | Stages 1 and 3. |
| 5 | Protocol runners for smoke cells only, writing under `runs/` or `tmp_path`. No experiment `main`. No `--full-grid` in CI. | Stages 2 and 4. |
| 6 | Manifests (package versions, `default.qubit`, procedure ids, seed formula, `comparable_to_archive: false`). | Stage 5. |
| 7 | Optional explicit RNG, only as a new protocol version, and only after a draw-sequence test matches stage 1 fixtures. Reference modules keep global seeding. | A human decision after stage 6. Not required for the framework to be usable. |

Each stage ends in its own git commit when the person doing the work is asked to commit. A stage that fails its tests is not checkpointed as complete. No stage's acceptance criterion is "the archived CSV was regenerated."
