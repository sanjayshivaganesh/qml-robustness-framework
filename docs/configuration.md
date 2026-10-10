# Benchmark configuration schema

Date: 2026-10-10. This is the schema for one implemented protocol. `benchmark.config_loader.load_benchmark_config` reads a request file into a `BenchmarkConfig`. It does not train, does not build a circuit, and does not start a run. The command line that calls the loader and the runner is `python -m benchmark`, documented in `docs/cli_usage.md`. `baseline/configs/original_experiments.json` remains the audit record and is not rewritten.

The example request is `configs/noise_robustness_sweep.smoke.json`. The historical grid is `configs/noise_robustness_sweep.json`, described in `docs/noise_robustness_migration.md`. That file loads with `executable` false. `python -m benchmark run` does not start it unless `--full-grid` is passed, and this repository's tests do not pass that flag.

Scaling uses a separate request shape in `configs/scaling.json` and `configs/scaling.smoke.json`. Its models are `hopfield_async_random` and `qam_rot_minibatch`. The noise-sweep fields (`capacity_ratio`, `train_steps`, reseed 42) are not that protocol. `docs/scaling_migration.md` records the schedule and the smoke budget.

Statistical significance uses another request shape in `configs/statistical_significance.json` and `configs/statistical_significance.smoke.json`. Its models are `hopfield_synchronous` and `qam_paulix`. Depth 10, 100 epochs, and learning rate 0.02 stay closed over by `experiments.statistical_significance`. The paired bootstrap stays unseeded. `docs/statistical_significance_migration.md` records the cell schedule and the one-cell smoke.

Storage capacity uses `configs/storage_capacity.json` and `configs/storage_capacity.smoke.json`. Its models are `hopfield_synchronous` and `qam_ry_linear`. Memory load, the 0.75 critical ratio, five seeds, 100 steps, and the bootstrap that continues `numpy.random` stay the module settings. `docs/storage_capacity_migration.md` records the cell schedule and the one-cell smoke.

Circuit depth uses `configs/circuit_depth.json` and `configs/circuit_depth.smoke.json`. Its only model is `qam_ry_linear` with `return_tuple` false. Depth is the ansatz layer count `1, 2, 4, 6, 8, 10, 12`, not `min(8, n_bits)`. Recall is the inline sign map. `docs/depth_migration.md` records the cell schedule and the one-cell smoke.

Inference uses `configs/noise_robustness_inference.json` and `configs/noise_robustness_inference.smoke.json`. Its models are `hopfield_async_sequential` and `hamming_nearest`. The archive column `model="qam"` stays the Hamming label. There is no training step. `docs/inference_migration.md` records the task schedule and the one-cell smoke.

## Serialization

The project does not depend on a YAML parser. PyYAML is not listed in `pyproject.toml`. The example is JSON so a future loader can use the standard library. The field names below are the schema. They are not tied to JSON syntax beyond what an object, an array, a string, a number, and a boolean already express.

Formula strings are not part of the schema. Pattern counts and depth clamping are rule ids that name Python already in the tree. A loader must not `eval` a configuration string.

## What the example is

`document_role` is `benchmark_request`. The Day 8 example describes the paired noise-sweep smoke. `load_benchmark_config` accepts that file and `to_runner_request()` returns the `NoiseSweepRequest` the runner already accepts. The loader does not call `run_benchmark`.

`loader_status` stays `not_implemented`. The loader requires that string. It means the file is not a switch that runs the historical grid. It does not mean `load_benchmark_config` is absent.

Historical constants stay in `baseline/configs/original_experiments.json` under the key `noise_robustness_sweep`. The example only points at that file:

| Pointer field | Value | Role |
|---|---|---|
| `historical_reference.path` | `baseline/configs/original_experiments.json` | Audit file. Not loaded by the runner. |
| `historical_reference.experiment_key` | `noise_robustness_sweep` | Object inside that file. |
| `historical_reference.role` | `audit_record_only` | The only allowed role. |

`setting_origin` labels each scientific choice `historical` or `demonstration`. A `historical` label means the value is the one in the audit record or the module that record transcribes. A `demonstration` label means the value was chosen so the current smoke runner can execute it. Demonstration values are not the archived experiment.

| Choice | This request | Audit record | Origin |
|---|---|---|---|
| Protocol | `noise_robustness_sweep` | same module | historical pairing |
| Pattern generator | `numpy.random` | `np.random.choice` on the global stream | historical |
| Alphabet | `[-1, 1]` | `[-1, 1]` | historical |
| `n_bits_list` | `[2]` | `[4, 6, 8, 10]` | demonstration |
| Pattern-count rule | `max(1, int(capacity_ratio * n_bits))` | same formula | historical |
| `capacity_ratio` | `0.3` | `0.3` | historical |
| First model | `hopfield_synchronous`, 10 steps | `src.hopfield.Hopfield`, 10 steps | historical |
| Second model | `qam_rot_fullbatch` | `experiments.noise_robustness_sweep.QAM` | historical identity |
| Depth argument | `6`, then `min(6, n_bits)` | same | historical |
| Learning rate | `0.05` | `0.05` | historical |
| Constructor reseed | `42` | `QAM.__init__` calls `pennylane.numpy.random.seed(42)` | historical behaviour, recorded, not configured |
| `train_steps` | `1` | `300` | demonstration |
| Perturbation | `bernoulli` | `corrupt_bernoulli` | historical |
| Levels | `[0.0]` | `[0.0, 0.1, 0.2, 0.3, 0.4]` | demonstration |
| Trial seeds | `[0, 1]` | `0..29` (30 trials) | demonstration |
| Metrics | `batch_recall`, `gaussian_ci_population` | full-pattern accuracy, bit accuracy, `mean_std_ci` | historical quantities, framework ids |
| Timings | train seconds and recall seconds per bit | mean train time and mean inference time per bit | historical quantities, not metrics |
| Process model | `single_process` | `multiprocessing.Pool` in `main` | demonstration; the smoke runner has no pool |
| Grid | `smoke` | the full module grid | demonstration |
| Output | `runs/noise_robustness_sweep_smoke` | `results/noise_robustness_sweep/` | demonstration; the archive path is refused |
| `comparable_to_archive` | `false` | the CSV is the archive | required label |

At `n_bits = 2` and `capacity_ratio = 0.3`, the historical formula yields `max(1, int(0.6)) = 1` pattern. That consequence is computed by `n_patterns_for`. It is not stored as a pattern matrix in the file.

Cell count is `1 width × 1 level × 2 seeds = 2`, which is inside the runner cap of 4. The historical grid is `4 × 5 × 30 = 600` cells at 300 Adam steps. Putting those numbers in this file would not make them runnable. The smoke runner rejects `grid` other than `smoke`, `train_steps` outside `1..5`, and a cell count above 4.

## Executable today

`run_benchmark` accepts an executable `BenchmarkConfig`. It resolves `procedure_ids` through `benchmark.registry` and checks the factory arguments before any adapter constructor runs. A non-executable document raises `ConfigNotExecutable` and does not construct a model. The loader itself still does not call `run_benchmark`.

A caller can also pass the request the runner already accepts:

- `n_bits_list = (2,)`
- `corruption_rates = (0.0,)`
- `seeds = (0, 1)`
- `capacity_ratio = 0.3`
- `train_steps = 1`
- `grid = "smoke"`
- `output_dir = runs/noise_robustness_sweep_smoke` passed beside the request, or omitted for an in-memory result

`run_benchmark` then calls `hopfield_synchronous` and `qam_rot_fullbatch` in that order, Bernoulli corruption, `batch_recall`, and `gaussian_ci_population`. Those pieces exist. This document does not invoke them.

Fields the smoke cell does not treat as a free choice:

| Field | Why a mismatch matters |
|---|---|
| Model list, order, and both `procedure_id`s | `run_benchmark(BenchmarkConfig)` requires `hopfield_synchronous` then `qam_rot_fullbatch` and looks both up in the registry before construction. Another list is refused. The cell still builds those two adapters, in that order. |
| `hopfield_synchronous` `steps` | `build(n_bits)` omits `steps`, so the class default 10 is what runs. |
| `depth_argument`, `depth_used_rule_id`, `lr` | The smoke call is `QAM(n_bits, train_steps=...)`. Depth and learning rate stay the class defaults. |
| `constructor_reseeds`, `reseed_value`, `seed_policy_id` | The class resets the shared stream to 42. The file cannot turn that off. |
| `perturbation.procedure_id` and `when` | The runner always uses Bernoulli after each model trains. |
| `metrics` and `timings` | The runner always scores `batch_recall` and `gaussian_ci_population`, and always records the two timings. |
| `execution.max_cells`, `execution.max_train_steps` | The integers `4` and `5` live in the protocol module. The file does not raise them. |
| `execution.process_model` | There is no pool. |
| `output.overwrite_archived_results`, `if_destination_non_empty`, `files` | `benchmark/output.py` already refuses the archive and a non-empty run directory, and it writes those four names. |
| `comparable_to_archive` | The runner writes `false`. |

`output.destination` is the path a caller would pass as `output_dir`. Null, if a future loader allows it, matches today's in-memory call. The example names a directory under `runs/`, which is gitignored.

## Identifiers that exist

Procedure ids implemented in `benchmark.adapters`:

`hopfield_synchronous`, `hopfield_async_sequential`, `hopfield_async_random`, `qam_ry_linear`, `qam_rot_fullbatch`, `qam_rot_minibatch`, `qam_paulix`, `hamming_nearest`.

This request uses the first and the fifth. `qam` is not a procedure id. The archived noise-sweep columns that start with `qam_` refer to `qam_rot_fullbatch` at 300 steps and 30 trials, which this request does not run.

Perturbation ids implemented in `benchmark.data`: `bernoulli`, `fixed_count`. This protocol's request allows only `bernoulli`.

Metric ids this protocol may name, from `benchmark.scoring.PROTOCOL_METRICS`: `batch_recall`, `gaussian_ci_population`. Other ids in `METRICS` belong to other protocols. Naming one here is invalid even though the function exists.

## Field catalogue

Types are JSON types. "Invalid" means a future loader rejects the document before calling the runner.

### Document

| Field | Type | Allowed | Default | Scientific effect | Validation |
|---|---|---|---|---|---|
| `schema_version` | integer | `1` | none; required | none | Any other integer is an unknown schema. |
| `document_role` | string | `benchmark_request` | none; required | none | Distinguishes this file from the audit JSON. |
| `loader_status` | string | `not_implemented` | none; required | none | A loader, once it exists, removes this field in a later schema version rather than flipping it to run historical code. |
| `experiment.experiment_id` | string | `noise_robustness_sweep_smoke` for this example | none; required | none | Label of this request. It is not a module name. |
| `experiment.protocol` | string | `noise_robustness_sweep` | none; required | selects the paired procedure, the Bernoulli order, and the metric set | The only protocol `run_benchmark` accepts. Any other name is unsupported. |
| `grid` | string | `smoke` or `full` | none; required | `smoke` is the budget the runner starts; `full` names the historical grid and is not started | Any other string is a schema error. `full` loads with `executable` false. |
| `comparable_to_archive` | boolean | `false` | none; required | none; it does not alter scores | `true` is invalid. Archive equality has not been shown. |
| `declared_cell_count` | integer | product of the three list lengths | none; required | none if it matches; a mismatch means the file disagrees with itself | Must equal `len(n_bits_list) * len(levels) * len(trial_seeds)`. |
| `setting_origin` | object of strings | each value is `historical` or `demonstration` | none; required | none by itself | Every path listed in the example must be present. A `historical` value must still equal the audit record. |

Unknown top-level keys are invalid. There is no plot key and no worker-count key.

### Dataset

| Field | Type | Allowed | Default | Scientific effect | Validation |
|---|---|---|---|---|---|
| `dataset.pattern_generator` | string | `numpy.random` | none; required | which stream `choice` reads after the trial seed | `pennylane.numpy.random` is the scaling experiment's generator name, not this protocol. `default_rng` is invalid. |
| `dataset.alphabet` | array of integers | `[-1, 1]` | none; required | bipolar patterns | Any other alphabet is invalid. |
| `dataset.n_bits_list` | array of integers | each integer `>= 1`; the product of lengths must satisfy the cell cap | none; required | width of each pattern | Empty is invalid. The example's `[2]` is a demonstration. The audit list is not substituted automatically. |
| `dataset.n_patterns.rule_id` | string | `max_one_or_int_capacity_ratio_times_n_bits` | none; required | `m = max(1, int(capacity_ratio * n_bits))` | This is `n_patterns_for`. An unknown rule id is invalid. |
| `dataset.n_patterns.capacity_ratio` | number | `0.3` | none; required | the ratio inside that formula | The runner rejects any other ratio. `0.3` is the historical constant, not a new default. |

The dataset does not contain a pattern matrix. Patterns are drawn when a trial runs. A literal pattern block is an unknown key and is invalid.

### Models

`models` is an array of length 2. Position 0 is trained and recalled first. Position 1 is second. That order is the historical noise-sweep order: Hopfield Bernoulli draws happen before `QAM.__init__` reseeds the stream. Reversing the array is invalid for this protocol.

The two objects do not share one parameter schema.

**`hopfield_synchronous`**

| Field | Type | Allowed | Default | Scientific effect | Validation |
|---|---|---|---|---|---|
| `procedure_id` | string | `hopfield_synchronous` | none; required | selects `src.hopfield.Hopfield` | Must match `source`. |
| `source` | string | `src.hopfield.Hopfield` | none; required | the class that runs | A different source is invalid. |
| `parameters.steps` | integer | `10` | class default is 10; the runner does not pass `steps` | synchronous `sign(Wx)` updates; zeros become `+1` | Any other integer is invalid until a runner path actually passes `steps`. |
| `training.operation` | string | `train` | none; required | Hebbian weights, deterministic given the patterns | `store` belongs to `hopfield_async_sequential` and is invalid here. |
| `recall.operation` | string | `recall_timed` | none; required | recall, then wall time divided by `n_bits` | `recall` without the timing pair is a different adapter operation. This protocol records the pair. |
| `recall.returns` | array of strings | `["pattern", "wall_time_per_bit"]` | none; required | the second value is a timing, not a metric | A different list is invalid. |

Keys that belong to the variational model (`depth_argument`, `train_steps`, `lr`, `rng`) are invalid on this object. Hopfield construction does not reseed.

**`qam_rot_fullbatch`**

| Field | Type | Allowed | Default | Scientific effect | Validation |
|---|---|---|---|---|---|
| `procedure_id` | string | `qam_rot_fullbatch` | none; required | selects `experiments.noise_robustness_sweep.QAM` | `qam`, `qam_ry_linear`, `qam_rot_minibatch`, and `qam_paulix` are different procedures. |
| `source` | string | `experiments.noise_robustness_sweep.QAM` | none; required | the class that runs | Must match the procedure id. |
| `parameters.depth_argument` | integer | `6` | class default `6` | argument before clamping | The runner does not pass `depth`. A value other than 6 is invalid. |
| `parameters.depth_used_rule_id` | string | `min_depth_argument_and_n_bits` | the class sets `depth = min(depth, n_bits)` | for `n_bits = 2` the used depth is 2 | Unknown rule ids are invalid. |
| `parameters.train_steps` | integer | any integer `>= 1` is schema-valid; only `1` through `5` is executable | class default is 300 | Adam step count. `1` does not reproduce a 300-step model | `1.0` is rejected rather than coerced. `300` loads with `executable` false when `setting_origin` is `historical`. |
| `parameters.lr` | number | `0.05` | class default `0.05` | Adam step size | The runner does not pass `lr`. Any other number is invalid. |
| `rng.family` | string | `legacy_global` | none; required | `numpy.random.seed` and `pennylane.numpy.random.seed` share one stream | `default_rng` is invalid. |
| `rng.constructor_reseeds` | boolean | `true` | the class always reseeds | after Hopfield queries, parameter init does not continue the trial seed | `false` would describe a different experiment. |
| `rng.reseed_value` | integer | `42` | hardcoded in `QAM.__init__` | the reset value | Any other integer is invalid. This field does not add a new seed call. |
| `rng.seed_policy_id` | string | `legacy_global_trial_seed_then_qam_constructor_reseed_42` | the protocol's existing order | names the order already implemented | Any other id is invalid. |
| `training.operation` | string | `train` | none; required | the class `train` method | The loss and optimizer stay in the class. They are not configuration fields. |
| `recall.operation` | string | `recall` | none; required | `QAM.recall`, which returns the timed pair | Hopfield's `recall_timed` name is not this method's name. |
| `recall.returns` | array of strings | `["pattern", "wall_time_per_bit"]` | none; required | same timing divisor as Hopfield, `n_bits` | A different list is invalid. |

`steps` is invalid on this object. It is the Hopfield update count, not an Adam step count.

The same `seed_policy_id` is repeated under `seeds` so the trial-seed half and the constructor-reset half stay one policy. The two copies must be equal.

### Perturbation

| Field | Type | Allowed | Default | Scientific effect | Validation |
|---|---|---|---|---|---|
| `perturbation.procedure_id` | string | `bernoulli` | none; required | each bit flips independently with probability equal to the level; level `0` flips nothing | `fixed_count` is a different rule, including a flip at level `0`. It is invalid here. |
| `perturbation.levels` | array of numbers | each level in `[0, 1]`; the cell product must fit the cap | none; required | the Bernoulli probability | Empty is invalid. The example's `[0.0]` is a demonstration. Rate `0` still draws a mask. |
| `perturbation.when` | string | `each_stored_pattern_after_that_model_trains` | none; required | Hopfield masks use the trial seed; QAM masks use the stream after the reseed to 42 | One shared mask for both models would be a different experiment and is invalid. |

### Metrics and timings

| Field | Type | Allowed | Default | Scientific effect | Validation |
|---|---|---|---|---|---|
| `metrics` | array of strings | `batch_recall` and `gaussian_ci_population`, both present | none; required | full-pattern accuracy is the mean of exact matches; bit accuracy is the mean of per-pattern match fractions; the interval is mean, population standard deviation, and `1.96 * std / sqrt(n_trials)` | A missing id, an extra id, a runtime name, or an id reserved in `NOT_SELECTED` is invalid. |
| `timings` | array of strings | `train_time_seconds` and `mean_recall_time_per_bit`, both present | none; required | recorded durations | These names are not metric ids. Placing them in `metrics` is invalid. They are not acceptance criteria. |

`batch_recall` is the framework name for the noise sweep's full and bit accuracies. On equal-length patterns it is the same ratio as a pooled bit count and can differ by one floating-point unit. The experiment module still uses the pooled count. This file does not change that.

### Seeds

| Field | Type | Allowed | Default | Scientific effect | Validation |
|---|---|---|---|---|---|
| `seeds.trial_seeds` | array of integers | non-empty, distinct integers | none; required | both global seed functions are set to this integer at the start of the trial, before the pattern draw | Duplicates are invalid. The example `[0, 1]` is a demonstration, not `0..29`. |
| `seeds.trial_count` | integer | length of `trial_seeds` | none; required | number of trials averaged by `gaussian_ci_population` | Must equal that length. There is no separate `range` expression. |
| `seeds.seed_policy_id` | string | the same id as on the QAM model | none; required | documents the existing stream order | Must equal `models[1].rng.seed_policy_id`. |

The trial seed does not keep Hopfield and QAM on the same queries. That is the recorded class behaviour. The schema does not add a generator argument, a `Generator` seed, or a second reseed.

### Execution limits

| Field | Type | Allowed | Default | Scientific effect | Validation |
|---|---|---|---|---|---|
| `execution.process_model` | string | `single_process` | none; required | none inside an allowed cell; a pool would change wall-clock measurements | `pool` is invalid. The historical `main` is not selected by this field. |
| `execution.max_cells` | integer | `4` | the module constant `MAX_SMOKE_CELLS` | refuses a request larger than the smoke | Must equal the module constant. Editing the JSON does not raise the cap. |
| `execution.max_train_steps` | integer | `5` | the module constant `MAX_SMOKE_TRAIN_STEPS` | the upper bound on `train_steps` | Must equal the module constant. |
| `execution.caps_are_authoritative_in` | string | `benchmark.protocols.noise_robustness_sweep` | none; required | none | Names the module a future check reads. It is not an import hook. |

`declared_cell_count` must equal the product of the three list lengths. A product above 4, or equal to the historical 600, loads with `executable` false. `execution.max_cells` must stay 4; the file cannot raise the cap.

### Output

| Field | Type | Allowed | Default | Scientific effect | Validation |
|---|---|---|---|---|---|
| `output.destination` | string or null | a path that does not resolve to `results/`, to a parent of `results/`, or to a child of `results/` | null means in-memory; the example sets a `runs/` path | none | `results/` and `results/noise_robustness_sweep` are invalid, including via a symlink. The runner's archive check is the authority. |
| `output.overwrite_archived_results` | boolean | `false` | none; required | none | `true` is invalid. There is no flag on this runner that writes the archive. |
| `output.if_destination_non_empty` | string | `refuse` | `create_run_dir` refuses a non-empty directory and reuses an empty one | none | `overwrite` is invalid. |
| `output.files` | array of strings | `manifest.json`, `trials.csv`, `aggregates.csv`, `trials.json` | the runner writes these and also `configuration.json` | none | A different list in the file is invalid. Plot filenames are invalid. |

## Loader contract

`load_benchmark_config(path)` returns a `BenchmarkConfig` or raises `ConfigValidationError`. Relative output paths are resolved from the repository root. The directory is not created.

Schema failures, raised before a configuration object exists:

- Malformed JSON, a missing required field, an unknown field, or a wrong JSON type. `1.0` is not an integer and `true` is not a number.
- `experiment.protocol` outside the protocols named in `PROTOCOL_METRICS`. A known protocol other than `noise_robustness_sweep` has no request schema yet.
- `procedure_id` outside the eight adapters, including `qam`. A registered id in the wrong position for this protocol.
- Hopfield parameters other than `steps`, or QAM parameters that include `steps`.
- `fixed_count` or any unregistered perturbation. A Bernoulli level outside `[0, 1]`.
- `capacity_ratio` other than `0.3`, or a pattern-count rule id other than `max_one_or_int_capacity_ratio_times_n_bits`. Rule strings are not evaluated.
- A metric id from another protocol, a runtime name (`infer_time`, `energy`, `energy_drop`), or an id in `NOT_SELECTED`. The noise-sweep list must be exactly `batch_recall` and `gaussian_ci_population`.
- `trial_count` different from `len(trial_seeds)`, or duplicate seeds.
- `seed_policy_id` other than `legacy_global_trial_seed_then_qam_constructor_reseed_42`, or `reseed_value` other than `42`.
- `comparable_to_archive` true, `overwrite_archived_results` true, `if_destination_non_empty` other than `refuse`, or a destination that resolves to `results/`, a parent of it, or a child of it. A symlink into the archive is the same refusal.
- `setting_origin` missing a required path, or labelling a value `historical` when it differs from the audit record. `train_steps` `300` must be labelled `historical`. `train_steps` `1` must be labelled `demonstration`.

Not executable, returned as a `BenchmarkConfig` with `executable` false:

- `grid` is `full`.
- `train_steps` is an integer outside `1..5`, including the audit value `300`.
- The cell product is above `4`, or equal to the historical `4 × 5 × 30` grid.

`to_runner_request()` raises `ConfigNotExecutable` in those cases and does not call the runner. `execution.max_cells` and `execution.max_train_steps` must still equal the module constants `4` and `5`.

## Other protocols

Convergence is not given a request file in this schema. Inference, circuit depth, storage capacity, scaling, and statistical significance each have their own document, listed above. Those documents do not reuse the Rot full-batch parameter block.
