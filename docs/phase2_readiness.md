# Phase 2 readiness

Date: 2026-10-10. The registry is `benchmark/registry.py`. A validated noise-sweep file can name `hopfield_synchronous` and `qam_rot_fullbatch`, and `run_benchmark` resolves those ids before any adapter factory runs. This note does not claim that a smoke row matches an archived CSV. The evidence for the checks below is in `docs/phase2_validation.md`.

The Day 10 audit re-ran `MPLCONFIGDIR="$PWD/.mplconfig" ./venv/bin/python -m pytest -q`: 142 passed. The end-to-end test loads `configs/noise_robustness_sweep.smoke.json`, writes under a temporary directory, and records `hopfield_synchronous` and `qam_rot_fullbatch`. `git status -- results` was empty. `runs/` was not created. No experiment `main` ran.

## What the framework can execute

`load_benchmark_config` reads the smoke schema into a `BenchmarkConfig`. It still does not build a circuit. `run_benchmark` accepts that object when `executable` is true. It looks up `procedure_ids` in the registry, checks factory arguments, then runs the existing smoke cell. A document with `grid: full` or `train_steps: 300` raises `ConfigNotExecutable` and does not construct `QAM`.

`python -m benchmark` validates that file or runs it. `validate` and `run --dry-run` do not train. Commands and exit codes are in `docs/cli_usage.md`. Stage logging and failure status are in `docs/execution_and_failure_handling.md`. A failed or interrupted run is not resumed.

The executable smoke is the Day 8 example: `n_bits = 2`, Bernoulli level `0.0`, seeds `0` and `1`, `capacity_ratio = 0.3`, one Adam step, `grid = "smoke"`. That is two cells. The cell order is unchanged: both global seed functions take the trial seed, patterns use `m = max(1, int(0.3 * n))`, synchronous Hopfield trains and recalls, then Rot full-batch `QAM(n_bits, train_steps=...)` reseeds to 42. Depth 6 and learning rate 0.05 are checked and not forwarded, so the call stays the class defaults. An explicit `output_dir` overrides the file's `runs/` destination. The archive guard still refuses `results/`.

Eight procedure ids are registered, and only those eight. Each points at the adapter that already exists:

| `procedure_id` | Factory the registry returns | Object stored on the registration |
|---|---|---|
| `hopfield_synchronous` | `build` | `src.hopfield.Hopfield` |
| `hopfield_async_sequential` | `store` (`hopfield_hebbian_weights`) | `hopfield_async_sequential_recall` |
| `hopfield_async_random` | `build` | `HopfieldAsyncRandom` |
| `qam_ry_linear` | `build_circuit` | `train_ry_linear_qam` |
| `qam_rot_fullbatch` | `build` | `experiments.noise_robustness_sweep.QAM` |
| `qam_rot_minibatch` | `build` | `experiments.scaling.QAM` |
| `qam_paulix` | `build` (`build_qam`) | `build_qam`; recall is that module's `qam_recall`, not `src.qam.qam_recall` |
| `hamming_nearest` | none; `recall(patterns, query)` | `nearest_neighbor_recall` |

`prepare` returns the factory and its arguments. It does not call the factory. Unknown ids, including `"qam"`, fail in `get` before a constructor argument is interpreted. Hamming has no constructor. PauliX rejects `epochs` and `lr` as constructor arguments because those values stay the module constants. Minibatch requires an explicit `depth`, `steps`, and `lr` so the class default depth 3 is not filled in. The two Rot classes remain different objects.

Metrics on the smoke path are still `batch_recall` and `gaussian_ci_population`. Timings are recorded and are not metrics.

## What remains unsupported

The registry listing a procedure does not mean a protocol runner will train it.

| Gap | Why it is not executable from a config file |
|---|---|
| Convergence | No protocol module and no schema. Its adapter can be prepared, and nothing dispatches it. |
| Historical noise sweep, default command | `configs/noise_robustness_sweep.json` loads the module grid and stays non-executable until `--full-grid`. The smoke cap stays 4 cells and 5 steps. The flag was not used in the Day 15 tests. |
| Circuit-depth recall | The depth runner signs the `return_tuple=False` circuit in `sign_circuit_output`. That helper is not `qam_recall`, and it is not a ninth model. A fixed vector `[0, 0.2, -0.3]` maps the same way under both functions. The historical file still inlines the sign. |
| Rot minibatch, historical grid | The scaling class is registered and `configs/scaling.json` names it. The 200-step, 150-training grid stays behind `--full-grid` and was not run. The smoke budget is one step at width 4. See `docs/scaling_migration.md`. |
| Statistical significance, historical grid | `configs/statistical_significance.json` names synchronous Hopfield and `qam_paulix`. The 720-task, 100-epoch grid stays behind `--full-grid` and was not run. The smoke is one module cell. See `docs/statistical_significance_migration.md`. |
| Storage capacity, historical grid | `configs/storage_capacity.json` names synchronous Hopfield and `qam_ry_linear`. The 33-cell, 100-step grid stays behind `--full-grid` and was not run. The smoke is `n=8`, `m=1`, five seeds. See `docs/storage_capacity_migration.md`. |
| Circuit depth, historical grid | `configs/circuit_depth.json` names `qam_ry_linear` with `return_tuple` false. The 21-cell, 100-step grid stays behind `--full-grid` and was not run. The smoke is `n=8`, `m=4`, depth `1`, five seeds. See `docs/depth_migration.md`. |
| Inference, historical grid | `configs/noise_robustness_inference.json` names sequential Hopfield and `hamming_nearest`. The CSV label for Hamming remains `qam`. The 216,000-task grid stays behind `--full-grid` and was not run. The smoke is one cell at 300 trials. See `docs/inference_migration.md`. |
| Sequential Hopfield and Hamming nearest neighbour | Registered, and the inference request selects them. The noise-sweep smoke is still not allowed to select them. The inference CSV label `model="qam"` remains Hamming, and the result row also stores `procedure_id` `hamming_nearest`. |
| Inline Bernoulli in scaling and inference | Left in those modules so their RNG order stays put. |
| Unseeded inference bootstrap | The inference runner calls `aggregate_results` and does not seed it. That call is not a reproducible interval. The 216,000-task summary was not rebuilt. |
| Process pools | The smoke runner is single-process. Historical `main` functions still own their pools. |
| Plotting, statistical reporting | Not implemented. |
| Explicit `RandomState` migration, launcher overwrite flag | Not implemented. `python -m experiments.<name>` can still overwrite `results/`. The framework manifest records the run id, versions, git commit, `execution_status`, and the noise-sweep RNG record in `docs/randomness_and_reproducibility.md`. |
| Archived CSV equality | Not shown for any full table. Smoke rows set `comparable_to_archive` false. One circuit-depth cell and one inference task were compared with archived rows; those notes are in the migration documents and are not table-wide equality. |

## What must be validated before another protocol moves

Do this for that protocol on a small controlled input before adding a schema or a runner:

1. Call the adapter and the original function on the same inputs and compare the returned object. Do not change a circuit, loss, recall update, or corruption rule to make them match.
2. Write down the RNG order, including draws that happen inside a constructor and Bernoulli masks that are still inline. A shared integer seed is not a shared query stream after Rot full-batch reseeds to 42. `default_rng` is a different family from the legacy global stream.
3. Give the new schema that protocol's constructor arguments. Do not reuse the noise-sweep block (`depth_argument` 6, `train_steps`, reseed 42) for minibatch, PauliX, or RY-linear.
4. Refuse the historical grid with an explicit smoke cap. Keep output outside `results/`. Do not call the experiment `main`.
5. Compare a small run with the original loop only after steps 1–4. Archive bytes stay a separate check, and a match has to be measured before anyone describes the run as the archived experiment.

Rot minibatch needs the extra check that training still evaluates the loss twice inside `opt.step`. PauliX needs the check that depth 10, 100 epochs, and learning rate 0.02 stay closed over by the original functions. The circuit-depth sign check is the fixed vector in `docs/depth_migration.md`: the runner uses `sign_circuit_output` on the `return_tuple=False` circuit.

## Diff review

Day 10 adds `benchmark/registry.py`, `tests/test_registry.py`, and this note. It stores `procedure_ids` on `BenchmarkConfig` and routes the smoke cell's two constructors through `prepare`. Training and recall stay the adapter methods.

The Day 10 audit inspected the diff again. Tracked edits under `src/hopfield.py`, `src/qam.py`, `src/data_generation.py`, and the historical experiment modules remain comments, docstrings, and print text. `src/metrics.py` has no diff. Assignments such as `NOISE_LEVELS`, `RANDOM_SEED_BASE`, `TRIALS_PER_CONFIG`, `n_bits_list`, `trials_per_m`, and `recall_noise` keep the same values. The `pnp.random.seed(42)` call and `n_flip = max(1, int(len(pattern) * noise_level))` are not in the diff. `results/` has no content change. `baseline/configs/original_experiments.json` was not edited.

## Audit verdict

The registry, loader, and noise-sweep smoke are a usable base for execution-layer work: more protocol runners, still outside `results/`, still without historical `main`. No integration defect turned up in the audit. That base does not yet run the other six protocols, the 300-step noise sweep, or an archive comparison. `docs/phase2_validation.md` separates what was executed from what is only inferred.
