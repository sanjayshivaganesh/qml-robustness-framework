# Inference migration

Date: 2026-10-10. The protocol is `experiments/noise_robustness_inference.py`. The framework request files are `configs/noise_robustness_inference.json` and `configs/noise_robustness_inference.smoke.json`. The runner is `benchmark.protocols.noise_robustness_inference`. `experiments.noise_robustness_inference.main` is unchanged and was not called.

## Protocol

Each task is `(n_bits, capacity_ratio, noise_p, trial_id)`. The runner calls `run_trial` for that tuple. It does not reimplement the draw, the mask, or either recall.

`run_trial` does the following:

1. Seeds NumPy with `42 + trial_id + int(n_bits * 1000) + int(capacity_ratio * 100) + int(noise_p * 1000)`. There is no PennyLane seed call.
2. Draws `m = max(1, int(n_bits * capacity_ratio))` patterns.
3. Draws one stored target with `randint`.
4. Applies the inline mask `numpy.random.rand(n_bits) < noise_p`, then multiplies those bits by `-1`. A level of `0` flips nothing.
5. Stores Hebbian weights and recalls with `hopfield_async_sequential_recall(..., steps=20)`. The return is the state, the final energy, and the energy drop. Energy is `E = -0.5 * x^T W x`.
6. Recalls with `nearest_neighbor_recall`. That function has no circuit and no training method.
7. Scores both predictions with `compute_metrics` and sets recovery gain to input Hamming error minus output Hamming error.

The two result identities are:

| Procedure id | Archive column `model` | Implementation |
|---|---|---|
| `hopfield_async_sequential` | `hopfield` | `src.hopfield.hopfield_async_sequential_recall` |
| `hamming_nearest` | `qam` | `src.qam.nearest_neighbor_recall` |

`qam` in this CSV is the Hamming label. It is not `qam_ry_linear`, `qam_rot_fullbatch`, `qam_rot_minibatch`, or `qam_paulix`. The registry rejects the id `"qam"`. Hamming has no constructor, so `prepare` does not build one. The framework rows keep `model` and add `procedure_id`.

The historical grid is 9 widths, 5 capacities, 16 noise levels from `numpy.linspace(0, 0.5, 16)`, and 300 trials: 216,000 tasks and 432,000 raw rows. `main` runs them with a `spawn` pool, `imap_unordered`, chunk size 50, and `min(12, cpu_count())` workers. This runner is single-process and keeps the module's task order. Each task seeds itself, so the recall metrics do not depend on that order. Wall-clock columns do.

`aggregate_results` then groups by model, width, capacity, and noise. It writes means and sample standard deviations, a 200-resample percentile bootstrap, and Gaussian half-widths `1.96 * std / sqrt(300)`. The bootstrap does not seed NumPy. Historical `main` draws it on the pool parent, which did not run the tasks. This runner draws it after the last in-process task. Those two stream positions differ. The runner does not add a seed.

The critical-noise tables at floors `0.50` and `0.90` are written by plot scripts. `main` does not write them, and this runner does not either.

## Schema differences

The raw columns from `run_trial` are unchanged in meaning. The framework adds `protocol`, `grid`, `comparable_to_archive`, and `procedure_id`. Hamming energy stays missing: the historical CSV leaves those cells blank, and the framework writes an empty cell (`None` in JSON).

The summary keeps the `aggregate_results` columns and adds the same identity fields plus `sample_size`. The Gaussian denominator remains the module constant 300. `gaussian_ci_sample_std` in `benchmark/scoring.py` divides by the length of the series it is given. The runner does not use that helper for these columns, because a shorter series would change the historical denominator.

`comparable_to_archive` is false on every framework row.

## Smoke and historical settings

| Setting | Historical module | Smoke request |
|---|---|---|
| Widths | `8, 12, 16, 24, 32, 48, 64, 96, 128` | `8` |
| Capacities | `0.1, 0.25, 0.5, 0.75, 1.0` | `0.1` |
| Noise | 16 levels, `linspace(0, 0.5, 16)` | the second level only |
| Trials | `0..299` | `0..299` |
| Tasks | 216,000 | 300 |
| Raw rows | 432,000 | 600 |
| Summary rows | 1,440 | 2 |
| Training | none | none |
| `executable` | false until `--full-grid` | true |

The omitted widths, capacities, and noise levels are demonstration settings. The trial count, seed formula, both recall methods, and the unseeded summary stay the module settings. `configs/noise_robustness_inference.json` loads the full grid and does not start it. The tests do not pass `--full-grid` into a real run.

## Validation performed

`tests/test_inference_migration.py` checks:

- The historical file matches the module and `baseline/configs/original_experiments.json`: 216,000 tasks, 432,000 raw rows, 300 trials, 20 Hopfield steps, and `model_labeled_qam.circuit` false.
- The registered callables are `hopfield_async_sequential_recall` and `nearest_neighbor_recall`. `get("qam")` fails. `prepare("hamming_nearest", {})` fails because there is no constructor.
- Pattern counts are `max(1, int(n * capacity))`: 1 at `n=8`, capacity `0.1`; 2 at capacity `0.25`; 128 at `n=128`, capacity `1`.
- Loading either file, and refusing the historical grid, does not call `main` or `run_trial`.
- The smoke cell was executed: 300 tasks, 600 rows, two summary groups. Every scientific field matched a direct `run_trial` loop. The summary means and percentile endpoints matched `aggregate_results` on that same in-process sequence. Runtimes were not compared across machines.
- Trial 12 of that cell flips one bit of eight (`input_hamming_error` `0.125`). Both the Hopfield row and the `model="qam"` row matched `results/noise_robustness_inference/raw/noise_inference_rows.csv` on the metric columns. The Hamming energy cells are blank in the archive and empty in the framework row. The other 215,999 tasks were not compared.
- The archive raw file is still 50,153,034 bytes. The summary still has 1,440 data rows. The raw file, the summary, both critical-noise CSVs, and `experiment.log` were hashed before and after the smoke run and did not change.
- A destination under `results/noise_robustness_inference` is refused before any trial.
- A request with 1 trial, `variational: true`, a Hamming CSV label other than `qam`, a variational procedure id, or a noise level outside the module list is rejected.
- `validate` prints the smoke plan as executable and the 216,000-task plan as not executable. The smoke file rejects `--full-grid`.

## Remaining limitations

The 216,000-task grid was not run. The archived summary's percentile columns were not recomputed, and a match would not be expected: the bootstrap is unseeded, and the historical draw happens on the pool parent. The smoke summary's percentiles match `aggregate_results` only for this single-process sequence. They are not the archived summary.

One matching task is not a reproduction of the 432,000-row file, the 1,440-row summary, or either critical-noise table. Runtime columns were not treated as scientific results. A recorded seed formula is not a claim that those percentiles or wall-clock numbers will match a later run.
