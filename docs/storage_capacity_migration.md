# Storage-capacity migration

Date: 2026-10-10. The source module is `experiments/storage_capacity.py`. The framework requests are `configs/storage_capacity.json` and `configs/storage_capacity.smoke.json`. `main` was not rewritten and was not called.

The pair is `hopfield_synchronous` (`src.hopfield.Hopfield`) and `qam_ry_linear` (`src.qam.train_ry_linear_qam` and `src.qam.qam_recall`). PauliX, Rot full-batch, and Rot minibatch are different circuits and are not selected.

## Preserved protocol

Constants were read from `baseline/configs/original_experiments.json` under `experiments.storage_capacity` and from the module.

| Field | Value |
|---|---|
| Lengths | `8, 12, 16` |
| `m` at `n < 16` | `range(1, n + 1)` |
| `m` at `n >= 16` | `1..10, 12, 14, 16` when `reduced_scale_for_large_n` is true |
| Cells | `8 + 12 + 13 = 33` |
| Trials | Seeds `0..4`, reapplied with `numpy.random.seed` at every cell |
| Noise | Fixed-count, `p = 0.15`, flipping `max(1, int(n * 0.15))` bits |
| Hopfield | Synchronous, 10 steps. A zero becomes `+1`. |
| QAM | RY encoding `RY(x * pi)`, RY plus linear CNOT, depth `min(8, n)`, 100 steps, learning rate `0.05`, init scale `0.01`, `return_tuple=True` |
| Capacity ratio | `m / n_bits` |
| Critical ratio | First ratio, in ascending order, whose mean full accuracy is strictly below `0.75` |
| Interval | `bootstrap_ci`: 2000 resamples of the mean through `bootstrap_mean_ci`, confidence `0.95` |
| Wilcoxon | Hopfield scores, then QAM scores. An exception stores `None`. |
| QAM trainings | `33 × 5 = 165` |

`reduced_scale_for_large_n` is the module flag. With it true, `n=16` uses 13 memory loads rather than `1..16`. The framework does not turn that flag off, and it does not treat the flag as a smoke switch.

Each trial seeds NumPy, draws `±1` patterns, trains Hopfield, then scores every stored pattern with `corrupt_fixed_count(..., rng=numpy.random)`. The RY-linear trainer runs after that and draws `pennylane.numpy.random.randn` on the same legacy stream. A second fixed-count pass scores `qam_recall`. The module does not also call `pennylane.numpy.random.seed`.

Scores are `full_and_bitwise_accuracy`: the mean of per-pattern exact matches, and the mean of per-pattern bit fractions. Hopfield train time is an outer timer because `train` returns its own elapsed time and `main` does not use that return value. Hopfield inference time is the recall loop divided by `m`. QAM train time is the value returned by `train_ry_linear_qam`.

After the five trials, `bootstrap_ci` is called on Hopfield full accuracy, Hopfield bit accuracy, QAM full accuracy, QAM bit accuracy, QAM train times, and QAM inference times. It passes `numpy.random` and does not call `seed`. Those resamples continue the stream left by seed `4`. Hopfield time means are ordinary means. The QAM time columns store the bootstrap mean. Parameter count is `n_bits ** 2` for Hopfield and the mean of `params.size` for the QAM. Both model rows store the same Wilcoxon p-values. The QAM note is `Variational QAM | depth=<depth> | steps=100 | lr=0.05`.

`confidence_interval` in the storage-capacity module is a student-t helper. `main` does not call it, and the runner does not call it.

The phase table scans each model and width by `m / n_bits`. The significance table keeps one copy of the p-value columns per `(n_bits, m)`, which is what `drop_duplicates` keeps when Hopfield was written first.

`main` is a single process with a `tqdm` bar over `m`. The framework runner is single-process. It logs cell progress, each trial, model construction, training, evaluation, and one statistics line per cell. It does not log each Adam step or each bootstrap draw. Plotting stays in `main`.

## How a run starts

`configs/storage_capacity.json` is the 33-cell grid at 100 steps and 2000 bootstrap draws. It loads with `executable` false. `python -m benchmark validate` prints that plan and does not train. `python -m benchmark run` without `--full-grid` does not train. `--full-grid` is the switch that would start the 165 QAM trainings. The Day 18 tests do not pass that flag through to training. Output would go under `runs/storage_capacity/<run_id>/`. A path that resolves to `results/` is refused. `comparable_to_archive` is false. `resumable` is false.

`configs/storage_capacity.smoke.json` is one cell of that grid: `n=8`, `m=1`, seeds `0..4`, 100 steps, noise `0.15`, and 2000 bootstrap draws. At width 8 the module list is still `1..8`; this request scores `m=1` only. `setting_origin` labels the widths `demonstration` and the training constants `historical`. The manifest records `historical_cells` 33 and `smoke_cells` 1.

Loading or validating either file does not call `train_ry_linear_qam` or `main`.

## Validation performed

`m_values_for` matches the module branch, including the `n=16` list of 13 values and the `1..16` list that the flag leaves unused. Fixed-count widths are 1 bit at `n=8` and 2 bits at `n=16`. Critical-ratio scans on small arrays follow `critical_capacity_ratio`: a mean equal to `0.75` is not critical, and a series that never falls below `0.75` returns no ratio. `summarize_cell` matched `bootstrap_ci` on a five-value sample after the test seeded NumPy. The helper itself was not given a new seed. Wilcoxon failure for this protocol stays `None`, which is distinct from the statistical-significance sentinel `1.0`.

The smoke run matched a direct cell that calls `Hopfield`, `corrupt_fixed_count`, `train_ry_linear_qam`, `qam_recall`, `full_and_bitwise_accuracy`, and `bootstrap_ci` in the module order. Patterns, per-seed scores, trained parameters, aggregate means, and bootstrap endpoints matched. The NumPy state after the direct cell matched the state after the framework cell. The run log contains `trial_start`, `model_train`, `model_evaluate`, `statistics_start`, and `persist_complete`.

That cell's Hopfield full-accuracy mean matches the first data row of `results/storage_capacity/summary/storage_capacity_sweep.csv` (`n=8`, `m=1`, mean `1.0`). The smoke phase rows are null: one perfect mean does not cross `0.75`. The archived Hopfield ratio at `n=8` is `0.375`. That table was not recomputed. The sweep file has 66 data rows, the phase file has 6, and the significance file has 33. None of those files were modified.

One seed at `n=8`, `m=1`, and 100 steps took about 2 seconds on this machine. The smoke cell trains five such seeds, twice in the comparison test.

## Full-run prerequisites

Starting the historical grid requires `python -m benchmark run configs/storage_capacity.json --full-grid` and an output directory outside `results/`. That run is 33 cells, 5 seeds, and 100 RY-linear steps: 165 QAM trainings, plus the matching Hopfield trainings and 2000 bootstrap draws per accuracy series.

The archived log `results/storage_capacity/storage_capacity_sweep.log` records these tqdm totals:

| Sweep | Recorded total |
|---|---|
| `n=8` (8 values of `m`) | 6 min 26 s |
| `n=12` (12 values of `m`) | 29 min 50 s |
| `n=16` (13 values of `m`) | 2 h 41 min 54 s |

Those three lines sum to about 3 hours 18 minutes. This session did not rerun that sweep, and the 2-second seed does not estimate the `n=16` cells. The command `python -m experiments.storage_capacity` would overwrite the summary CSV, the phase-transition CSV, the Wilcoxon CSV, the summary text, and the plots under `results/storage_capacity/`.

## Remaining limitations

These points are not established:

- The 33-cell framework tables equal the archived CSVs. One cell was trained. The critical ratios in `phase_transition.csv` were not recomputed.
- Train-time or inference-time columns match the archive. Those values are wall-clock measurements.
- A recorded seed list reproduces the archived bootstrap intervals. The `RandomState` is not stored. The resamples are determined only after the five trials of that cell have left the stream where seed `4` stopped.
- The smoke p-value, or the smoke critical ratio of null, describes the archived phase transition. The smoke scans one ratio, `0.125`, whose mean stays above `0.75`.
