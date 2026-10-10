# Statistical-significance migration

Date: 2026-10-10. The framework requests are `configs/statistical_significance.json` and `configs/statistical_significance.smoke.json`. The scientific reference remains `experiments/statistical_significance.py`. That file was not rewritten, and its `main` was not called.

The pair is `hopfield_synchronous` (`src.hopfield.Hopfield`) and `qam_paulix` (`build_qam`, `train_qam`, and the module-local `qam_recall`). `qam_ry_linear`, `qam_rot_fullbatch`, and `qam_rot_minibatch` are different circuits and are not selected. The module-local `qam_recall` is not `src.qam.qam_recall`.

## Preserved protocol

Constants were read from `baseline/configs/original_experiments.json` under `experiments.statistical_significance` and from the module. The audit records seeds as `range(30)`. The module list is `0 .. 29`.

| Field | Value |
|---|---|
| Lengths | `8, 12, 16` |
| Capacity ratios | `0.2, 0.3, 0.4, 0.5` |
| Bernoulli noise | `0.1, 0.2` |
| Seeds | `0 .. 29`, reused at every cell |
| Stored patterns | `m = max(1, int(capacity_ratio * n_bits))` |
| Tasks | `3 × 4 × 2 × 30 = 720`, in that nested order |
| Hopfield | Synchronous, 10 steps. A zero becomes `+1`. |
| QAM | PauliX where the bit is `-1`, RY plus linear CNOT, depth 10, 100 epochs, learning rate `0.02`, init `normal(0, 0.1)` |
| Comparison groups | `n_bits`, `capacity_ratio`, `noise` |
| Paired difference | QAM column minus Hopfield column |
| Bootstrap | 5000 resamples inside `bootstrap_diff`, then percentiles `2.5` and `97.5` |
| Wilcoxon | `wilcoxon(qam, hopfield)`. An exception stores `1.0`. |
| Alpha cutoff | None. `main` stores the p-value. |
| Effect size | Cliff's delta with Hopfield as `x` and the QAM column as `y`. `main` does not compute it. |

Each task calls `numpy.random.seed` and `pennylane.numpy.random.seed` with the task seed, then draws patterns with `numpy.random.choice`. Hopfield trains and scores every stored pattern through `corrupt_bernoulli`. The PauliX circuit is built and trained after that, and a second Bernoulli pass scores it. `train_qam` draws `numpy.random.normal(0, 0.1)` and does not reseed. Depth, epoch count, and learning rate are the module constants those functions close over. A request that changes the epoch count is rejected, because the function would still run 100 epochs.

Noise is part of the task. The same seed at a second noise level is another training. The two noise levels are not two scores of one shared parameter vector. The task list was checked: task 0 is `(8, 0.2, 0.1, 0)` and task 30 is `(8, 0.2, 0.2, 0)`. That second cell was not trained.

Per-pattern scores are `pattern_accuracy_pair`. The trial record stores the mean of those full-match indicators and the mean of those bit fractions, which is what `run_trial` writes.

After the tasks, `summarize_groups` calls `paired_bootstrap_diff`, which is `bootstrap_diff` with no seed argument added. Full-accuracy resampling runs before bit-accuracy resampling. The percentile call is `numpy.percentile(diffs, [2.5, 97.5])`. The p-value comes from `wilcoxon_pvalue` with failure value `1.0`. Argument order is QAM then Hopfield.

Cliff's delta is a separate table. It uses `cliffs_hopfield_vs_qam`, which calls `src.metrics.cliffs_delta` and `cliffs_delta_magnitude`. Positive delta means Hopfield scores tend to be larger. The magnitude bands are the ones in that helper: `0.147`, `0.33`, and `0.474`. `scripts/cliffs_delta.py` rounds the delta to three decimals and writes under `results/`. The framework stores the unrounded value in the run directory and does not call that script.

`main` plots two error-bar figures. The framework manifest records `plotting` false.

`main` maps tasks with `Pool(cpu_count())`. The framework runner is single-process. `bootstrap_diff` in `main` therefore runs on the parent process, after the workers have finished, and does not see the worker streams. The framework call continues from the last task on the same process. The manifest records both process models. It sets `bootstrap_seeded` false, `confidence_intervals_reproducible` false, and `p_values_reproducible` false.

Wilcoxon itself does not draw. A p-value is a function of the score columns that were passed to it. The smoke group has one score per column, so SciPy raises and the stored value is the failure sentinel `1.0`. That sentinel is not a 30-seed result.

## How a run starts

`configs/statistical_significance.json` is the module grid: 720 tasks, 100 epochs, 5000 unseeded bootstrap draws, 24 comparison groups. It loads with `executable` false. `python -m benchmark run configs/statistical_significance.json` does not train. `--full-grid` is the switch that would start that grid. The Day 17 tests do not pass the flag through to training. Output would go under `runs/statistical_significance/<run_id>/`. A path that resolves to `results/` is refused. `comparable_to_archive` is false. `resumable` is false.

`configs/statistical_significance.smoke.json` is one module cell: `n=8`, capacity `0.2`, noise `0.1`, seed `0`. At that cell `m = 1`. Depth, epochs, learning rate, initialisation, and the bootstrap count stay the module values. `setting_origin` labels the reduced axes `demonstration` and the closed-over training constants `historical`. The manifest records `historical_tasks` 720 and `smoke_tasks` 1, and `historical_sample_size` 30 beside `sample_size` 1.

Trial rows keep `n_bits`, `capacity_ratio`, `noise`, `seed`, `hopfield_full`, `hopfield_bit`, `qam_full`, and `qam_bit`. They also record `hopfield_procedure_id` and `qam_procedure_id`, so the QAM column is `qam_paulix`. Summary rows keep the historical mean, difference, percentile, and p-value fields, and add the sample size and `bootstrap_seeded`.

## Validation performed

On controlled rows that are not training output, `summarize_groups` matched `bootstrap_diff`, `numpy.percentile`, `wilcoxon`, and `cliffs_delta` for the same arrays and the same resample count. The test seeded NumPy before each call. The helper itself has no seed call. Full-accuracy resampling was checked to run before bit-accuracy resampling. Group order follows the sorted keys `n_bits`, `capacity_ratio`, `noise`. Means of `pattern_accuracy_pair` matched the direct mean of those pairs. Those rows are not a scientific conclusion.

The one-cell framework run matched `run_trial(8, 0.2, 0.1, 0)` on the four mean scores, and the stored patterns matched a fresh `numpy.random.choice` after the same two seed calls. That cell's four scores also match the first data row of `results/statistical_significance/raw/statistical_rows.csv` (`seed` 0, all four values `1.0`). The archived file has a header and 720 data rows. The summary file has a header and 24 data rows. Neither file was modified.

The smoke summary has `sample_size` 1. Its full-accuracy p-value is `1.0`, the Wilcoxon failure sentinel. The first archived summary row (`n=8`, capacity `0.2`, noise `0.1`) stores `7.74421643104407e-06`, from 30 scores. That archived interval and p-value were not recomputed. The smoke effect-size row is one observation and is not the Cliff's delta table described in `docs/baseline_experiments.md`.

The historical file selects `experiments.statistical_significance.build_qam`. Factory preparation for PauliX accepts `n_bits` and rejects `epochs`. The 720-task request was refused before `train_qam` or `main`.

## Remaining limitations

These points are not established:

- A 720-task framework table equals `statistical_rows.csv` or `statistical_summary.csv`. One cell was trained. The other 719 were not.
- The smoke percentile interval equals the archived interval. The smoke group has one tied pair, so every resample of the difference is zero. The archived interval was computed from 30 scores. The bootstrap was left unseeded, and the framework call does not see the historical pool parent's stream.
- A recorded task seed reproduces the archived summary. The `RandomState` is not stored. Early cells in a full single-process run would also leave the stream where the next unseeded bootstrap starts, which is not where `main` starts it.
- The one-cell p-value of `1.0` says anything about the archived Wilcoxon result. It is the exception path for a sample of size 1.
- Cliff's delta on that one cell equals `scripts/cliffs_delta.py` output. `main` does not compute the effect size, and the script's archived table was not rewritten or compared.
- Wall-clock equality with a pool run. The framework does not start a pool.
