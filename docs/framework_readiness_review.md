# Framework readiness review

Date: 2026-10-10. This review follows the inference migration. It records what the suite and the small inference cell actually exercised. A passing test is not a reproduction of an archived experiment.

## What was run

The inference smoke in `configs/noise_robustness_inference.smoke.json` ran one cell: `n=8`, capacity `0.1`, the second module noise level, and trials `0..299`. That is 300 tasks. It wrote under a temporary directory through `begin_run` / `complete_run`. `comparable_to_archive` stayed false. The full suite then passed: 224 tests. The 216,000-task grid was not started, and `--full-grid` was not passed into a real run.

Loading `configs/noise_robustness_inference.json` and `configs/noise_robustness_inference.smoke.json` does not call `run_trial`. The same is already true for the noise-sweep, scaling, statistical-significance, storage-capacity, and circuit-depth loaders: validation parses the file and returns a config object.

## Registry

Registered procedure ids, and only these, are:

| Id | What it actually calls |
|---|---|
| `hopfield_synchronous` | `src.hopfield.Hopfield` |
| `hopfield_async_sequential` | `hopfield_async_sequential_recall` |
| `hopfield_async_random` | `HopfieldAsyncRandom` |
| `qam_ry_linear` | `src.qam.train_ry_linear_qam` |
| `qam_rot_fullbatch` | the noise-sweep Rot class |
| `qam_rot_minibatch` | the scaling Rot class |
| `qam_paulix` | `experiments.statistical_significance.build_qam` |
| `hamming_nearest` | `src.qam.nearest_neighbor_recall` |

`get("qam")` raises `UnknownModel`. The four variational ids are different callables. `hamming_nearest` has no constructor. Inference rows still store the archive label `model="qam"` beside `procedure_id` `hamming_nearest`, so the column is not renamed into a circuit.

## Result store

Each migrated runner writes with `benchmark.output.begin_run` and finishes with `complete_run` or `fail_run`:

- `benchmark/protocols/noise_robustness_sweep.py`
- `benchmark/protocols/scaling.py`
- `benchmark/protocols/statistical_significance.py`
- `benchmark/protocols/storage_capacity.py`
- `benchmark/protocols/circuit_depth.py`
- `benchmark/protocols/noise_robustness_inference.py`

A path that resolves to `results/` is refused before the trial loop. `overwrite` does not lift that refusal. Runs are not resumable. The inference smoke hashed the archived inference files before and after and they did not change.

## Experiment inventory

| Module | Classification | What the small case covers | What was not shown |
|---|---|---|---|
| `experiments/noise_robustness_sweep.py` | Migrated and tested on small cases | Smoke cell, reseed to 42, paired Hopfield and Rot full-batch | The 4×5×30 grid at 300 steps. The archived sweep CSV. |
| `experiments/scaling.py` | Migrated and tested on small cases | Width 4, one demonstration training step, depth `min(6, n)` | The 5×30×200-step grid. Archived scaling rows. |
| `experiments/statistical_significance.py` | Migrated and tested on small cases | One cell, `n=8`, capacity `0.2`, noise `0.1`, seed `0`, 100 epochs | The 720-task table. The archived summary p-value. The unseeded bootstrap on the pool parent. |
| `experiments/storage_capacity.py` | Migrated and tested on small cases | `n=8`, `m=1`, five seeds, 100 steps. The first archived sweep row's full-accuracy mean is `1.0`. | The 33-cell sweep. The archived critical ratio `0.375`. Wall-clock columns. |
| `experiments/circuit_depth.py` | Migrated and tested on small cases | `n=8`, `m=4`, depth `1`, five seeds. That cell's accuracy and parameter count match the first archived row. | The other 20 depth rows. Train times. |
| `experiments/noise_robustness_inference.py` | Migrated and tested on small cases | One cell, 300 trials. Trial 12, including a one-bit flip, matches the archived raw row for both models. | The other 215,999 tasks. Summary percentiles. Both critical-noise tables. |
| `experiments/convergence.py` | Not yet migrated | No runner and no request file | The 300-step loss curves |

No module is only partially wired. Convergence is the protocol with no runner. None of the six migrated modules is scientifically validated at its historical grid. The classification "migrated and tested on small cases" means a direct call on a reduced schedule agreed with the framework on that schedule. It does not mean the archived CSV was reproduced.

## Git diff of scientific files

`git diff --stat` against HEAD shows comment, docstring, and print-text edits under `src/`, `experiments/`, and `scripts/`. Numeric assignments that remain, including inference `N_BITS_LIST`, `CAPACITY_RATIOS`, `linspace(0, 0.5, 16)`, `RANDOM_SEED_BASE = 42`, and `TRIALS_PER_CONFIG = 300`, are the same values. This review did not find a changed recall, loss, corruption count, or seed formula in those hunks. Day 20 did not edit those files. `results/` has no content change. `baseline/configs/original_experiments.json` was not edited.

Two print strings differ from HEAD: storage capacity's completion line, and statistical significance's plot lines. Those prints are not metrics.

## Remaining work

Protocol still missing:

- `experiments/convergence.py`. Same `(n, m)` cases as circuit depth, seeds `0, 1, 2`, depth `min(8, n)`, 300 Adam steps, `train_ry_linear_qam_record_loss`. No corruption and no recall metric. No request file, no runner, no small-case comparison through the framework.

Full historical grids not executed, and not accepted by the tests:

- Noise sweep: 4 widths, 5 rates, 30 seeds, 300 steps.
- Scaling: 5 widths, 30 trials, 4 noise levels, 200 steps.
- Statistical significance: 720 tasks, 100 epochs, process pool.
- Storage capacity: 33 memory loads, 5 seeds, 100 steps.
- Circuit depth: 21 cells, 5 seeds, 100 steps.
- Inference: 216,000 tasks. The archived log line is `Progress: 216000/216000`. This review did not repeat that run.

RNG limitations that a recorded seed does not remove:

- One legacy `RandomState` is shared by `numpy.random` and `pennylane.numpy.random`. `default_rng` is a different family.
- The noise-sweep constructor reseeds to 42 inside `QAM.__init__`. The trial seed does not govern the draws after that call.
- Storage-capacity `bootstrap_ci` continues `numpy.random` with no separate seed. Statistical-significance `bootstrap_diff` draws with no seed, on the pool parent in `main` and after the last task in the single-process runner.
- Inference `aggregate_results` draws 200 unseeded resamples. The framework calls that function and does not seed it. The smoke percentiles match that call only for the in-process sequence. They were not compared with the archived summary.
- `scripts/plot_bootstrap_distribution.py` draws 5,000 unseeded resamples. The runner does not call it.
- Scaling and the inference module still own an inline `rand(n) < p` mask. The inference runner reaches that mask by calling `run_trial`, so the experiment file was not edited to use `corrupt_bernoulli`.
- Wall-clock columns are machine measurements. They were not used as equality checks.

Scientific checks that are still missing:

- Convergence loss curves against `train_ry_linear_qam_record_loss` on a small pattern.
- Any full-grid comparison with an archived CSV.
- Inference summary percentiles and both critical-noise tables (`0.50` last noise still at or above that floor, `0.90` first noise below that floor). The plot scripts that write those tables were not run.
- Storage-capacity critical ratios for `n=12` and `n=16`, and the archived phase table beyond the fact that the smoke cell does not cross `0.75`.
- Circuit-depth rows other than `n=8`, `m=4`, depth `1`.
- Statistical-significance summary p-values. The one-cell Wilcoxon result is not the archived summary.
- A process-pool runner. Adding one would change where an unseeded bootstrap reads the global stream. That has not been implemented or tested.

`python -m experiments.<name>` can still overwrite `results/`. The framework does not.
