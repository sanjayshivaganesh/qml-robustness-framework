# Noise-robustness sweep migration

Date: 2026-10-10. The framework request for the paired Hopfield and Rot full-batch experiment is `configs/noise_robustness_sweep.json`. The historical implementation remains `experiments/noise_robustness_sweep.py`. That file was not rewritten, and its `main` was not called.

The request uses `hopfield_synchronous` and `qam_rot_fullbatch`. The second id is `experiments.noise_robustness_sweep.QAM`: RX and RY encoding, `Rot` plus linear CNOT, loss equal to the mean over patterns of the sum of squared Pauli-Z errors. `qam_ry_linear` is a different circuit and is not selected.

## Historical constants

These values were read from `baseline/configs/original_experiments.json` under `experiments.noise_robustness_sweep` and checked against the module attributes.

| Field | Value |
|---|---|
| `n_bits_list` | `4, 6, 8, 10` |
| `corruption_rates` | `0.0, 0.1, 0.2, 0.3, 0.4` |
| Trials and seeds | `30`, seeds `0..29` |
| `capacity_ratio` | `0.3`, so `m = max(1, int(0.3 * n_bits))` |
| Hopfield | `src.hopfield.Hopfield`, 10 synchronous steps |
| QAM depth argument | `6`, then `min(6, n_bits)` inside the class |
| `train_steps` | `300` |
| Learning rate | `0.05` |
| Constructor reseed | `pennylane.numpy.random.seed(42)` inside `QAM.__init__`, then `numpy.random.randn` |
| Cell count | `4 × 5 × 30 = 600` |

The cell order is the module order: width, then corruption rate, then seed. Each cell sets both global seed functions, draws patterns, trains Hopfield, Bernoulli-corrupts every stored pattern and recalls, then constructs QAM. That constructor reseeds to 42, so the QAM queries are not the Hopfield queries. Full-batch Adam does not draw a minibatch. The Gaussian interval is `mean_std_ci` and does not draw a bootstrap sample.

`main` runs those cells in a `multiprocessing.Pool`. The framework runner stays single-process. Each trial still seeds itself before the pattern draw, which is the interaction the manifest records. Pool scheduling was not compared. Wall-clock columns are not an acceptance criterion.

Plotting stays in `main` and in `scripts/plot_noise_robustness_sweep.py`. The framework runner does not import a plotting call, and the manifest field `plotting` is false.

## How a run starts

`load_benchmark_config` of the historical file returns `executable` false. `python -m benchmark run configs/noise_robustness_sweep.json` prints the plan and exits 1. It does not construct `QAM`.

`python -m benchmark run configs/noise_robustness_sweep.json --full-grid` is the execution path. It calls `run_benchmark(..., allow_historical_grid=True)`, which prepares the two adapters and then uses the same cell function as the smoke runner. Output goes under `runs/noise_robustness_sweep/<run_id>/` unless `--output-dir` selects another parent. A path that resolves to `results/` is refused before training. `comparable_to_archive` is false. `resumable` is false.

`--full-grid` on the smoke file is refused. The flag does not raise the smoke cap for an arbitrary request. The Day 15 tests do not pass it through to `_execute`, so the 600-cell grid was not run.

The manifest `rng` record lists the configured trial seeds, the two seed calls at the start of each trial, and the internal reset to 42 with `configured` false. `exact_reproducibility_claimed` is false. Recording those seeds does not make a later run a copy of the archived CSV.

## What was compared

On `n_bits = 4`, corruption `0.0`, seed `0`, and `train_steps = 1`, the framework cell and a direct call in the historical order produced the same patterns, the same queries, the same initial parameters, and the same recalled patterns. Depth used was 4 and the learning rate was 0.05. The constructor reseed recorded on the QAM arm was 42. This is one Adam step, not the historical 300.

`batch_recall` matched the historical ratios `full_matches / m` and `matching_bits / (m * n_bits)` on a fixed two-pattern batch. `gaussian_ci_population` matched `mean_std_ci` on a fixed list of scores, including the factor `1.96 * population_std / sqrt(n)`.

The archived summary `results/noise_robustness_sweep/summary/noise_robustness_sweep.csv` has a header and 20 data rows. Its columns include `n_bits`, `corruption`, the Hopfield and QAM full and bit means, standard deviations, CI half-widths, and the four timing means. Those means were not compared with a framework run. `results/noise_robustness_sweep/experiment.log` is a failed path error, not a log of that CSV. Per-trial rows, stored patterns, and trained parameters are not in the archive. Plot files named in the audit record were not treated as a comparison target.

## What the full experiment would still have to show

These points are not established:

- A framework aggregate at 300 Adam steps equals any cell of the archived CSV.
- Train-time or inference-time means match the archive. Those columns are wall-clock measurements.
- Single-process execution and `Pool.map` produce the same scores on this machine. The per-trial seed makes that plausible, and it was not measured.
- A 300-step parameter vector matches `QAM(n_bits)` trained inside `run_trial`. The one-step comparison used `QAM(n_bits, train_steps=1)`, which is the same constructor with the step count set to the smoke-sized budget. `QAM(n_bits)` still defaults to 300 steps, and the historical request asks the factory for `train_steps=300`. That default was not trained here.
