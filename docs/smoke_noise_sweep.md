# Noise-sweep smoke runner

`benchmark.runner.run_benchmark` runs one explicit `NoiseSweepRequest`. It composes pattern drawing, synchronous Hopfield, Bernoulli corruption, Rot full-batch QAM, recall, and the noise-sweep metrics. It does not plot, does not parse a CLI, and does not call `experiments.noise_robustness_sweep.main`.

The lightweight request is two trials: `n_bits=2`, corruption `0.0`, seeds `0` and `1`, `train_steps=1`. The pattern count stays `max(1, int(0.3 * n_bits))`, which is 1. Output goes to a caller-supplied directory or stays in memory. The writer refuses any path that resolves to `results/`, to a child of it, or to a parent of it.

## Model identities

| Arm | `procedure_id` | Source |
|---|---|---|
| First | `hopfield_synchronous` | `src.hopfield.Hopfield` |
| Second | `qam_rot_fullbatch` | `experiments.noise_robustness_sweep.QAM` |

The archived summary prefixes the second arm with `qam_`. This runner does not use that label. `qam` is not a procedure id here.

## Order that is preserved

For each trial the runner sets `numpy.random.seed` and `pennylane.numpy.random.seed` to the trial seed, draws patterns with `choice`, trains Hopfield, and applies `corrupt_bernoulli` with no explicit rng. It then calls `QAM(n_bits, train_steps=<smoke>)`. That constructor reseeds the shared stream to 42 and draws `0.01 * randn(depth, n_bits, 3)` with `depth = min(6, n_bits)`. QAM training and the second Bernoulli pass happen after that reset. Recall scores use `batch_recall`. Across trials, full and bit accuracy use `gaussian_ci_population` (`mean_std_ci`, population standard deviation). Train time and per-bit recall time are stored as timings, not as metrics.

## Result files

When an output directory is given, the runner creates a new subdirectory and writes a `running` manifest and `configuration.json` before training. After the tables exist it marks `execution_status` succeeded and writes `trials.csv`, `aggregates.csv`, and `trials.json`. `comparable_to_archive` is false. `trials.json` holds the patterns, queries, recalls, and the QAM parameter draw from before training. The CSV columns are the flat identity, dataset, perturbation, metric, and timing fields.

## What this run is not

- It is not the historical grid: 4 widths, 5 corruption rates, 30 seeds, and 300 Adam steps. A request with `grid="full"`, more than 4 cells, or more than 5 training steps is refused before any model is built.
- It does not use a process pool. The historical `main` does. Trial values are seeded inside the trial; times are not compared with the archive.
- The same integer seed does not make the two arms share queries. Hopfield masks are drawn on the trial seed. QAM masks are drawn after the constructor resets the stream to 42.
- `default_rng` is not this stream.
- Smoke `train_steps` is 1, not 300. The archived CSV is a 30-trial aggregate and has no per-trial rows. This runner does not claim those rows match.
- Bit accuracy is the mean of per-pattern match fractions. On equal-length patterns that is the same ratio as the sweep's pooled bit count, and the two can differ by one floating-point unit. The experiment module still uses the pooled count.
