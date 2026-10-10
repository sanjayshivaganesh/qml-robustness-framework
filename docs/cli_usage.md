# Benchmark command line

Date: 2026-10-10. The entry point is `python -m benchmark`, from the repository root. It uses the standard-library `argparse` parser. There is no separate CLI package.

The command loads a request with `benchmark.config_loader` and, when a run is allowed, calls `benchmark.runner.run_benchmark`. It does not call `python -m experiments.<name>` or `python -m scripts.run_all_experiments`. Those commands still write under `results/`.

The runner starts six reduced requests. The noise-sweep smoke is `configs/noise_robustness_sweep.smoke.json`: `n_bits` 2, Bernoulli level 0, seeds 0 and 1, one Adam step. The scaling smoke is one width and one training step. The statistical-significance smoke is one module cell at 100 epochs. The storage-capacity smoke is one memory load, `n=8` and `m=1`, at 100 steps and five seeds. The circuit-depth smoke is one cell, `n=8`, `m=4`, depth `1`, at 100 steps and five seeds. The inference smoke is one cell, `n=8`, capacity `0.1`, the second module noise level, and 300 trials. None of those is the historical grid for that protocol.

## Commands

| Command | What it does |
|---|---|
| `python -m benchmark --help` | Lists `validate` and `run`, and prints the examples below. |
| `python -m benchmark validate CONFIG` | Loads `CONFIG`. Prints the experiment id, model ids, dataset, perturbation, trial count, and output destination. Does not train, does not construct a circuit, and does not write a result. |
| `python -m benchmark run CONFIG` | Prints that same plan, then runs the file. `CONFIG` is required. |
| `python -m benchmark run CONFIG --dry-run` | Prints the plan and stops. Does not train or write a result. |
| `python -m benchmark run CONFIG --output-dir DIR` | Creates a new run directory inside `DIR` instead of using the destination in the file. |
| `python -m benchmark run CONFIG --overwrite` | Writes into the destination itself when that directory already has files. A path under `results/` is still refused. |
| `python -m benchmark validate configs/noise_robustness_sweep.json` | Loads the historical grid and prints `executable: no`. Does not train. |
| `python -m benchmark run configs/noise_robustness_sweep.json --dry-run` | Prints that grid and does not start it. |
| `python -m benchmark run configs/noise_robustness_sweep.json --full-grid` | Starts the 4×5×30 grid at 300 Adam steps through the framework runner. It does not call `experiments.noise_robustness_sweep.main` and it does not write under `results/`. The tests do not pass this flag. |
| `python -m benchmark validate configs/scaling.smoke.json` | Loads the reduced scaling budget: width 4, trial 0, noise 0, one training step. |
| `python -m benchmark run configs/scaling.json --dry-run` | Prints the historical scaling grid and does not start it. |
| `python -m benchmark run configs/scaling.json --full-grid` | Would start 5 widths, 30 trials, and 200 steps. The tests do not pass this flag. See `docs/scaling_migration.md`. |
| `python -m benchmark validate configs/statistical_significance.smoke.json` | Loads one module cell: `n=8`, capacity `0.2`, noise `0.1`, seed `0`, 100 epochs, 5000 unseeded bootstrap draws. |
| `python -m benchmark run configs/statistical_significance.json --dry-run` | Prints the 720-task grid and does not start it. |
| `python -m benchmark run configs/statistical_significance.json --full-grid` | Would start 3 widths, 4 capacities, 2 noise levels, 30 seeds, and 100 epochs. The tests do not pass this flag. See `docs/statistical_significance_migration.md`. |
| `python -m benchmark validate configs/storage_capacity.smoke.json` | Loads one storage-capacity cell: `n=8`, `m=1`, seeds 0 through 4, 100 steps. Does not train. |
| `python -m benchmark run configs/storage_capacity.json --dry-run` | Prints the 33-cell grid and does not start it. |
| `python -m benchmark run configs/storage_capacity.json --full-grid` | Would start 33 memory loads, 5 seeds, and 100 steps. The tests do not pass this flag. See `docs/storage_capacity_migration.md`. |
| `python -m benchmark validate configs/circuit_depth.smoke.json` | Loads one circuit-depth cell: `n=8`, `m=4`, depth `1`, seeds 0 through 4, 100 steps, `return_tuple` false. Does not train. |
| `python -m benchmark run configs/circuit_depth.json --dry-run` | Prints the 21-cell depth list and does not start it. |
| `python -m benchmark run configs/circuit_depth.json --full-grid` | Would start 3 cases, 7 depths, 5 seeds, and 100 steps. The tests do not pass this flag. See `docs/depth_migration.md`. |
| `python -m benchmark validate configs/noise_robustness_inference.smoke.json` | Loads one inference cell: `n=8`, capacity `0.1`, one module noise level, 300 trials. Does not recall. |
| `python -m benchmark run configs/noise_robustness_inference.json --dry-run` | Prints the 216,000-task grid and does not start it. |
| `python -m benchmark run configs/noise_robustness_inference.json --full-grid` | Would start 9 widths, 5 capacities, 16 noise levels, and 300 trials. The tests do not pass this flag. See `docs/inference_migration.md`. |

`validate CONFIG` and `run CONFIG` both require the path. A bare `python -m benchmark`, `validate`, or `run` exits with status 2.

## Examples

```bash
python -m benchmark validate configs/noise_robustness_sweep.smoke.json
python -m benchmark run configs/noise_robustness_sweep.smoke.json --dry-run
python -m benchmark run configs/noise_robustness_sweep.smoke.json --output-dir runs/noise_robustness_sweep_smoke
```

The last command is the smoke in the example file. `runs/` is gitignored. Omit `--output-dir` to use the destination written in the file, which for that example is `runs/noise_robustness_sweep_smoke`. Each run is a new subdirectory of that destination, so a second run does not replace the first. `--overwrite` writes into the named directory itself and still cannot select `results/`.

A plan from the example looks like this:

```text
experiment_id: noise_robustness_sweep_smoke
protocol: noise_robustness_sweep
models: hopfield_synchronous qam_rot_fullbatch
dataset.n_bits_list: 2
dataset.capacity_ratio: 0.3
dataset.pattern_generator: numpy.random
perturbation: bernoulli
perturbation.levels: 0.0
trial_count: 2
trial_seeds: 0 1
train_steps: 1
grid: smoke
output_destination: <repository>/runs/noise_robustness_sweep_smoke
overwrite: no
run_directory: a new subdirectory of output_destination
executable: yes
```

`run` prints that block before it calls the runner. After a completed run it also prints `run_complete: yes`, the unique run directory, the procedure ids, and `comparable_to_archive: false`.

The same commands write stage lines to stderr through the `benchmark` logger. A line starts with `event=`. Validation, a dry run, each smoke cell, model construction, training, evaluation, and result writes each produce one line. Circuit evaluations do not. A failure is logged with its traceback and re-raised; the manifest status becomes `failed` when the runner can still replace the file. `resumable` is false. The run is not resumed. Details are in `docs/execution_and_failure_handling.md`.

The run directory contains `manifest.json`, `configuration.json`, `trials.csv`, `aggregates.csv`, and `trials.json`. The manifest records the run id, experiment id, configuration, start and completion times, Python and package versions, the git commit when `git rev-parse HEAD` succeeds, the selected models, the dataset and perturbation, `execution_status`, the output filenames, and an `rng` record. That record lists the configured trial seeds, the seed calls at the start of each trial, and the Rot constructor reset to 42. `exact_reproducibility_claimed` is false. Details are in `docs/randomness_and_reproducibility.md`. `execution_status` is `succeeded` only after those tables are written. A failure leaves `failed` and the error text. `output_class` is `framework_run`, and `comparable_to_archive` is false. Files are written by a temporary name and `os.replace`, so a failed replace does not leave the finished filename.

## Exit status

| Status | Meaning |
|---|---|
| 0 | `validate` loaded the file. `run --dry-run` loaded an executable file and did not run it. `run` finished. |
| 1 | The file loaded, and the current runner will not start it, or the output path resolves to the archived `results/` tree. Nothing is trained in those cases. |
| 2 | The arguments are missing or unknown, or the file is not a valid request. Unknown model ids, including `qam`, land here. |

`validate` of a schema-valid historical budget (`grid: full`, `train_steps: 300`) exits 0 and prints `executable: no` plus the blockers. `run` of that same file exits 1 and does not call the runner.

A path under `results/`, including through a symlink, is rejected by the loader when it is the file's destination. `--output-dir` pointing at that tree is rejected by the run writer before training.

## What this command does not start

Convergence has no runner. Selecting that protocol name fails validation. The noise-sweep file cannot select sequential Hopfield, random-order Hopfield, RY-linear, Rot minibatch, PauliX, or Hamming nearest neighbour. Scaling, statistical significance, storage capacity, circuit depth, and inference have their own request files.

The historical noise sweep stays refused: 4 widths, 5 corruption rates, 30 seeds, and 300 Adam steps.
