# Phase 1 status

Date: 2026-10-10. Phase 1 is the framework layer added so far: model adapters, dataset and perturbation helpers, metric selection, the noise-sweep smoke runner, and the output guard. This note does not add a feature. It records whether that layer still calls the original science.

Full suite after the added comparisons: `./venv/bin/python -m pytest -q`, 123 passed. `results/` was not modified.

## What Phase 1 contains

| Piece | Location | Status |
|---|---|---|
| Eight procedure adapters | `benchmark/adapters/` | Present. Each calls the existing class or function. |
| Pattern batches and both corruptions | `benchmark/data/` | Present. Draws and flips go through `src.data_generation`. |
| Metric selection | `benchmark/scoring.py` | Present. Protocols ask for the metrics they already report. |
| Noise-sweep smoke | `benchmark/runner.py`, `benchmark/protocols/noise_robustness_sweep.py` | Present for at most 4 cells and at most 5 Adam steps. The historical grid is refused. |
| Output guard | `benchmark/output.py` | Present. Refuses the archived `results/` tree. |
| Config loader, versioned manifests, explicit `RandomState` migration, launcher overwrite flag | — | Not in Phase 1. |

Experiment modules still own their loops. Scaling and inference still use an inline Bernoulli mask. Circuit depth still signs its own circuit output. No `main` was moved onto the runner.

## Diff review

Against `HEAD`, the tracked scientific files differ in comments, docstrings, and a few print strings only. Checked paths: `src/hopfield.py`, `src/qam.py`, `src/data_generation.py`, `src/paths.py`, `experiments/*.py`, and `scripts/*.py`. Numeric grids, the reseed to 42, `max(1, int(n * p))`, losses, and recall updates are the same lines as `HEAD`.

Non-comment tracked edits are `pyproject.toml` (the `benchmark` package include) and `.gitignore` (`runs/`).

The framework itself is untracked new code under `benchmark/`, `tests/test_*.py` for the framework, `baseline/`, and `docs/` for the design and these validation notes. `results/` has no staged or unstaged change.

## Classification

| Finding | Class |
|---|---|
| Synchronous Hopfield: stored pattern recalls to itself; zero field becomes `+1`; adapter equals `Hopfield` | Validated |
| Sequential Hopfield: index order, energy `-0.5 x^T W x`, zero field becomes `+1`; adapter equals the original function | Validated |
| Random-order Hopfield, one sweep at seed 7: adapter equals `HopfieldAsyncRandom`; the sweep consumes one PennyLane permutation | Validated only for controlled/small cases |
| Bernoulli helper, including rate 0 and rate 1, matches `corrupt_bernoulli` on a `RandomState` | Validated |
| Fixed-count, including `max(1, int(n * p))` at `p=0` and at `n=4`, `p=0.15` | Validated |
| Metric helpers versus `src.metrics` on hand patterns and seeded intervals | Validated |
| Hamming nearest neighbour versus `nearest_neighbor_recall` | Validated |
| Rot full-batch init follows reseed 42 rather than the preceding draws; adapter matches the noise-sweep class before training | Validated only for controlled/small cases |
| Rot minibatch init continues the current stream and matches the scaling class; training not run | Validated only for controlled/small cases |
| RY-linear, 2 Adam steps, seed 1, length-2 pattern: adapter matches `train_ry_linear_qam`; recall `[-1, 1]` | Validated only for controlled/small cases |
| RY-linear loss history at `steps=1` matches `train_ry_linear_qam_record_loss` | Validated only for controlled/small cases |
| PauliX train and recall on one length-2 pattern match the statistical-significance functions (depth 10, 100 epochs) | Validated only for controlled/small cases |
| Sign of a controlled expectation vector, including a 0 mapped to `+1` | Validated only for controlled/small cases |
| Noise-sweep smoke schema, procedure ids, and refusal of the historical grid | Validated only for controlled/small cases |
| Output paths cannot resolve into `results/` | Validated |
| Archived CSV values | Not yet validated |
| Full historical experiments and their process pools | Not yet validated |
| Circuit-depth inline recall on that experiment's circuit | Not yet validated |
| Rot minibatch training, including the second loss evaluation | Not yet validated |
| Inline Bernoulli left inside scaling and inference, in those modules' full RNG order | Not yet validated |
| Unseeded inference bootstrap and the bootstrap histogram script | Not yet validated |
| Config loader and package-version manifest | Not yet validated |
| Noise-sweep constructor reset to 42, so one trial seed is not one shared query stream | Known limitation |
| Fixed-count at `p=0` still flips one bit | Known limitation |
| `default_rng` is not the legacy stream | Known limitation |
| Empty full-pattern accuracy is `1.0`; `zip` truncates unequal batches | Known limitation |
| Pooled bit counts and mean-of-rates can differ by one floating-point unit | Known limitation |
| Wilcoxon on SciPy 1.17.1 does not hit the old failure branch for ties of length ≥ 2 | Known limitation |
| Wall-clock columns are not acceptance criteria | Known limitation |
| The legacy launcher can still overwrite `results/` | Known limitation |
| Framework result disagreed with the original function | Failure requiring correction: none |

Evidence, seeds, and the exact inputs are in `docs/regression_validation.md`.

## Reading the smoke result

`qam_rot_fullbatch` in a smoke row is the noise-sweep Rot class with a shortened Adam budget. It is not `qam_ry_linear`, `qam_rot_minibatch`, `qam_paulix`, or `hamming_nearest`. The archived noise-sweep column prefix `qam_` refers to that same Rot class at 300 steps and 30 trials. The smoke does not reproduce that column.
