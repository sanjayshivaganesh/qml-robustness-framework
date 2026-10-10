# Scaling migration

Date: 2026-10-10. The framework requests are `configs/scaling.json` and `configs/scaling.smoke.json`. The historical implementation remains `experiments/scaling.py`. That file was not rewritten, and its `main` was not called.

The pair is `hopfield_async_random` (`src.hopfield.HopfieldAsyncRandom`) and `qam_rot_minibatch` (`experiments.scaling.QAM`). The scaling module imports the Hopfield class under the local name `Hopfield`. The framework uses the procedure id, not that alias. `qam_ry_linear` and `qam_rot_fullbatch` are different circuits and are not selected.

## Preserved protocol

Constants were read from `baseline/configs/original_experiments.json` under `experiments.scaling` and from the module.

| Field | Value |
|---|---|
| Lengths | `4, 6, 8, 12, 16` |
| Noise | `0.0, 0.1, 0.2, 0.3` |
| Trials | `30`, ids `0..29` |
| Seed | `123 + trial_id + n_bits * 100` on `pennylane.numpy.random` |
| Stored patterns | `m = min(max(1, n_bits // 2), 6)` |
| Hopfield | 10 random-order sweeps. A non-negative local field becomes `+1`. Permutations use PennyLane NumPy. |
| QAM call | `QAM(n_bits, depth=min(6, n_bits), steps=200, lr=0.05)` |
| Class default depth | `3`. The protocol call does not use that default. |
| Training | RY encoding, `Rot` plus linear CNOT, minibatch of 2, early stop after 15 steps with absolute loss change below `1e-4` |
| Schedule | Both models train once per trial. Every noise level is scored after that training. |
| Scores | Mean of per-pattern exact matches, and mean of per-pattern bit fractions |
| Rows | One Hopfield row and one QAM row per noise level |

There is no constructor reseed. Parameter `randn` continues the trial seed. Each training step draws a minibatch, and the optimizer evaluates that draw a second time. Early stopping changes how many draws occur. Bernoulli masks and Hopfield permutations continue on the same stream. `bootstrap_ci` in the scaling module draws without a seed. `main` does not call it, and the framework runner does not call it.

The runner calls `corrupt_bernoulli` on the shared legacy stream for each stored pattern. A one-step comparison against the inline `rand` mask in `main` produced the same recalled patterns. `experiments/scaling.py` still contains the inline mask.

Plotting stays in `main` and in `scripts/plot_scaling.py`. The framework manifest records `plotting` false.

`main` is single-process. The framework runner is single-process as well.

For `n_bits = 4` the call uses depth `min(6, 4) = 4`. The parameter count is `4 × 4 × 3 = 48`. The archived CSV stores 48 for that width. The class default depth 3 is a different construction.

## How a run starts

`configs/scaling.json` is the module grid: 5 widths, 30 trials, 4 noise levels, 200 steps, 150 trainings, 1200 model rows. It loads with `executable` false. `python -m benchmark run configs/scaling.json` does not train. `--full-grid` is the switch that would start that grid through `run_scaling`. The Day 16 tests do not pass the flag through to training. Output would go under `runs/scaling/<run_id>/`, and a path that resolves to `results/` is refused. `comparable_to_archive` is false. `resumable` is false.

`configs/scaling.smoke.json` is a reduced budget: width `4`, trial id `0`, noise `0.0`, and `steps` `1`. The pattern rule at width 4 is still `m = 2`, and the seed is still `123 + 0 + 4 * 100 = 523`. The step count is not the historical 200. The manifest records `historical_steps` 200 and `smoke_steps` 1. `setting_origin` labels the reduced fields `demonstration`.

The result table keeps the historical fields `n_bits`, `m`, `trial_id`, `full_acc`, `bit_acc`, `train_time_s`, `infer_time_s`, `params_count`, `noise_p`, and `extra_notes`. It also records `procedure_id` and `historical_model_label`. The label `QAM` is the scaling class, not another QAM. The historical CSV column is named `model` and uses `Hopfield` and `QAM`. The framework column `historical_model_label` carries those two strings. `procedure_id` is the identity.

The manifest `rng` record lists the seed base, the formula, the seed applied to each training, and an empty internal-reset list. `exact_reproducibility_claimed` is false.

## Validation performed

The historical file matches the audit record and the module attributes, and the registry returns `experiments.scaling.QAM`. Factory arguments for width 4 at the historical budget are depth 4, 200 steps, and learning rate 0.05. The factory was not called for that budget.

On the smoke budget, the framework trial and a direct call of `HopfieldAsyncRandom` plus `experiments.scaling.QAM(..., steps=1)` produced the same patterns, the same initial parameters, and the same recalled patterns. Hopfield's parameter count was 16. The QAM parameter count was 48. The run wrote under a temporary directory. `results/scaling/raw/scaling_sweep.csv` was not modified. That file has a header and 1200 data rows. Those rows were not compared with the smoke trial. `results/scaling/` contains that CSV and does not contain the PNG names written at the end of `main`.

## Remaining limitations

These points are not established:

- A 200-step framework table equals the archived CSV. The smoke used one step. Early stopping did not have 15 steps in which to fire.
- Train-time or inference-time columns match the archive. Those values are wall-clock measurements.
- The recorded seed reproduces every minibatch after step 1. The second loss evaluation and the early-stop break are still inside the class, and the manifest does not store the `RandomState`.
- `bootstrap_ci` was left unseeded. Seeding it would be a change to a helper the historical `main` does not call.
