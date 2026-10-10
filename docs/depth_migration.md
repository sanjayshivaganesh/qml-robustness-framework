# Circuit-depth migration

Date: 2026-10-10. The protocol is `experiments/circuit_depth.py`. The framework request files are `configs/circuit_depth.json` and `configs/circuit_depth.smoke.json`. The runner is `benchmark.protocols.circuit_depth`. `experiments.circuit_depth.main` is unchanged and was not called.

## Protocol

The experiment trains one model, `src.qam.train_ry_linear_qam`, and scores it. There is no Hopfield arm.

The circuit is the RY-linear QNode built with `return_tuple=False`. That branch returns a list of Pauli-Z expectations. `return_tuple=True` is the storage-capacity and convergence QNode, and the depth configuration rejects it. Depth is the ansatz layer count passed into the trainer. The module list is `1, 2, 4, 6, 8, 10, 12`. The storage-capacity rule `min(8, n_bits)` is not applied, so depth 10 and depth 12 remain in the historical list at `n=8`.

The cases are the module map `(n, m) = (8, 4), (12, 8), (16, 12)`. `m` is that map, not a capacity formula.

Each trial, for every `(n, depth)` cell:

1. `numpy.random.seed(seed)` for `seed` in `0..4`. There is no PennyLane seed call.
2. `numpy.random.choice` draws `m` patterns of length `n`.
3. `train_ry_linear_qam` draws `0.01 * randn(depth, n)`, then runs 100 Adam steps at learning rate `0.05`. The loss stays inside that function.
4. Each stored pattern is corrupted with `corrupt_fixed_count` at level `0.15`, using `rng=numpy.random`. The flip count is `max(1, int(n * 0.15))`. These draws happen after training.
5. Recall is the three lines in `main`: PennyLane `array` of `circuit(params, noisy)`, `sign`, and zeros mapped to `+1`. The runner calls `sign_circuit_output` in `benchmark/protocols/circuit_depth.py`. It does not call `src.qam.qam_recall`.

After the five seeds, the cell reports the mean full accuracy, the mean bit accuracy, and the mean train time. The parameter column is `depth * n_bits`. There is no bootstrap and no Wilcoxon test.

A fixed stand-in that returns `[0, 0.2, -0.3]` maps to `[1, 1, -1]` under both `sign_circuit_output` and `qam_recall`. The runner still uses `sign_circuit_output` with the `return_tuple=False` circuit. `experiments/circuit_depth.py` is not edited to call `qam_recall`.

## Smoke and historical settings

| Setting | Historical module | Smoke request |
|---|---|---|
| Cases | `(8, 4), (12, 8), (16, 12)` | `(8, 4)` only |
| Depths | `1, 2, 4, 6, 8, 10, 12` | `1` only |
| Cells | 21 | 1 |
| Trainings | 105 | 5 |
| Seeds | `0..4`, reapplied on every cell | same five seeds |
| Steps, learning rate, noise | 100, `0.05`, `0.15` | same |
| `return_tuple` | false | false |
| Aggregation | mean over the five seeds | same |
| `executable` | false until `--full-grid` | true |

The smoke depth is the first historical depth. It is labelled `demonstration` because the other six depths are not scored. Steps, learning rate, noise, seeds, and `return_tuple=False` stay the module values and are labelled `historical`. `comparable_to_archive` stays false.

`configs/circuit_depth.json` loads the 21-cell grid and does not start it. `--full-grid` is the only start switch, and the tests do not pass it into a real run.

## Validation performed

`tests/test_circuit_depth_migration.py` checks:

- The historical file matches the module and `baseline/configs/original_experiments.json`: cases, depths, 100 steps, learning rate `0.05`, noise `0.15`, five seeds, and `return_tuple` false.
- The selected adapter is `qam_ry_linear`, whose train passthrough is `train_ry_linear_qam`. `prepare` accepts depth 12 with `return_tuple` false. The cell runner passes `return_tuple=False` and calls `sign_circuit_output`.
- A `return_tuple=False` QNode returns a list. A `return_tuple=True` QNode returns a tuple. A config that asks for `return_tuple` true, for `qam_recall`, for one training step, for `(n, m) = (8, 1)`, or for depth `3` is rejected.
- Loading either file, and refusing the historical grid, does not call `main` or the trainer.
- The smoke cell was executed: `n=8`, `m=4`, depth `1`, seeds `0..4`, 100 steps. Patterns, parameters, full accuracy, and bit accuracy matched a direct loop that uses `train_ry_linear_qam(..., return_tuple=False)` and the three-line sign map. The legacy `RandomState` matched after both runs. The manifest records `qam_ry_linear`, `return_tuple` false, the inline recall id, `depth_rule_not_applied` `min(8, n_bits)`, `bootstrap_used` false, and `exact_reproducibility_claimed` false.
- That cell's mean full accuracy, mean bit accuracy, and parameter count match the first row of `results/circuit_depth/summary/depth_scaling.csv` (`n=8`, `m=4`, depth `1`, full accuracy `0.0`, bit accuracy `0.55625`, 8 parameters). Train time was not compared. The other 20 archived rows were not recomputed.
- The archive CSV still has 21 data rows. Its bytes, and the bytes of `results/circuit_depth/depth_scaling.log`, were unchanged.
- A destination under `results/circuit_depth/summary` is refused before training.
- `validate` prints the smoke plan as executable and the 21-cell plan as not executable. `run` of the historical file without `--full-grid` does not train. The smoke file rejects `--full-grid`.

## Remaining limitations

The 21-cell equality was not measured. The archived log records training times from about 1.5 seconds at `n=8`, depth 1, to about 356 seconds at `n=16`, depth 12. The full grid is 105 trainings of 100 steps. `--full-grid` was not executed.

The one matching archive row is that single cell's accuracy and parameter count. It is not a reproduction of the depth table, the plots, or the train-time column. A recorded seed list is not a claim that a later machine will match those times.

The smoke critical comparison does not exist for this protocol: there is no threshold and no bootstrap. A depth-1 mean is not a statement about whether accuracy flattens at the later depths.
