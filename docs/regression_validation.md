# Regression validation

Date: 2026-10-10. This pass compares framework wrappers with the original functions on small inputs. It does not run an experiment `main`, does not start a historical grid, and does not write under `results/`.

`./venv/bin/python -m pytest -q` passed 123 tests after this pass. Archived files under `results/` were unchanged (`git status -- results` empty, `git diff --stat HEAD -- results` empty).

No research equation was edited to make a test pass. The one test fix in this pass was an assertion that compared a weight matrix to the scalar `0.0`; NumPy rejected the shapes. The recall result itself was already `[1, 1, 1]`.

## Controls

| Comparison | What was held fixed | What was not claimed |
|---|---|---|
| Synchronous Hopfield | Hand-built ±1 patterns. No seed. Adapter versus `Hopfield` on the same matrix. | A multi-pattern basin, or the 10-step path on a nonzero spurious state. |
| Sequential Hopfield | Hand-built `W` and query. `steps=1`, then the default 20-step call. | The inference experiment's pattern draw or its Bernoulli mask. |
| Random-order Hopfield | `numpy.random.seed(7)` and `pennylane.numpy.random.seed(7)`, one sweep, two stored patterns of length 4. State restored afterward. | The scaling trial, which also draws patterns, Rot parameters, and minibatches before the permutations. Ten default sweeps were not the comparison. |
| Bernoulli and fixed-count | Explicit `numpy.random.RandomState`. Same state passed to the framework helper and to `src.data_generation`. | Calls inside `experiments/scaling.py` or `experiments/noise_robustness_inference.py`. Those modules still use an inline mask. |
| Metrics | Hand patterns, plus `RandomState` for bootstrap helpers. Global state saved around the paired bootstrap. | Unseeded inference percentiles. Those functions were not seeded in order to make them repeatable. |
| Hamming nearest neighbour | One hand query. Adapter versus `nearest_neighbor_recall`. | The inference CSV column `model="qam"`. That string is still only a label. |
| Sign map | A stand-in circuit that returns `[0, 0.2, -0.3]`. | A trained depth-experiment circuit. `experiments/circuit_depth.py` still signs inline and was not executed. |
| Rot full-batch QAM | Global seeds saved and restored. A `randn` draw, then construction. `train_steps=1` was passed and `train` was not called in this comparison. | 300 Adam steps, the 4×5×30 grid, or the archived summary. |
| Rot minibatch QAM | Seed 3, then an optional extra `rand()`. Construction only, `depth=1`, `steps=1`, `lr=0.05`. | Minibatch index draws, the second `loss` call inside `opt.step`, early stopping, or `steps=200`. |
| RY-linear QAM | Seed 1, pattern `[1, 1]`, `n_bits=2`, `depth=1`, 2 Adam steps, `lr=0.05`, `return_tuple=True`. Loss history compared at `steps=1` only. | Capacity's 100-step grid, convergence's 300-step grid, or circuit-depth recall. |
| PauliX QAM | Seed 1, one length-2 pattern, the real `build_qam` / `train_qam`. Depth stays 10 and the loop stays 100 epochs because those constants are closed over. | The statistical-significance grid (30 seeds, several widths and noise levels). |
| Noise-sweep smoke | Two cells, `n_bits=2`, corruption 0, seeds 0 and 1, one Adam step. Output under `tmp_path`. | The historical 600-cell, 300-step sweep. `comparable_to_archive` is false. |

`default_rng(7).permutation(4)` was compared with the legacy permutation after seed 7. The sequences differ. Equal integers are not treated as the same stream.

## Findings

### Validated

These matched on the inputs above, and the check does not depend on a shortened optimizer.

- Synchronous recall of one stored pattern equals that pattern, and the adapter equals `Hopfield`. A zero weight matrix recalls to `+1` on every coordinate (`sign(0)` is rewritten). `tests/test_phase1_equivalence.py`.
- Sequential recall follows index order. For `W = [[0, 1], [0, 0]]` and query `[-1, -1]`, one sweep ends at `[-1, 1]`. Final energy and energy drop match `-0.5 x^T W x` on the input and the final state. A zero matrix writes `+1`. The adapter equals `hopfield_async_sequential_recall`, including the default 20-step call. Same file.
- Random-order adapter recall equals `HopfieldAsyncRandom` under seed 7. One sweep consumes exactly one `pennylane.numpy.random.permutation` and no other legacy draw. That permutation is `[2, 1, 0, 3]`, and it equals `numpy.random.permutation` on the same shared stream. Same file.
- Bernoulli rate 0 flips nothing; rate 1 flips every bit. The framework helper matches `corrupt_bernoulli` and leaves the same leftover `RandomState`. The three-line inline mask matches the helper. `tests/test_datasets.py`.
- Fixed-count flips `max(1, int(n * p))`. At `n=4`, `p=0.15` that is one bit. At `p=0` it is still one bit. A request above length `n` raises `ValueError`. Level `0` is not the same procedure as Bernoulli. Same file.
- Metric helpers match `src.metrics` on the hand cases in `tests/test_metrics_regression.py`: exact recall, total mismatch, 3-of-4 bits, batch means, both Gaussian standard deviations, seeded bootstrap mean and percentile intervals, paired bootstrap sign, Cliff's delta bands, Wilcoxon sentinels, and the 0.75 / 0.50 / 0.90 rules.
- Hamming adapter output equals `nearest_neighbor_recall` and is a view of the stored row. Procedure id remains `hamming_nearest`.
- The noise-sweep writer refuses `results/`, a child of it, a parent of it, and a symlink into it. `tests/test_output_safety.py`.

### Validated only for controlled or small cases

- Rot full-batch construction, after an unrelated `randn`, matches `0.01 * randn` drawn after `pennylane.numpy.random.seed(42)`. The draw that would have continued without that reset does not match. Adapter parameters and untrained recall match `experiments.noise_robustness_sweep.QAM` on the same prefix. The smoke runner's one-step train is a separate check and is not this 300-step procedure.
- Rot minibatch parameters match `experiments.scaling.QAM` when both constructors see the same stream position. An extra `rand()` before construction changes the parameters, so this class does not reset to 42. Training was not run.
- RY-linear training for 2 Adam steps matches `train_ry_linear_qam` from seed 1. Recall is `[-1, 1]` on target `[1, 1]`, which is the value already written in `docs/baseline_experiments.md`. `train_recording_loss` matches the original history at `steps=1` (two loss values, the step-0 record and the final record).
- PauliX `train` and `recall` match the statistical-significance functions on one length-2 pattern from seed 1. The parameter shape is `(10, 2)`. This still executes the module's 100-epoch loop. It does not execute the published grid.
- The sign map used by `src.qam.qam_recall` and by the PauliX `qam_recall` sends a controlled 0 to `+1`. Both adapters return that vector. The depth experiment's inline block was not called.
- The smoke orchestration writes the procedure ids `hopfield_synchronous` and `qam_rot_fullbatch`, keeps `comparable_to_archive` false, and refuses the historical cell count before building a model. `tests/test_smoke_noise_sweep.py`.

### Not yet validated

- Any archived CSV, including `results/noise_robustness_sweep/summary/noise_robustness_sweep.csv`. That file is a 30-trial aggregate. A one-step smoke is not a comparison with it.
- Historical grids: noise sweep (600 cells, 300 steps), storage capacity, circuit depth, convergence at 300 steps, scaling trials, statistical significance, and the inference task list.
- Circuit-depth recall on a real depth-script circuit. The adapter deliberately does not wrap that inline sign.
- Rot minibatch training: minibatch `choice`, the extra `loss` evaluation inside `opt.step`, and the patience-15 stop.
- Where the inline Bernoulli mask sits inside a scaling trial or an inference trial. The mask matches the helper when both are given the same `RandomState`. The experiment modules were not switched over, and their mains were not run.
- Multiprocessing pools. Smoke execution is single-process. Trial seeds are applied inside the trial; wall-clock columns were not compared.
- Inference summary percentiles and `scripts/plot_bootstrap_distribution.py`. Both are unseeded.
- Config loading against `baseline/configs/original_experiments.json` (not implemented).
- A manifest of package versions (the smoke manifest records procedure ids, the reseed, and `comparable_to_archive: false` only).
- An explicit `RandomState` argument on the historical call path (not implemented).

### Known limitation

- The noise-sweep QAM constructor calls `pennylane.numpy.random.seed(42)` before drawing parameters. Hopfield queries use the trial seed. QAM queries use whatever follows that reset. The same trial integer is not a shared query sequence.
- Fixed-count at `p=0` flips one bit. Bernoulli at rate 0 flips none.
- `numpy.random.default_rng` is a different generator from the legacy stream shared by `numpy.random.seed` and `pennylane.numpy.random.seed`.
- Empty full-pattern accuracy is `1.0` because `np.all` of an empty comparison is true. `batch_recall` follows `zip` and stops at the shorter list.
- On equal-length batches, the noise sweep's pooled bit count and the mean of per-pattern bit fractions are the same ratio and can differ by one unit in the last place. The experiment module still uses the pooled count.
- On SciPy 1.17.1, five identical Wilcoxon pairs return p-value `1.0` and do not take the `except` branch. A one-element tie still returns the caller sentinel (`None` for capacity, `1.0` for statistical significance). Archived `None` cells were not recomputed.
- Train time and recall time are machine-dependent and are not acceptance criteria.
- `scripts/run_all_experiments.py` can still launch a main that overwrites `results/`. The new runner cannot. The launcher flag described in the migration plan was not added.

### Failure requiring correction

None. Every framework-versus-original comparison in this pass agreed on the controlled input. No adapter, dataset helper, metric, or smoke path was changed.
