# Baseline experiments

Audit date: 2026-10-09.

This record distinguishes three kinds of statement:

- **Verified from source:** read from the current module, and unchanged by this audit.
- **Verified from archive:** read from a file under `results/` during this audit. These numbers were not produced by a fresh scientific run in this session.
- **Unknown / not re-executed:** a full `python -m experiments.<name>` was not run. Those commands overwrite `results/<name>/`. The archived storage-capacity log records about 2 hours 42 minutes for the `n=16` portion alone.

Machine-readable copy of the in-code constants: `baseline/configs/original_experiments.json`. That file is a transcription. It is not a runner and it contains no regenerated metrics.

The primary paired Hopfield-versus-variational-QAM protocol is `experiments.noise_robustness_sweep`. The other modules are additional archived protocols. Their QAM columns are different procedures.

## What was executed in this audit

**Tests (verified this session).**

```bash
./venv/bin/python -m pytest -q
```

Exit code 0. Forty tests passed. They check imports and paths. They do not check recall accuracy.

**In-memory smoke check (verified this session, not a baseline result).** A short script called the existing helpers and printed the values below. It did not write into `results/` and it did not use any experiment grid.

| Check | Observed output |
|---|---|
| `pennylane.numpy.random.seed(123)` then `numpy.random.rand()` | `0.6964691855978616`, equal to `numpy.random.seed(123)` followed by `numpy.random.rand()`. The two `seed` functions are not the same object. The stream is shared. |
| `numpy.random.seed(0)`, `random_bipolar_patterns(2, 4)` | `[[-1, 1, 1, -1], [1, 1, 1, 1]]` |
| `Hopfield(4).train` then `recall` on the first pattern | Exact match. `compute_metrics` returned `(1.0, 1.0, 0.0)`. Train time `2.6941299438476562e-05` seconds. |
| `corrupt_bernoulli` at rate `0.25` on that pattern, continuing the same RNG | Query `[-1, 1, 1, -1]` (no bits flipped on this draw). Recall metrics `(1.0, 1.0, 0.0)`. |
| `corrupt_fixed_count` at `0.15` | Query `[-1, -1, 1, -1]`. One bit flipped (`max(1, int(4 * 0.15))`). |
| `train_ry_linear_qam` on one length-2 pattern, depth 1, 2 Adam steps, lr `0.05`, after `numpy.random.seed(1)` | Target `[1, 1]`. Recall `[-1.0, 1.0]`. Metrics `(0.5, 0.0, 0.5)`. Train time `0.003270864486694336` seconds. Parameter shape `(1, 2)`. |
| `nearest_neighbor_recall` and `hopfield_async_sequential_recall` on the length-4 noisy query | Nearest neighbor returned `[-1, 1, 1, -1]`. Async recall metrics `(1.0, 1.0, 0.0)`, energy `-2.0`, energy drop `0.0`. |

These smoke numbers are not comparable to the archived CSVs. Two Adam steps on two qubits is not the capacity, depth, or convergence protocol.

**Scientific experiment mains.** Not run.

## Environment used for the checks

Verified by import in this session:

- Python 3.13.7
- numpy 2.4.4, pandas 3.0.2, pennylane 0.44.1, scipy 1.17.1, matplotlib 3.10.8, seaborn 0.13.2, psutil 7.2.2
- `pennylane_lightning` imports. Experiment source requests `qml.device("default.qubit", ...)`.

The pins are `pyproject.toml` and `requirements.txt`. Python requirement in the package metadata is `>=3.10`. The archived logs do not record the package versions that produced the CSVs.

## Primary baseline: noise robustness sweep

**Source:** `experiments/noise_robustness_sweep.py`. **Status of the run:** archived CSV, not re-executed here.

### Configuration (verified from source)

| Item | Value |
|---|---|
| Pattern lengths | `4, 6, 8, 10` |
| Trials per cell | `30` |
| Seeds | `0 .. 29`, applied inside `run_trial` |
| Corruption | Bernoulli probabilities `0.0, 0.1, 0.2, 0.3, 0.4` |
| Stored patterns | `m = max(1, int(0.3 * n_bits))` |
| Hopfield | `src.hopfield.Hopfield`, 10 synchronous steps, zeros → `+1` |
| QAM | Experiment-local class. `default.qubit`. Encoding `RX(x·π)` and `RY(x·π/2)`. `Rot` plus linear CNOT. Depth `min(6, n_bits)`. 300 Adam steps, lr `0.05`. Parameter init scale `0.01`. |
| QAM reseed | `QAM.__init__` calls `pnp.random.seed(42)` before drawing parameters |
| Loss | Average over patterns of the sum of squared Pauli-Z errors |
| Recall | `sign` of expectations; zeros → `+1` |
| Queries | Every stored pattern, corrupted once per model |
| CI | `1.96 * population std / sqrt(30)` via `mean_std_ci` |

### Metric definitions (verified from source)

- Full-pattern accuracy: fraction of the `m` stored patterns for which the recalled vector equals the target.
- Bit accuracy: matching bits divided by `m * n_bits`.
- `hopfield_infer_mean` / `qam_infer_mean`: mean, over trials, of the mean per-pattern inference time. Each pattern's time is wall time divided by `n_bits`.

### Archived summary (verified from `results/noise_robustness_sweep/summary/noise_robustness_sweep.csv`)

20 data rows. Full-pattern accuracy means:

| n | p | Hopfield full | QAM full | Hopfield bit | QAM bit |
|---|---|---|---|---|---|
| 4 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 4 | 0.1 | 0.9333333333333333 | 1.0 | 0.9666666666666667 | 1.0 |
| 4 | 0.2 | 0.7666666666666667 | 0.0 | 0.8666666666666667 | 0.425 |
| 4 | 0.3 | 0.6 | 0.0 | 0.7166666666666667 | 0.75 |
| 4 | 0.4 | 0.4666666666666667 | 0.0 | 0.6166666666666667 | 0.75 |
| 6 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 6 | 0.1 | 1.0 | 1.0 | 1.0 | 1.0 |
| 6 | 0.2 | 0.9 | 1.0 | 0.95 | 1.0 |
| 6 | 0.3 | 0.7 | 0.0 | 0.8166666666666667 | 0.7055555555555556 |
| 6 | 0.4 | 0.5 | 0.0 | 0.6166666666666667 | 0.31666666666666654 |
| 8 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 8 | 0.1 | 0.8333333333333334 | 0.0 | 0.9520833333333333 | 0.5541666666666667 |
| 8 | 0.2 | 0.6 | 0.0 | 0.8770833333333333 | 0.5395833333333333 |
| 8 | 0.3 | 0.4 | 0.0 | 0.7354166666666667 | 0.5395833333333333 |
| 8 | 0.4 | 0.26666666666666666 | 0.0 | 0.625 | 0.525 |
| 10 | 0.0 | 0.8444444444444444 | 1.0 | 0.9822222222222222 | 1.0 |
| 10 | 0.1 | 0.6 | 0.0 | 0.9288888888888888 | 0.6055555555555556 |
| 10 | 0.2 | 0.5 | 0.0 | 0.8733333333333333 | 0.49444444444444446 |
| 10 | 0.3 | 0.3333333333333333 | 0.0 | 0.7999999999999999 | 0.4822222222222222 |
| 10 | 0.4 | 0.16666666666666666 | 0.0 | 0.691111111111111 | 0.48222222222222216 |

The CSV also stores standard deviations, CI half-widths, and mean train and inference times. Example, verified from the `n=4`, `p=0` row: Hopfield train mean `1.544952392578125e-05` seconds, QAM train mean `3.6917750040690103` seconds, Hopfield infer mean `5.646546681722005e-06`, QAM infer mean `0.0009309430917104085`.

Per-trial rows, patterns, and trained parameters are not in the archive.

`results/noise_robustness_sweep/experiment.log` is a failed path error against `ResearchCode`. It is not a log of the successful run. Wall-clock for the archived sweep is unknown.

### Command and outputs

```bash
python -m experiments.noise_robustness_sweep
```

Current code writes:

- `results/noise_robustness_sweep/summary/noise_robustness_sweep.csv`
- `results/noise_robustness_sweep/plots/noise_full_acc_n{4,6,8,10}.png`
- `results/noise_robustness_sweep/plots/noise_bit_acc_n{4,6,8,10}.png`

Also on disk, from `scripts/plot_noise_robustness_sweep.py`: `full_accuracy_vs_noise.png`, `bitwise_accuracy_vs_noise.png`.

On disk with no writer in the current tree: `bitwise_noise.png`, `full_pattern_noise.png`.

## Additional archived protocols

Constants below are verified from source. Table values are verified from the named CSVs unless a cell says otherwise.

### Storage capacity

Module: `experiments/storage_capacity.py`.

| Item | Value |
|---|---|
| Lengths | `8, 12, 16` |
| `m` | `1..n`, except `n=16` uses `1..10, 12, 14, 16` |
| Trials | `5` seeds `0..4`, repeated for every `(n, m)` |
| Noise | Fixed-count, `p=0.15` |
| Hopfield | Synchronous, 10 steps |
| QAM | `train_ry_linear_qam`, RY encoding, depth `min(8, n)`, 100 steps, lr `0.05`, `return_tuple=True` |
| Interval | Bootstrap mean CI, 2000 resamples |
| Test | Wilcoxon; failure stored as empty / `None` |
| Critical ratio | First `m/n` with mean full accuracy `< 0.75` |

`results/storage_capacity/summary/experiment_summary.txt` records the same `n_bits_list`, `trials_per_m`, `train_steps`, `recall_noise`, and `bootstrap_samples`.

Archived critical ratios (`phase_transition.csv`):

| Model | n | Critical capacity ratio |
|---|---|---|
| Hopfield | 8 | 0.375 |
| Hopfield | 12 | 0.3333333333333333 |
| Hopfield | 16 | 0.1875 |
| QAM | 8 | 0.25 |
| QAM | 12 | 0.16666666666666666 |
| QAM | 16 | 0.125 |

Example rows from `storage_capacity_sweep.csv`, `n=8`, `m=1`: both models have full and bit accuracy `1.0`. The QAM note is `Variational QAM | depth=8 | steps=100 | lr=0.05`. The file has 66 data rows.

Archived log timing (the log's own tqdm lines; output paths in that log are `results_storage_capacity/`, which is the pre-reorganization layout):

| Sweep | Recorded tqdm total |
|---|---|
| `n=8` (8 values of `m`) | 6 min 26 s |
| `n=12` (12 values of `m`) | 29 min 50 s |
| `n=16` (13 values of `m`) | 2 h 41 min 54 s |

Command: `python -m experiments.storage_capacity`. That command would overwrite the summary CSV, the phase-transition CSV, the Wilcoxon CSV, the summary text, and the plots.

### Circuit depth

Module: `experiments/circuit_depth.py`. QAM only. Cases `(8, 4)`, `(12, 8)`, `(16, 12)`. Depths `1, 2, 4, 6, 8, 10, 12`. Five seeds `0..4`. Same RY-linear trainer, `return_tuple=False`, 100 steps, lr `0.05`, fixed-count noise `0.15`.

`results/circuit_depth/summary/depth_scaling.csv` has 21 data rows. Archived means:

| n | m | depth | full accuracy | bit accuracy | mean train seconds | parameters |
|---|---|---|---|---|---|---|
| 8 | 4 | 1 | 0.0 | 0.55625 | 1.5498682022094727 | 8 |
| 8 | 4 | 8 | 0.0 | 0.65625 | 8.451866054534912 | 64 |
| 8 | 4 | 12 | 0.0 | 0.65625 | 11.930528259277343 | 96 |
| 12 | 8 | 1 | 0.0 | 0.54375 | 6.065131378173828 | 12 |
| 12 | 8 | 12 | 0.0 | 0.6145833333333334 | 52.39401812553406 | 144 |
| 16 | 12 | 1 | 0.0 | 0.5208333333333333 | 40.104098320007324 | 16 |
| 16 | 12 | 12 | 0.0 | 0.6104166666666666 | 356.19185457229617 | 192 |

The full 21-row file is the archive. Intermediate depths are in that CSV and are omitted from this excerpt. Full-pattern accuracy is `0.0` in every row except `n=8`, `m=4`, `depth=6`, where it is `0.1`.

Command: `python -m experiments.circuit_depth`.

### Convergence

Module: `experiments/convergence.py`. Same `(n, m)` pairs. Seeds `0, 1, 2`. Depth `min(8, n)`. 300 Adam steps, lr `0.05`. No corruption and no recall metric.

`results/convergence/summary/convergence_summary.csv`:

| n | m | depth | loss step 0 | loss step 100 | loss final | post-100 improvement % |
|---|---|---|---|---|---|---|
| 8 | 4 | 8 | 2.0816409734878945 | 0.7917833103071069 | 0.791667371393085 | 0.014642758001173631 |
| 12 | 8 | 8 | 1.9143786264215745 | 0.9120701166202823 | 0.9097222222402976 | 0.25742476781114665 |
| 16 | 12 | 8 | 2.017872822713252 | 0.9438350252731277 | 0.9218927927750219 | 2.324795320215641 |

Each raw file `results/convergence/raw/convergence_n{n}_m{m}.csv` has 301 rows (`step`, `mean_loss`, `std_loss`). `convergence.log` prints per-seed losses. The summary means are the archive to cite; the log is consistent with three seeds being averaged (for example the three `n=8` step-0 losses in the log are `2.248121`, `1.748760`, and `2.248042`).

Command: `python -m experiments.convergence`.

### Scaling

Module: `experiments/scaling.py`.

| Item | Value |
|---|---|
| Lengths | `4, 6, 8, 12, 16` |
| Noise | `0.0, 0.1, 0.2, 0.3` |
| Trials | `30` |
| Seed | `123 + trial_id + n_bits * 100` on PennyLane NumPy |
| Stored patterns | `min(max(1, n // 2), 6)` |
| Hopfield | `HopfieldAsyncRandom`, 10 random-order sweeps |
| QAM | Experiment-local Rot ansatz, RY encoding, depth `min(6, n)`, 200 steps, minibatch 2, early stop |

Verified from `results/scaling/raw/scaling_sweep.csv`: 1200 data rows; `m` is 2, 3, 4, 6, 6 for `n` of 4, 6, 8, 12, 16; 30 distinct `trial_id` values; noise levels `0.0, 0.1, 0.2, 0.3`.

Mean accuracy at `n=4`, `m=2`, `noise_p=0.0`, computed from that CSV in this audit: Hopfield full `0.6666666666666666`, bit `0.9166666666666666`; QAM full `0.35`, bit `0.7333333333333333`. The first raw row (`trial_id=0`, noise `0`) has full accuracy `0.5` and bit accuracy `0.875` for both models. Hopfield parameter count on that row is `16`. QAM parameter count is `48` (`depth 3 × 4 qubits × 3 Rot angles`).

Command: `python -m experiments.scaling`. Figures currently on disk match `scripts/plot_scaling.py`. The PNG names written at the end of `experiments/scaling.py` are not in `results/scaling/plots/`.

### Statistical significance

Module: `experiments/statistical_significance.py`.

| Item | Value |
|---|---|
| Lengths | `8, 12, 16` |
| Capacity ratios | `0.2, 0.3, 0.4, 0.5` |
| Bernoulli noise | `0.1, 0.2` |
| Seeds | `0 .. 29` |
| Hopfield | Synchronous, 10 steps |
| QAM | PauliX encoding, RY + linear CNOT, depth 10, 100 epochs, lr `0.02`, init `N(0, 0.1)` |
| Bootstrap | 5000 resamples of the paired difference QAM minus Hopfield |
| Wilcoxon failure | p-value stored as `1.0` |

`statistical_rows.csv` has 720 data rows. `statistical_summary.csv` has 24. Example, verified from the first summary row (`n=8`, capacity `0.2`, noise `0.1`): mean Hopfield full `1.0`, mean QAM full `0.3333333333333333`, mean difference `-0.6666666666666667`, full-accuracy percentile interval `[-0.8333333333333334, -0.5]`, Wilcoxon p `7.74421643104407e-06`.

`cliffs_delta_table.csv` is produced by `python -m scripts.cliffs_delta`, not by the experiment `main`. First row: `n=8`, capacity `0.2`, noise `0.1`, `delta_full=0.667` (Large), `delta_bit=0.667` (Large). Positive delta means Hopfield scores tend to be larger.

Command: `python -m experiments.statistical_significance`.

### Large inference grid

Module: `experiments/noise_robustness_inference.py`.

| Item | Value |
|---|---|
| Lengths | `8, 12, 16, 24, 32, 48, 64, 96, 128` |
| Capacity ratios | `0.1, 0.25, 0.5, 0.75, 1.0` |
| Noise | `numpy.linspace(0.0, 0.5, 16)` |
| Trials per cell | `300` |
| Seed | `42 + trial_id + int(n*1000) + int(capacity*100) + int(noise*1000)` |
| Hopfield | Hebbian weights, 20 sequential async sweeps, energy recorded |
| Column `model="qam"` | `nearest_neighbor_recall`. No circuit. |
| Tasks | `216000`, matching the archived log line `Progress: 216000/216000` |
| Raw rows | `432000` (two model rows per task), file size 50,153,034 bytes |

Summary file: 1440 data rows. Bootstrap inside `aggregate_results` uses 200 unseeded resamples, so those percentile columns are not pinned by the trial seed.

Critical-noise archives:

| File | Rule | Rows |
|---|---|---|
| `results/noise_robustness_inference/summary/critical_noise_thresholds.csv` | Last noise with mean full accuracy `≥ 0.50` | 90 |
| `results/noise_robustness_inference/paper_figures/critical_noise_thresholds.csv` | First noise with mean full accuracy `< 0.90`; omit the group if accuracy never falls below 0.90 | 87 |

Example from the 0.50 table: Hopfield, `n=8`, capacity `0.1`, threshold `0.5`; capacity `0.25`, threshold `0.3`; capacity `0.5`, threshold `0.0333333333333333`. Example from the 0.90 table: Hopfield, `n=8`, capacity `0.1`, critical noise `0.3`.

The log's completion paths are `noise_robustness_inference_results/noise_inference_rows.csv` and `noise_inference_summary.csv`. The files now live under `results/noise_robustness_inference/`. Wall-clock of that run is not in the log.

Command: `python -m experiments.noise_robustness_inference`.

## Reproduction commands

Install, from the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

List experiments without running them:

```bash
python -m scripts.run_all_experiments
```

Run one protocol (overwrites that protocol's files under `results/`):

```bash
python -m experiments.noise_robustness_sweep
python -m experiments.storage_capacity
python -m experiments.circuit_depth
python -m experiments.convergence
python -m experiments.scaling
python -m experiments.statistical_significance
python -m experiments.noise_robustness_inference
```

Rebuild figures and derived tables from the archived CSVs (these also overwrite their outputs):

```bash
python -m scripts.plot_noise_robustness_sweep
python -m scripts.plot_scaling
python -m scripts.plot_critical_noise_threshold
python -m scripts.plot_inference_paper_figures
python -m scripts.plot_noise_robustness_inference
python -m scripts.cliffs_delta
python -m scripts.plot_bootstrap_distribution
```

`plot_critical_noise_threshold` rewrites `summary/critical_noise_thresholds.csv`. `cliffs_delta` rewrites `statistical_significance/summary/cliffs_delta_table.csv`. `plot_inference_paper_figures` rewrites `paper_figures/critical_noise_thresholds.csv`. `plot_bootstrap_distribution` draws 5000 unseeded bootstrap means from the pooled raw columns and writes `bootstrap_full_accuracy_distribution.png`.

## Missing information

- No fresh scientific run was performed, so equality between today's code path and the archived CSV bytes is not demonstrated. Logs still name `results_storage_capacity/` and `noise_robustness_inference_results/`.
- Package versions and the machine that produced the CSVs are not stored in the result files. Train-time columns cannot be checked against this machine without a new run.
- The noise sweep has no successful run log and no per-trial file.
- Patterns, Hopfield weights, and QAM parameters were not saved.
- Inference summary bootstrap intervals and the bootstrap histogram script do not set a seed.
- Which archived PNG came from which historical script is unknown when the current tree has no matching `savefig` (`bitwise_noise.png`, `full_pattern_noise.png`).
