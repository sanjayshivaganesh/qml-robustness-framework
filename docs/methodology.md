# Experiment notes

How the runs actually differ. Numbers from different CSVs are only comparable when the circuit, noise process, Hopfield update, and hyperparameters match.

## Hopfield

Hebbian storage is the same everywhere:

```
W ← (1/m) Σ_μ ξ^μ (ξ^μ)^T ,    diag(W) = 0
```

Recall is not:

| Experiment | Recall |
|---|---|
| Noise sweep, storage capacity, statistical significance | Synchronous `x ← sign(Wx)`, 10 steps; zeros → `+1` |
| Scaling | Async random order, 10 sweeps; `x_i ← +1` if `W_i · x ≥ 0` else `−1` |
| Large-scale inference | Async sequential index order, 20 sweeps; zeros → `+1`; energy `E = −½ x^T W x` |

## QAM circuits

Training: Adam on MSE between Pauli-Z expectations and the ±1 target. `sign` is used at recall only.

| Experiment | Encoding | Ansatz | Training |
|---|---|---|---|
| Noise robustness sweep | RX(πx), RY(πx/2) | `Rot` + linear CNOT; params `(depth, n, 3)`; depth `min(6, n)` | 300 steps, lr=0.05. `__init__` reseeds PennyLane RNG to 42 |
| Storage capacity, circuit depth, convergence | RY(πx) | RY + linear CNOT; params `(depth, n)`; depth `min(8, n)` except when depth is swept | 100 steps (capacity, depth), 300 (convergence); lr=0.05 |
| Scaling | RY(πx) | `Rot` + linear CNOT; depth `min(6, n)` | 200 steps, minibatch 2, stop if loss change `< 1e-4` for 15 steps, lr=0.05 |
| Statistical significance | `PauliX` where bit is `−1` | RY + linear CNOT; depth 10 | 100 epochs, lr=0.02; init `N(0, 0.1)` |

These are different models. A `"QAM"` row in one CSV is not the same object as in another.

### Inference experiment: nearest neighbor

`experiments/noise_robustness_inference.py` does not run a quantum circuit. After Hopfield recall it returns the stored pattern with smallest Hamming distance to the query and writes `model="qam"`. That was so the 128-bit, 300-trial grid was feasible. Treat those rows as a classical baseline.

## Corruption

| Method | Rule | Used by |
|---|---|---|
| Bernoulli | each bit flips with probability `p` | noise sweep, scaling, stats, inference |
| Fixed count | flip `max(1, int(n * p))` bits | capacity and depth (`p=0.15`) |

The `max(1, ...)` means `p=0` would still flip one bit. Capacity and depth always use `p=0.15`, so that branch is unused there; the helper still matches the original code.

## Metrics

- Bit accuracy: matching bits.
- Full-pattern accuracy: 1 iff the whole vector matches.
- Recovery gain (inference): input Hamming error minus output Hamming error.
- Gaussian 95% CI: `1.96 * std / sqrt(n)` (noise sweep; some scaling plots).
- Bootstrap CI: 200 (inference), 2000 (capacity), 5000 (stats). The implementations differ (loop vs vectorized vs paired-difference).
- Wilcoxon signed-rank: capacity and statistical-significance experiments.
- Cliff's delta: `scripts/cliffs_delta.py`. Positive δ means Hopfield scores tend to be larger than QAM.

## Designs

**Noise robustness sweep.** `n ∈ {4,6,8,10}`, Bernoulli `p ∈ {0,0.1,0.2,0.3,0.4}`, `m = max(1, int(0.3 n))`, 30 seeds, recall on every stored pattern.

**Large-scale inference.** `n ∈ {8,12,16,24,32,48,64,96,128}`, capacity `{0.1,0.25,0.5,0.75,1.0}`, 16 noise levels in `[0, 0.5]`, 300 trials per cell, one random target per trial. Hopfield: 20 sequential async steps. `qam` column: nearest neighbor.

Two critical-noise tables, on purpose:

- `results/noise_robustness_inference/summary/critical_noise_thresholds.csv` — last noise with mean full accuracy `≥ 0.50`
- `results/noise_robustness_inference/paper_figures/critical_noise_thresholds.csv` — first noise with mean full accuracy `< 0.90`

**Storage capacity.** `n ∈ {8,12,16}`, `m` up to `n` (for `n=16`, `m ∈ 1..10, 12, 14, 16` to keep runtime down), 5 trials, 15% fixed-count noise, bootstrap 2000, Wilcoxon vs QAM.

**Scaling.** `n ∈ {4,6,8,12,16}`, `m = min(max(1, n//2), 6)`, Bernoulli `{0,0.1,0.2,0.3}`, 30 trials, seed `123 + trial_id + n*100`. QAM trained once per trial, then evaluated at every noise level.

**Circuit depth.** `(n,m) ∈ {(8,4),(12,8),(16,12)}`, `depth ∈ {1,2,4,6,8,10,12}`, 5 seeds, same RY-linear QAM and 15% noise as capacity.

**Convergence.** Same `(n,m)` pairs, 300 Adam steps, 3 seeds. Compares loss at step 100 vs the end, as a check on `train_steps=100` in the capacity sweep.

**Statistical significance.** `n ∈ {8,12,16}`, capacity `{0.2,0.3,0.4,0.5}`, noise `{0.1,0.2}`, 30 seeds, paired bootstrap on QAM − Hopfield, Wilcoxon.

## Implementation notes

Left as they are so numbers stay aligned with the original scripts:

1. Inference `qam` = nearest neighbor.
2. QAM circuits and Hopfield updates differ by experiment.
3. Noise-sweep `QAM.__init__` reseeds PennyLane numpy to 42 after the trial seed, so QAM parameters are less trial-randomized than Hopfield.
4. Scaling uses PennyLane numpy for patterns **and** Hopfield permutation order (same RNG stream).
5. `corrupt_fixed_count` uses `max(1, int(n * p))`.
6. Wilcoxon used a bare `except:` originally; the current scripts catch `Exception` and still return `None` / `1.0`.
7. Timing is not defined the same way everywhere (per bit vs per pattern vs whole recall loop).
