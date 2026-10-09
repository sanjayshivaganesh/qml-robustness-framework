# Hybrid Quantum-Classical Associative Memory

Code and saved results for a comparison of classical Hopfield networks and hybrid variational quantum associative memory (QAM) on bipolar pattern recall.

The experiments look at noise, storage load, pattern length, and circuit depth. This is **not** a claim of practical quantum advantage. The QAM circuits also differ across experiments, so those results should not be pooled without checking the setup.

One large experiment is a special case: `experiments/noise_robustness_inference.py` labels a column `qam`, but that column is a **classical nearest-neighbor match on Hamming distance**, not the variational circuit. Details are in [`docs/methodology.md`](docs/methodology.md).

## Research question

How does a hybrid variational QAM compare with a Hopfield network on:

- recall from bit-flip noise
- storage capacity (accuracy vs number of stored patterns)
- scaling with pattern length
- sensitivity to circuit depth
- statistical differences (bootstrap CIs, Wilcoxon tests)

Hopfield: discrete bipolar network, Hebbian weights. Most runs use synchronous updates; scaling and the large inference study use asynchronous updates.

QAM: a parameterized circuit trained with Adam on mean-squared Pauli-Z readout error (`default.qubit` CPU simulator). Recall takes the sign of the measured vector. This is a hybrid model on a classical simulator, not a hardware speedup.

## Experiments

| Module | What it varies |
|---|---|
| `experiments.noise_robustness_sweep` | bit-flip rate, small `n` |
| `experiments.noise_robustness_inference` | noise × capacity × large `n`; Hopfield vs **NN proxy** |
| `experiments.storage_capacity` | number of stored patterns `m` |
| `experiments.scaling` | pattern length |
| `experiments.circuit_depth` | ansatz depth |
| `experiments.convergence` | QAM training loss vs step |
| `experiments.statistical_significance` | bootstrap / Wilcoxon on accuracy gaps |

QAM is not one object in this repo. The noise sweep uses a Rot ansatz; capacity / depth / convergence share an RY-linear circuit; scaling uses Rot + minibatch and early stopping; statistical significance uses PauliX encoding. Inference uses nearest neighbor. See `docs/methodology.md` for encodings, update rules, and hyperparameter tables.

## Install

Python 3.10+. From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Tests:

```bash
pip install -e ".[dev]"
pytest
```

## Running experiments

```bash
python -m experiments.noise_robustness_sweep
python -m experiments.noise_robustness_inference
python -m experiments.storage_capacity
python -m experiments.scaling
python -m experiments.circuit_depth
python -m experiments.convergence
python -m experiments.statistical_significance
```

```bash
python -m scripts.run_all_experiments
python -m scripts.run_all_experiments --only convergence
```

`noise_robustness_inference` and `statistical_significance` are the slow ones. Prefer `python -m` if an experiment uses multiprocessing. Re-running overwrites that experiment's CSVs.

## Results

Saved outputs live under `results/<experiment>/` (raw trials, summaries, plots). Paths are resolved from the repo root via `src/paths.py`, independent of cwd.

The files currently in `results/` are the original saved outputs, not a fresh re-run.

The inference study has two critical-noise tables with **different** definitions (accuracy floor 0.50 vs 0.90). Do not treat them as the same analysis.

Plot from existing CSVs:

```bash
python -m scripts.plot_noise_robustness_sweep
python -m scripts.plot_noise_robustness_inference
python -m scripts.plot_scaling
python -m scripts.plot_inference_paper_figures
python -m scripts.plot_critical_noise_threshold
python -m scripts.cliffs_delta
python -m scripts.plot_bootstrap_distribution
```

## Shared code

```python
from src.hopfield import Hopfield
from src.qam import train_ry_linear_qam, qam_recall
```

`src.qam` is the RY-linear circuit (functions, not a `QAM` class). Other circuits stay in their experiment files.

Main dependencies: NumPy, pandas, SciPy, PennyLane, matplotlib, seaborn, tqdm, psutil. Pins are in `pyproject.toml` / `requirements.txt`.

## Limits

These runs compare a Hopfield attractor memory with specific hybrid reconstruction circuits (plus one NN baseline). A significant accuracy gap is not quantum advantage. The simulator is not hardware. Depth, encoding, and training budget were not held fixed across every experiment, so cross-CSV "QAM" numbers are not interchangeable.

Paper: **Comparing Classical Hopfield Networks and Hybrid Quantum-Classical Models for Associative Memory**.
# qml-robustness-framework
