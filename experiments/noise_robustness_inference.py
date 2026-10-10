"""Large-n noise/capacity grid.

Hopfield uses sequential async updates. CSV rows with ``model="qam"`` are
Hamming nearest neighbor, not the variational circuit.
"""

import os
import time
import numpy as np
import pandas as pd
import multiprocessing as mp

from src.hopfield import hopfield_hebbian_weights, hopfield_async_sequential_recall
from src.qam import nearest_neighbor_recall
from src.metrics import compute_metrics
from src.paths import results_subdir


# ==============================
# CONFIG
# ==============================
N_BITS_LIST = [8, 12, 16, 24, 32, 48, 64, 96, 128]
CAPACITY_RATIOS = [0.1, 0.25, 0.5, 0.75, 1.0]
NOISE_LEVELS = np.linspace(0.0, 0.5, 16)
RANDOM_SEED_BASE = 42
TRIALS_PER_CONFIG = 300
EXPERIMENT_NAME = "noise_robustness_inference"


def seed_all(seed):
    np.random.seed(seed)


# ==============================
# CORE EXPERIMENT
# ==============================
def run_trial(task):
    n_bits, capacity_ratio, noise_p, trial_id = task

    seed = (
        RANDOM_SEED_BASE
        + trial_id
        + int(n_bits * 1000)
        + int(capacity_ratio * 100)
        + int(noise_p * 1000)
    )
    seed_all(seed)

    n_patterns = max(1, int(n_bits * capacity_ratio))
    patterns = np.random.choice([-1, 1], size=(n_patterns, n_bits))
    true_pattern = patterns[np.random.randint(0, n_patterns)].copy()

    noise_mask = np.random.rand(n_bits) < noise_p
    noisy_input = true_pattern.copy()
    noisy_input[noise_mask] *= -1
    input_bit_accuracy = np.mean(noisy_input == true_pattern)
    input_hamming_error = np.mean(noisy_input != true_pattern)

    # ==============================
    # Hopfield Network
    # ==============================
    t0 = time.time()

    W = hopfield_hebbian_weights(patterns)
    hopfield_pred, final_energy, energy_drop = hopfield_async_sequential_recall(
        W, noisy_input, steps=20
    )
    hopfield_time = time.time() - t0

    # ==============================
    # QAM proxy (nearest neighbor)
    # ==============================
    t0 = time.time()
    qam_pred = nearest_neighbor_recall(patterns, noisy_input)
    qam_time = time.time() - t0

    hop_bit, hop_full, hop_hamming_error = compute_metrics(hopfield_pred, true_pattern)
    qam_bit, qam_full, qam_hamming_error = compute_metrics(qam_pred, true_pattern)

    hop_recovery_gain = input_hamming_error - hop_hamming_error
    qam_recovery_gain = input_hamming_error - qam_hamming_error

    return [
        {
            "model": "hopfield",
            "n_bits": n_bits,
            "capacity_ratio": capacity_ratio,
            "noise_p": noise_p,
            "trial": trial_id,
            "bit_accuracy": hop_bit,
            "full_accuracy": hop_full,
            "hamming_error": hop_hamming_error,
            "input_bit_accuracy": input_bit_accuracy,
            "input_hamming_error": input_hamming_error,
            "recovery_gain": hop_recovery_gain,
            "runtime": hopfield_time,
            "energy": final_energy,
            "energy_drop": energy_drop,
        },
        {
            "model": "qam",
            "n_bits": n_bits,
            "capacity_ratio": capacity_ratio,
            "noise_p": noise_p,
            "trial": trial_id,
            "bit_accuracy": qam_bit,
            "full_accuracy": qam_full,
            "hamming_error": qam_hamming_error,
            "input_bit_accuracy": input_bit_accuracy,
            "input_hamming_error": input_hamming_error,
            "recovery_gain": qam_recovery_gain,
            "runtime": qam_time,
            "energy": np.nan,
            "energy_drop": np.nan,
        },
    ]


# ==============================
# AGGREGATION
# ==============================
def aggregate_results(df):
    grouped = df.groupby(["model", "n_bits", "capacity_ratio", "noise_p"])

    def bootstrap_ci(series, n_boot=200):
        samples = np.random.choice(series, (n_boot, len(series)), replace=True)
        means = samples.mean(axis=1)
        return np.percentile(means, [2.5, 97.5])

    summary = grouped.agg(
        bit_acc_mean=("bit_accuracy", "mean"),
        bit_acc_std=("bit_accuracy", "std"),
        full_acc_mean=("full_accuracy", "mean"),
        full_acc_std=("full_accuracy", "std"),
        hamming_error_mean=("hamming_error", "mean"),
        hamming_error_std=("hamming_error", "std"),
        input_bit_acc_mean=("input_bit_accuracy", "mean"),
        input_hamming_error_mean=("input_hamming_error", "mean"),
        recovery_gain_mean=("recovery_gain", "mean"),
        recovery_gain_std=("recovery_gain", "std"),
        runtime_mean=("runtime", "mean"),
        runtime_std=("runtime", "std"),
        energy_mean=("energy", "mean"),
        energy_std=("energy", "std"),
        energy_drop_mean=("energy_drop", "mean"),
        energy_drop_std=("energy_drop", "std"),
    ).reset_index()

    # bootstrap CI (percentile, not Gaussian)
    bit_ci_low = []
    bit_ci_high = []
    full_ci_low = []
    full_ci_high = []

    for _, group in grouped:
        b_low, b_high = bootstrap_ci(group["bit_accuracy"].values)
        f_low, f_high = bootstrap_ci(group["full_accuracy"].values)

        bit_ci_low.append(b_low)
        bit_ci_high.append(b_high)
        full_ci_low.append(f_low)
        full_ci_high.append(f_high)

    summary["bit_ci_low"] = bit_ci_low
    summary["bit_ci_high"] = bit_ci_high
    summary["full_ci_low"] = full_ci_low
    summary["full_ci_high"] = full_ci_high

    # parametric CI (kept for comparison with bootstrap)
    summary["bit_acc_ci95"] = 1.96 * summary["bit_acc_std"] / np.sqrt(TRIALS_PER_CONFIG)
    summary["full_acc_ci95"] = 1.96 * summary["full_acc_std"] / np.sqrt(TRIALS_PER_CONFIG)
    summary["runtime_ci95"] = (
        1.96 * summary["runtime_std"] / np.sqrt(TRIALS_PER_CONFIG)
    )

    summary["recovery_gain_ci95"] = (
        1.96 * summary["recovery_gain_std"] / np.sqrt(TRIALS_PER_CONFIG)
    )

    summary["hamming_error_ci95"] = (
        1.96 * summary["hamming_error_std"] / np.sqrt(TRIALS_PER_CONFIG)
    )

    return summary


def main():
    print("Starting noise robustness experiment...")

    raw_dir = results_subdir(EXPERIMENT_NAME, "raw")
    summary_dir = results_subdir(EXPERIMENT_NAME, "summary")

    tasks = [
        (n, c, p, t)
        for n in N_BITS_LIST
        for c in CAPACITY_RATIOS
        for p in NOISE_LEVELS
        for t in range(TRIALS_PER_CONFIG)
    ]

    results = []

    workers = min(12, os.cpu_count() or 1)
    print(f"Using {workers} workers")

    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_trial, tasks, chunksize=50), start=1):
            results.extend(res)

            if i % 50 == 0 or i == len(tasks):
                print(f"Progress: {i}/{len(tasks)}")

    df = pd.DataFrame(results)

    rows_path = raw_dir / "noise_inference_rows.csv"
    df.to_csv(rows_path, index=False)

    summary = aggregate_results(df)

    summary_path = summary_dir / "noise_inference_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nExperiment Summary")
    print(f"Total configurations: {len(N_BITS_LIST) * len(CAPACITY_RATIOS) * len(NOISE_LEVELS)}")
    print(f"Trials per configuration: {TRIALS_PER_CONFIG}")
    print(f"Total trials executed: {len(tasks)}")

    print("\nCompleted.")
    print(rows_path)
    print(summary_path)


if __name__ == "__main__":
    mp.freeze_support()
    main()
