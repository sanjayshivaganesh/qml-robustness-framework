"""Hopfield vs PauliX-encoding QAM: bootstrap CIs and Wilcoxon tests.

Varies ``n_bits``, capacity ratio, and Bernoulli noise.
"""

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from multiprocessing import Pool, cpu_count
from scipy.stats import wilcoxon
import time
import matplotlib.pyplot as plt

from src.hopfield import Hopfield
from src.data_generation import corrupt_bernoulli
from src.metrics import pattern_accuracy_pair
from src.paths import results_subdir

# =========================
# CONFIG
# =========================
N_BITS_LIST = [8, 12, 16]
CAPACITY_RATIOS = [0.2, 0.3, 0.4, 0.5]
CORRUPTION_LEVELS = [0.1, 0.2]
SEEDS = list(range(30))
BOOTSTRAP_SAMPLES = 5000

DEPTH = 10
EPOCHS = 100
LR = 0.02

EXPERIMENT_NAME = "statistical_significance"


# =========================
# QAM (PauliX encoding)
# =========================
def build_qam(n_bits):
    dev = qml.device("default.qubit", wires=n_bits)

    @qml.qnode(dev)
    def circuit(params, x):
        for i, bit in enumerate(x):
            if bit == -1:
                qml.PauliX(wires=i)

        for d in range(params.shape[0]):
            for i in range(n_bits):
                qml.RY(params[d, i], wires=i)
            for i in range(n_bits - 1):
                qml.CNOT(wires=[i, i+1])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_bits)]

    return circuit


def train_qam(patterns, circuit):
    n = patterns.shape[1]

    params = pnp.array(
        np.random.normal(0, 0.1, (DEPTH, n)),
        requires_grad=True
    )

    opt = qml.AdamOptimizer(LR)

    for _ in range(EPOCHS):
        def loss_fn(w):
            loss = 0
            for p in patterns:
                out = pnp.array(circuit(w, p))
                loss += pnp.mean((out - p) ** 2)
            return loss / len(patterns)
        params = opt.step(loss_fn, params)

    return params


def qam_recall(params, circuit, x):
    out = np.array(circuit(params, x), dtype=float)
    r = np.sign(out)
    r[r == 0] = 1
    return r


# =========================
# SINGLE TRIAL
# =========================
def run_trial(args):
    n_bits, cap, noise, seed = args
    print(f"[RUNNING] n_bits={n_bits} | cap={cap} | noise={noise} | seed={seed}")
    np.random.seed(seed)
    pnp.random.seed(seed)

    m = max(1, int(cap * n_bits))
    patterns = np.random.choice([-1, 1], size=(m, n_bits))

    results = []
    for noise in [noise]:
        # Hopfield
        hop = Hopfield(n_bits)
        hop.train(patterns)
        hf_full, hf_bit = [], []
        for p in patterns:
            r = hop.recall(corrupt_bernoulli(p, noise))
            f, b = pattern_accuracy_pair(r, p)
            hf_full.append(f)
            hf_bit.append(b)

        # QAM
        circuit = build_qam(n_bits)
        params = train_qam(patterns, circuit)

        q_full, q_bit = [], []
        for p in patterns:
            r = qam_recall(params, circuit, corrupt_bernoulli(p, noise))
            f, b = pattern_accuracy_pair(r, p)
            q_full.append(f)
            q_bit.append(b)

        results.append({
            "n_bits": n_bits,
            "capacity_ratio": cap,
            "noise": noise,
            "seed": seed,
            "hopfield_full": np.mean(hf_full),
            "hopfield_bit": np.mean(hf_bit),
            "qam_full": np.mean(q_full),
            "qam_bit": np.mean(q_bit),
        })

    return results[0]


def bootstrap_diff(a, b, n=BOOTSTRAP_SAMPLES):
    paired_diff = a - b
    diffs = []

    for _ in range(n):
        idx = np.random.randint(0, len(paired_diff), len(paired_diff))
        sample = paired_diff[idx]
        diffs.append(np.mean(sample))

    return np.array(diffs)


def main():
    raw_dir = results_subdir(EXPERIMENT_NAME, "raw")
    summary_dir = results_subdir(EXPERIMENT_NAME, "summary")
    plots_dir = results_subdir(EXPERIMENT_NAME, "plots")

    overall_start = time.time()
    print("\n=== STARTING STATISTICAL SIGNIFICANCE EXPERIMENT ===")

    print("Preparing experiment tasks...")
    tasks = [
        (n, c, noise, s)
        for n in N_BITS_LIST
        for c in CAPACITY_RATIOS
        for noise in CORRUPTION_LEVELS
        for s in SEEDS
    ]

    print(f"Launching multiprocessing pool with {cpu_count()} workers...")

    with Pool(cpu_count()) as pool:
        rows = pool.map(run_trial, tasks)

    print("All experiment trials completed.")
    df = pd.DataFrame(rows)
    df.to_csv(raw_dir / "statistical_rows.csv", index=False)

    print("Computing bootstrap confidence intervals and Wilcoxon tests...")
    summaries = []

    for (n, c, noise), g in df.groupby(["n_bits", "capacity_ratio", "noise"]):
        print(f"[ANALYSIS] n_bits={n} | cap={c} | noise={noise}")

        hf_full = g["hopfield_full"].values
        q_full = g["qam_full"].values

        hf_bit = g["hopfield_bit"].values
        q_bit = g["qam_bit"].values

        full_diffs = bootstrap_diff(q_full, hf_full)
        full_ci_low, full_ci_high = np.percentile(full_diffs, [2.5, 97.5])

        try:
            full_p = wilcoxon(q_full, hf_full).pvalue
        except Exception:
            full_p = 1.0

        bit_diffs = bootstrap_diff(q_bit, hf_bit)
        bit_ci_low, bit_ci_high = np.percentile(bit_diffs, [2.5, 97.5])

        try:
            bit_p = wilcoxon(q_bit, hf_bit).pvalue
        except Exception:
            bit_p = 1.0

        summaries.append({
            "n_bits": n,
            "capacity_ratio": c,
            "noise": noise,

            "mean_hopfield_full": hf_full.mean(),
            "mean_qam_full": q_full.mean(),
            "mean_diff_full": q_full.mean() - hf_full.mean(),
            "full_ci_2p5": full_ci_low,
            "full_ci_97p5": full_ci_high,
            "full_p_value": full_p,

            "mean_hopfield_bit": hf_bit.mean(),
            "mean_qam_bit": q_bit.mean(),
            "mean_diff_bit": q_bit.mean() - hf_bit.mean(),
            "bit_ci_2p5": bit_ci_low,
            "bit_ci_97p5": bit_ci_high,
            "bit_p_value": bit_p,
        })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(summary_dir / "statistical_summary.csv", index=False)

    print("Saving plots...")

    plt.figure(figsize=(9, 6))

    for noise in CORRUPTION_LEVELS:
        sub = summary_df[summary_df["noise"] == noise]

        grouped = sub.groupby("capacity_ratio").agg({
            "mean_diff_full": "mean",
            "full_ci_2p5": "mean",
            "full_ci_97p5": "mean"
        }).reset_index()

        plt.errorbar(
            grouped["capacity_ratio"],
            grouped["mean_diff_full"],
            yerr=[
                grouped["mean_diff_full"] - grouped["full_ci_2p5"],
                grouped["full_ci_97p5"] - grouped["mean_diff_full"]
            ],
            marker="o",
            capsize=4,
            label=f"Noise={noise}"
        )

    plt.axhline(0, linestyle="--")
    plt.xlabel("Capacity Ratio")
    plt.ylabel("QAM - Hopfield Full Recall Accuracy")
    plt.title("Full Recall Accuracy Gap with 95% Confidence Intervals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "full_recall_gap_ci.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 6))

    for noise in CORRUPTION_LEVELS:
        sub = summary_df[summary_df["noise"] == noise]

        grouped = sub.groupby("capacity_ratio").agg({
            "mean_diff_bit": "mean",
            "bit_ci_2p5": "mean",
            "bit_ci_97p5": "mean"
        }).reset_index()

        plt.errorbar(
            grouped["capacity_ratio"],
            grouped["mean_diff_bit"],
            yerr=[
                grouped["mean_diff_bit"] - grouped["bit_ci_2p5"],
                grouped["bit_ci_97p5"] - grouped["mean_diff_bit"]
            ],
            marker="s",
            capsize=4,
            label=f"Noise={noise}"
        )

    plt.axhline(0, linestyle="--")
    plt.xlabel("Capacity Ratio")
    plt.ylabel("QAM - Hopfield Bit Accuracy")
    plt.title("Bit Accuracy Gap with 95% Confidence Intervals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "bit_accuracy_gap_ci.png", dpi=300)
    plt.close()

    total_time = time.time() - overall_start

    print("\n=== EXPERIMENT COMPLETE ===")
    print("Statistical summary saved")
    print(f"Total runtime: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
