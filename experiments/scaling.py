"""Pattern-length scaling with Bernoulli noise.

Hopfield: random-order async updates. QAM: Rot ansatz, minibatch, early stop.
``m = min(max(1, n_bits // 2), 6)``.
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import platform

from src.hopfield import HopfieldAsyncRandom as Hopfield
from src.paths import results_subdir

# ------------------------------
# QAM (Rot ansatz, minibatch, early stopping)
# ------------------------------
class QAM:
    def __init__(self, n_bits, depth=3, steps=200, lr=0.05):
        self.n_bits = n_bits
        self.depth = min(depth, n_bits)
        self.steps = steps
        self.lr = lr
        self.params = np.random.randn(self.depth, n_bits, 3) * 0.01
        self.dev = qml.device("default.qubit", wires=n_bits)

    def ansatz(self, params, x):
        for i in range(self.n_bits):
            qml.RY(x[i] * np.pi, wires=i)
        for d in range(self.depth):
            for i in range(self.n_bits):
                qml.Rot(*params[d, i], wires=i)
            for i in range(self.n_bits-1):
                qml.CNOT(wires=[i, i+1])

    def train(self, patterns):
        @qml.qnode(self.dev)
        def circuit(params, x):
            self.ansatz(params, x)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_bits)]

        def loss(params):
            # minibatch of 2, same as the original scaling script
            batch_size = min(2, len(patterns))
            idx = np.random.choice(len(patterns), batch_size, replace=False)

            l = 0
            for i in idx:
                p = patterns[i]
                out = np.array(circuit(params, p))
                l += np.mean((p - out)**2)

            return l / batch_size

        opt = qml.AdamOptimizer(self.lr)
        start = time.time()
        prev_loss = None
        patience_counter = 0

        for step in range(self.steps):
            current_loss = loss(self.params)
            self.params = opt.step(loss, self.params)

            # early stopping if convergence plateaus
            if prev_loss is not None:
                if abs(prev_loss - current_loss) < 1e-4:
                    patience_counter += 1
                else:
                    patience_counter = 0

            if patience_counter >= 15:
                break

            prev_loss = current_loss
        train_time = time.time() - start
        self.circuit = circuit
        return train_time

    def recall(self, x):
        out = np.sign(np.array(self.circuit(self.params, x)))
        out[out==0] = 1
        return out

    @property
    def param_count(self):
        return self.params.size

# ------------------------------
# Experiment Config
# ------------------------------
n_bits_list = [4, 6, 8, 12, 16]
noise_levels = np.array([0.0, 0.1, 0.2, 0.3])  # reduced sweep for faster runtime
RANDOM_SEED_BASE = 123
trials = 30  # sufficient statistical power with much lower runtime
EXPERIMENT_NAME = "scaling"


def bootstrap_ci(series, n_boot=200):
    samples = np.random.choice(series, (n_boot, len(series)), replace=True)
    means = samples.mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    raw_dir = results_subdir(EXPERIMENT_NAME, "raw")
    plots_dir = results_subdir(EXPERIMENT_NAME, "plots")
    results = []

    for n_bits in n_bits_list:
        m = min(max(1, n_bits // 2), 6)
        print(f"Running n_bits={n_bits}, m={m}")
        for trial_id in range(trials):
            seed = RANDOM_SEED_BASE + trial_id + n_bits * 100
            np.random.seed(seed)
            patterns = np.random.choice([-1,1], size=(m, n_bits))

            # Hopfield
            hop = Hopfield(n_bits)
            t0 = time.time()
            hop.train(patterns)
            train_time_hop = time.time() - t0

            # QAM (train once per trial, not per noise level)
            qam = QAM(n_bits, depth=min(6, n_bits), steps=200, lr=0.05)
            train_time_qam = qam.train(patterns)

            for noise_p in noise_levels:
                t0 = time.time()
                full_acc_hop = []
                bit_acc_hop = []

                for p in patterns:
                    noisy = p.copy()
                    flip_mask = np.random.rand(n_bits) < noise_p
                    noisy[flip_mask] *= -1
                    recalled = hop.recall(noisy)

                    full_acc_hop.append(int(np.array_equal(recalled, p)))
                    bit_acc_hop.append(np.mean(recalled == p))

                infer_time_hop = (time.time() - t0) / m

                t0 = time.time()
                full_acc_qam = []
                bit_acc_qam = []
                for p in patterns:
                    noisy = p.copy()
                    flip_mask = np.random.rand(n_bits) < noise_p
                    noisy[flip_mask] *= -1
                    recalled = qam.recall(noisy)
                    full_acc_qam.append(int(np.array_equal(recalled, p)))
                    bit_acc_qam.append(np.mean(recalled == p))
                infer_time_qam = (time.time() - t0) / m

                results.append({
                    "model":"Hopfield",
                    "n_bits":n_bits,
                    "m":m,
                    "trial_id":trial_id,
                    "full_acc":np.mean(full_acc_hop),
                    "bit_acc":np.mean(bit_acc_hop),
                    "train_time_s":train_time_hop,
                    "infer_time_s":infer_time_hop,
                    "params_count":n_bits**2,
                    "noise_p": noise_p,
                    "extra_notes":""
                })
                results.append({
                    "model":"QAM",
                    "n_bits":n_bits,
                    "m":m,
                    "trial_id":trial_id,
                    "full_acc":np.mean(full_acc_qam),
                    "bit_acc":np.mean(bit_acc_qam),
                    "train_time_s":train_time_qam,
                    "infer_time_s":infer_time_qam,
                    "params_count":qam.param_count,
                    "noise_p": noise_p,
                    "extra_notes":""
                })

    df = pd.DataFrame(results)

    csv_file = raw_dir / "scaling_sweep.csv"
    df.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")

    import seaborn as sns
    sns.set(style="whitegrid")

    plt.figure(figsize=(8,6))
    for model in ["Hopfield","QAM"]:
        sub = df[df["model"]==model].groupby(["n_bits","noise_p"])["full_acc"].mean().unstack()
        for noise_p in sub.columns:
            plt.plot(sub.index, sub[noise_p], marker="o", label=f"{model} noise={round(noise_p,2)}")
    plt.xlabel("Pattern length n_bits")
    plt.ylabel("Full-pattern accuracy")
    plt.title("Full-pattern accuracy vs n_bits (noise sweep)")
    plt.legend()
    plt.savefig(plots_dir / "full_acc_vs_n_bits.png", dpi=300)

    plt.figure(figsize=(8,6))

    for model in ["Hopfield", "QAM"]:
        sub = (
            df[df["model"] == model]
            .groupby(["n_bits", "noise_p"])["bit_acc"]
            .agg(["mean", "std"])
            .reset_index()
        )

        for noise_p in sorted(sub["noise_p"].unique()):
            noise_sub = sub[sub["noise_p"] == noise_p]

            plt.errorbar(
                noise_sub["n_bits"],
                noise_sub["mean"],
                yerr=1.96 * noise_sub["std"] / np.sqrt(trials),
                marker="o",
                capsize=3,
                label=f"{model} noise={noise_p:.1f}"
            )

    plt.xlabel("Pattern length n_bits")
    plt.ylabel("Bitwise accuracy")
    plt.title("Bitwise accuracy vs n_bits across noise levels")
    plt.legend()
    plt.savefig(plots_dir / "bit_acc_vs_n_bits.png", dpi=300)

    plt.figure(figsize=(8,6))

    pivot = df.pivot_table(
        index=["n_bits", "noise_p"],
        columns="model",
        values="full_acc",
        aggfunc="mean"
    )

    pivot["gap"] = pivot["QAM"] - pivot["Hopfield"]

    for noise_p in sorted(pivot.index.get_level_values("noise_p").unique()):
        sub = pivot.xs(noise_p, level="noise_p")

        plt.plot(
            sub.index,
            sub["gap"],
            marker="o",
            label=f"noise={noise_p:.1f}"
        )

    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("Pattern length n_bits")
    plt.ylabel("Accuracy Gap (QAM - Hopfield)")
    plt.title("Performance Gap vs n_bits across noise levels")
    plt.legend()
    plt.savefig(plots_dir / "accuracy_gap.png", dpi=300)

    plt.figure(figsize=(8,6))
    for model in ["Hopfield","QAM"]:
        sub = df[df["model"]==model].groupby(["n_bits","noise_p"]).agg(["mean","std"])["train_time_s"]
        plt.errorbar(sub.index.get_level_values(0), sub["mean"], yerr=1.96*sub["std"]/np.sqrt(trials),
                     label=model, marker="o", capsize=3)
    plt.xlabel("Pattern length n_bits")
    plt.ylabel("Training time (s)")
    plt.yscale("log")
    plt.title("Training time vs n_bits (log scale)")
    plt.legend()
    plt.savefig(plots_dir / "train_time_vs_n_bits.png", dpi=300)

    plt.figure(figsize=(8,6))
    for model in ["Hopfield","QAM"]:
        sub = df[df["model"]==model].groupby("noise_p")["full_acc"].mean()
        plt.plot(sub.index, sub.values, marker="o", label=model)
    plt.xlabel("Noise probability")
    plt.ylabel("Full accuracy")
    plt.title("Noise robustness comparison")
    plt.legend()
    plt.savefig(plots_dir / "noise_robustness.png", dpi=300)

    print("Figures saved. Experiment complete.")
    print(f"System: {platform.platform()}, CPU cores: {os.cpu_count()}")
    print("Device: default.qubit (CPU) simulator)")


if __name__ == "__main__":
    main()
