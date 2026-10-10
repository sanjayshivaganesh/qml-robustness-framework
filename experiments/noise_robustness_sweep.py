"""Rot-ansatz QAM vs synchronous Hopfield vs Bernoulli bit-flip noise.

``n_bits`` and corruption rate vary; ``m = max(1, int(0.3 * n_bits))``.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp
import time
import platform
import psutil
from multiprocessing import Pool, cpu_count

from src.hopfield import Hopfield
from src.data_generation import corrupt_bernoulli
from src.metrics import mean_std_ci
from src.paths import results_subdir


# ------------------------------
# QAM (Rot ansatz; this experiment only)
# ------------------------------
class QAM:
    def __init__(self, n_bits, depth=6, train_steps=300, lr=0.05):
        self.n_bits = n_bits
        self.depth = min(depth, n_bits)
        self.train_steps = train_steps
        self.lr = lr
        self.dev = qml.device("default.qubit", wires=n_bits)

        pnp.random.seed(42)

        self.params = pnp.array(
            0.01 * np.random.randn(self.depth, n_bits, 3),
            requires_grad=True
        )

        @qml.qnode(self.dev)
        def circuit(params, x):
            # RX + RY encoding (not the RY-only circuit used in capacity/depth)
            for i in range(self.n_bits):
                qml.RX(x[i] * np.pi, wires=i)
                qml.RY(x[i] * np.pi / 2, wires=i)

            for d in range(self.depth):
                for i in range(self.n_bits):
                    qml.Rot(*params[d, i], wires=i)
                for i in range(self.n_bits - 1):
                    qml.CNOT(wires=[i, i+1])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_bits)]

        self.circuit = circuit

    def train(self, patterns):
        start_time = time.time()

        def loss(params):
            total = 0
            for p in patterns:
                out = pnp.array(self.circuit(params, p))
                total += pnp.sum((p - out) ** 2)
            return total / len(patterns)

        opt = qml.AdamOptimizer(stepsize=self.lr)
        params = self.params

        for _ in range(self.train_steps):
            params = opt.step(loss, params)

        self.params = params
        return time.time() - start_time

    def recall(self, x):
        start_time = time.time()
        out = np.sign(self.circuit(self.params, x))
        out[out == 0] = 1
        return out, (time.time() - start_time)/len(x)


# ------------------------------
# Experiment Config
# ------------------------------
n_bits_list = [4, 6, 8, 10]
trials = 30
corruption_rates = [0.0, 0.1, 0.2, 0.3, 0.4]

capacity_ratio = 0.3

depth = 6
train_steps = 300


def run_trial(args):
    n_bits, corruption, seed = args
    np.random.seed(seed)
    pnp.random.seed(seed)

    m = max(1, int(capacity_ratio * n_bits))
    patterns = np.random.choice([-1,1], size=(m, n_bits))

    hop = Hopfield(n_bits)
    hop_train_time = hop.train(patterns)

    hop_full_correct = 0
    hop_bits_correct = 0
    hop_infer_times = []

    for p in patterns:
        x = corrupt_bernoulli(p, corruption)
        rec, t_inf = hop.recall_timed(x)
        hop_infer_times.append(t_inf)
        if np.array_equal(rec, p):
            hop_full_correct += 1
        hop_bits_correct += np.sum(rec == p)

    qam = QAM(n_bits)
    qam_train_time = qam.train(patterns)

    qam_full_correct = 0
    qam_bits_correct = 0
    qam_infer_times = []

    for p in patterns:
        x = corrupt_bernoulli(p, corruption)
        rec_q, t_inf_q = qam.recall(x)
        qam_infer_times.append(t_inf_q)
        if np.array_equal(rec_q, p):
            qam_full_correct += 1
        qam_bits_correct += np.sum(rec_q == p)

    return {
        "hop_full": hop_full_correct/m,
        "hop_bits": hop_bits_correct/(m*n_bits),
        "qam_full": qam_full_correct/m,
        "qam_bits": qam_bits_correct/(m*n_bits),
        "hop_train": hop_train_time,
        "qam_train": qam_train_time,
        "hop_infer": np.mean(hop_infer_times),
        "qam_infer": np.mean(qam_infer_times)
    }


def main():
    summary_dir = results_subdir("noise_robustness_sweep", "summary")
    plots_dir = results_subdir("noise_robustness_sweep", "plots")
    results = []

    for n_bits in n_bits_list:
        for corruption in corruption_rates:
            print(f"Starting n_bits={n_bits}, corruption={corruption}")

            tasks = [(n_bits, corruption, seed) for seed in range(trials)]

            with Pool(max(1, cpu_count() - 1)) as pool:
                outputs = pool.map(run_trial, tasks)

            hop_full_acc = [o["hop_full"] for o in outputs]
            hop_bit_acc = [o["hop_bits"] for o in outputs]
            qam_full_acc = [o["qam_full"] for o in outputs]
            qam_bit_acc = [o["qam_bits"] for o in outputs]
            hop_train_time = [o["hop_train"] for o in outputs]
            qam_train_time = [o["qam_train"] for o in outputs]
            hop_infer_time = [o["hop_infer"] for o in outputs]
            qam_infer_time = [o["qam_infer"] for o in outputs]

            hop_full_mean, hop_full_std, hop_full_ci = mean_std_ci(hop_full_acc)
            hop_bit_mean, hop_bit_std, hop_bit_ci = mean_std_ci(hop_bit_acc)
            qam_full_mean, qam_full_std, qam_full_ci = mean_std_ci(qam_full_acc)
            qam_bit_mean, qam_bit_std, qam_bit_ci = mean_std_ci(qam_bit_acc)

            results.append({
                "n_bits": n_bits,
                "corruption": corruption,

                "hopfield_full_acc_mean": hop_full_mean,
                "hopfield_full_acc_std": hop_full_std,
                "hopfield_full_acc_ci": hop_full_ci,

                "hopfield_bit_acc_mean": hop_bit_mean,
                "hopfield_bit_acc_std": hop_bit_std,
                "hopfield_bit_acc_ci": hop_bit_ci,

                "qam_full_acc_mean": qam_full_mean,
                "qam_full_acc_std": qam_full_std,
                "qam_full_acc_ci": qam_full_ci,

                "qam_bit_acc_mean": qam_bit_mean,
                "qam_bit_acc_std": qam_bit_std,
                "qam_bit_acc_ci": qam_bit_ci,

                "hopfield_train_mean": np.mean(hop_train_time),
                "qam_train_mean": np.mean(qam_train_time),
                "hopfield_infer_mean": np.mean(hop_infer_time),
                "qam_infer_mean": np.mean(qam_infer_time)
            })

            print(f"Done n_bits={n_bits}, corruption={corruption}")

    df = pd.DataFrame(results)
    df.to_csv(summary_dir / "noise_robustness_sweep.csv", index=False)

    for n_bits in n_bits_list:
        sub = df[df["n_bits"] == n_bits]

        plt.figure()
        plt.errorbar(sub["corruption"], sub["hopfield_full_acc_mean"],
                     yerr=sub["hopfield_full_acc_ci"], marker="o", label="Hopfield")
        plt.errorbar(sub["corruption"], sub["qam_full_acc_mean"],
                     yerr=sub["qam_full_acc_ci"], marker="s", label="QAM")

        plt.xlabel("Bit-flip corruption probability")
        plt.ylabel("Full-pattern accuracy")
        plt.title(f"Full-Pattern Recall Accuracy vs Corruption (n={n_bits})")
        plt.grid()
        plt.legend()
        plt.savefig(plots_dir / f"noise_full_acc_n{n_bits}.png", dpi=300)

    for n_bits in n_bits_list:
        sub = df[df["n_bits"] == n_bits]

        plt.figure()
        plt.errorbar(sub["corruption"], sub["hopfield_bit_acc_mean"],
                     yerr=sub["hopfield_bit_acc_ci"], marker="o", label="Hopfield")
        plt.errorbar(sub["corruption"], sub["qam_bit_acc_mean"],
                     yerr=sub["qam_bit_acc_ci"], marker="s", label="QAM")

        plt.xlabel("Bit-flip corruption probability")
        plt.ylabel("Bit accuracy")
        plt.title(f"Bitwise Recall Accuracy vs Corruption (n={n_bits})")
        plt.grid()
        plt.legend()
        plt.savefig(plots_dir / f"noise_bit_acc_n{n_bits}.png", dpi=300)

    print("Noise robustness experiment complete.")
    print("System Info:", platform.platform())
    print("CPU cores:", psutil.cpu_count(logical=True))


if __name__ == "__main__":
    main()
