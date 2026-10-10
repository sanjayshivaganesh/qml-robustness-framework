"""QAM bitwise accuracy vs circuit depth (RY-linear ansatz).

Fixed (n, m) pairs. If accuracy flattens with depth, depth is not the limiter.
"""

from pennylane import numpy as np
import numpy as onp
import pandas as pd
import matplotlib.pyplot as plt

from src.qam import train_ry_linear_qam
from src.data_generation import corrupt_fixed_count
from src.metrics import full_and_bitwise_accuracy
from src.paths import results_subdir

EXPERIMENT_NAME = "circuit_depth"

n_bits_list = [8, 12, 16]

representative_cases = {
    8: 4,
    12: 8,
    16: 12
}

depths = [1, 2, 4, 6, 8, 10, 12]
trials = 5
train_steps = 100
lr = 0.05
recall_noise = 0.15


def main():
    summary_dir = results_subdir(EXPERIMENT_NAME, "summary")
    plots_dir = results_subdir(EXPERIMENT_NAME, "plots")
    rows = []

    for n_bits in n_bits_list:

        m = representative_cases[n_bits]

        print(f"\n=== n={n_bits}, m={m} ===")

        for depth in depths:

            print(f"Depth={depth}")

            full_scores = []
            bit_scores = []
            train_times = []

            for seed in range(trials):

                onp.random.seed(seed)

                patterns = onp.random.choice(
                    [-1, 1],
                    size=(m, n_bits)
                )

                params, circuit, train_time = train_ry_linear_qam(
                    patterns,
                    n_bits,
                    depth,
                    steps=train_steps,
                    lr=lr,
                    return_tuple=False,
                )

                recalled = []

                for p in patterns:

                    noisy = corrupt_fixed_count(
                        p,
                        recall_noise,
                        rng=onp.random,
                    )

                    out = np.array(
                        circuit(params, noisy)
                    )

                    out = np.sign(out)
                    out[out == 0] = 1

                    recalled.append(out)

                full_acc, bit_acc = full_and_bitwise_accuracy(
                    recalled,
                    patterns
                )

                full_scores.append(full_acc)
                bit_scores.append(bit_acc)
                train_times.append(train_time)

            rows.append({
                "n_bits": n_bits,
                "m": m,
                "depth": depth,
                "full_acc_mean": float(np.mean(full_scores)),
                "bit_acc_mean": float(np.mean(bit_scores)),
                "train_time_mean": float(np.mean(train_times)),
                "params": depth * n_bits
            })

    df = pd.DataFrame(rows)
    csv_path = summary_dir / "depth_scaling.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    plt.figure(figsize=(8, 5))

    for n_bits in n_bits_list:

        sub = df[df["n_bits"] == n_bits]

        plt.plot(
            sub["depth"],
            sub["bit_acc_mean"],
            marker="o",
            label=f"n={n_bits}"
        )

    plt.xlabel("Circuit Depth")
    plt.ylabel("Bitwise Recall Accuracy")
    plt.title("QAM Recall Accuracy vs Circuit Depth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(plots_dir / "accuracy_vs_depth.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))

    for n_bits in n_bits_list:

        sub = df[df["n_bits"] == n_bits]

        plt.plot(
            sub["params"],
            sub["bit_acc_mean"],
            marker="o",
            label=f"n={n_bits}"
        )

    plt.xlabel("Parameter Count")
    plt.ylabel("Bitwise Recall Accuracy")
    plt.title("Accuracy vs Model Size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(plots_dir / "accuracy_vs_parameters.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))

    for n_bits in n_bits_list:

        sub = df[df["n_bits"] == n_bits]

        plt.plot(
            sub["depth"],
            sub["train_time_mean"],
            marker="o",
            label=f"n={n_bits}"
        )

    plt.xlabel("Circuit Depth")
    plt.ylabel("Training Time (s)")
    plt.title("Training Cost vs Circuit Depth")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(plots_dir / "training_time_vs_depth.png", dpi=300)
    plt.close()

    print("Depth scaling experiment complete.")


if __name__ == "__main__":
    main()
