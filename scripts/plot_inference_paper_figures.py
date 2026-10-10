"""Paper figures from the inference summary CSV.

Writes ``results/noise_robustness_inference/paper_figures/``.
Critical noise here is the first level with mean full accuracy below 0.90.
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.paths import results_subdir


CRITICAL_ACCURACY = 0.90
N_BITS = 32
CAPACITIES = [0.25, 0.5, 0.75]
NOISE_LEVEL = 0.20


def main():
    summary_dir = results_subdir("noise_robustness_inference", "summary")
    output_dir = results_subdir("noise_robustness_inference", "paper_figures")
    csv_path = summary_dir / "noise_inference_summary.csv"

    df = pd.read_csv(csv_path)

    for cap in CAPACITIES:

        plt.figure(figsize=(8, 5))

        subset = df[
            (df["n_bits"] == N_BITS)
            & (df["capacity_ratio"] == cap)
        ]

        for model in ["hopfield", "qam"]:

            m = subset[
                subset["model"] == model
            ].sort_values("noise_p")

            plt.plot(
                m["noise_p"],
                m["recovery_gain_mean"],
                marker="o",
                linewidth=2,
                label=model.upper()
            )

        plt.xlabel("Noise Probability")
        plt.ylabel("Recovery Gain")
        plt.title(
            f"Recovery Gain vs Noise\nn={N_BITS}, Capacity Ratio={cap}"
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            output_dir / f"recovery_gain_n{N_BITS}_cap{cap}.png",
            dpi=300
        )

        plt.close()

    plt.figure(figsize=(8, 5))

    for cap in CAPACITIES:

        subset = df[
            (df["model"] == "hopfield")
            & (df["n_bits"] == N_BITS)
            & (df["capacity_ratio"] == cap)
        ].sort_values("noise_p")

        plt.plot(
            subset["noise_p"],
            subset["energy_drop_mean"],
            marker="o",
            linewidth=2,
            label=f"Capacity={cap}"
        )

    plt.xlabel("Noise Probability")
    plt.ylabel("Mean Energy Drop")
    plt.title(
        f"Hopfield Energy Drop vs Noise\nn={N_BITS}"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_dir / f"hopfield_energy_drop_n{N_BITS}.png", dpi=300)
    plt.close()

    critical_rows = []

    for model in df["model"].unique():

        for n_bits in sorted(df["n_bits"].unique()):

            for cap in sorted(df["capacity_ratio"].unique()):

                subset = df[
                    (df["model"] == model)
                    & (df["n_bits"] == n_bits)
                    & (df["capacity_ratio"] == cap)
                ].sort_values("noise_p")

                threshold_noise = None

                for _, row in subset.iterrows():

                    if row["full_acc_mean"] < CRITICAL_ACCURACY:
                        threshold_noise = row["noise_p"]
                        break

                if threshold_noise is not None:

                    critical_rows.append(
                        {
                            "model": model,
                            "n_bits": n_bits,
                            "capacity_ratio": cap,
                            "critical_noise": threshold_noise,
                        }
                    )

    critical_df = pd.DataFrame(critical_rows)

    critical_df.to_csv(
        output_dir / "critical_noise_thresholds.csv",
        index=False
    )

    for n_bits in sorted(critical_df["n_bits"].unique()):

        plt.figure(figsize=(8, 5))

        subset_n = critical_df[
            critical_df["n_bits"] == n_bits
        ]

        for model in ["hopfield", "qam"]:

            m = subset_n[
                subset_n["model"] == model
            ].sort_values("capacity_ratio")

            plt.plot(
                m["capacity_ratio"],
                m["critical_noise"],
                marker="o",
                linewidth=2,
                label=model.upper()
            )

        plt.xlabel("Capacity Ratio")
        plt.ylabel("Critical Noise Threshold")
        plt.title(
            f"Critical Noise Threshold vs Capacity Ratio\nn={n_bits}"
        )

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            output_dir / f"critical_threshold_n{n_bits}.png",
            dpi=300
        )

        plt.close()

    plt.figure(figsize=(8, 5))

    for n in sorted(df["n_bits"].unique()):

        subset = df[
            (df["model"] == "hopfield")
            & (df["n_bits"] == n)
            & (abs(df["noise_p"] - NOISE_LEVEL) < 1e-6)
        ].sort_values("capacity_ratio")

        plt.plot(
            subset["capacity_ratio"],
            subset["full_acc_mean"],
            marker="o",
            linewidth=2,
            label=f"Hopfield n={n}"
        )

    for n in sorted(df["n_bits"].unique()):

        subset = df[
            (df["model"] == "qam")
            & (df["n_bits"] == n)
            & (abs(df["noise_p"] - NOISE_LEVEL) < 1e-6)
        ].sort_values("capacity_ratio")

        plt.plot(
            subset["capacity_ratio"],
            subset["full_acc_mean"],
            linestyle="--",
            marker="s",
            linewidth=2,
            label=f"QAM n={n}"
        )

    plt.xlabel("Capacity Ratio")
    plt.ylabel("Full Recall Accuracy")
    plt.title(
        f"Capacity Collapse at Noise={NOISE_LEVEL}"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / f"capacity_collapse_noise_{NOISE_LEVEL}.png",
        dpi=300
    )

    plt.close()

    print("\nAll figures saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()
