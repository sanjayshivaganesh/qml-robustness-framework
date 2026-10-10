"""Plot aggregated large-scale noise-robustness inference results."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.paths import results_subdir


def main():
    summary_dir = results_subdir("noise_robustness_inference", "summary")
    plots_dir = results_subdir("noise_robustness_inference", "plots")
    csv_path = summary_dir / "noise_inference_summary.csv"

    df = pd.read_csv(csv_path)

    plt.style.use("default")

    MODELS = {
        "hopfield": "Hopfield",
        "qam": "QAM"
    }

    for n_bits in sorted(df["n_bits"].unique()):

        fig, ax = plt.subplots(figsize=(7,5))

        subset = df[df["n_bits"] == n_bits]

        for model in ["hopfield", "qam"]:
            m = subset[subset["model"] == model]

            ax.plot(
                m["noise_p"],
                m["bit_acc_mean"],
                marker="o",
                label=MODELS[model]
            )

            ax.fill_between(
                m["noise_p"],
                m["bit_ci_low"],
                m["bit_ci_high"],
                alpha=0.2
            )

        ax.set_title(f"Bit Accuracy vs Noise (n={n_bits})")
        ax.set_xlabel("Noise probability")
        ax.set_ylabel("Mean bit accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(plots_dir / f"bit_accuracy_n{n_bits}.png", dpi=300)
        plt.close()

    for n_bits in sorted(df["n_bits"].unique()):

        fig, ax = plt.subplots(figsize=(7,5))

        subset = df[df["n_bits"] == n_bits]

        for model in ["hopfield", "qam"]:
            m = subset[subset["model"] == model]

            ax.plot(
                m["noise_p"],
                m["full_acc_mean"],
                marker="o",
                label=MODELS[model]
            )

            ax.fill_between(
                m["noise_p"],
                m["full_ci_low"],
                m["full_ci_high"],
                alpha=0.2
            )

        ax.set_title(f"Full Retrieval Accuracy vs Noise (n={n_bits})")
        ax.set_xlabel("Noise probability")
        ax.set_ylabel("Probability of perfect recall")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(plots_dir / f"full_accuracy_n{n_bits}.png", dpi=300)
        plt.close()

    runtime = (
        df.groupby(["model", "n_bits"])["runtime_mean"]
          .mean()
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7,5))

    for model in ["hopfield", "qam"]:
        m = runtime[runtime["model"] == model]

        ax.plot(
            m["n_bits"],
            m["runtime_mean"],
            marker="o",
            linewidth=2,
            label=MODELS[model]
        )

    ax.set_title("Runtime Scaling")
    ax.set_xlabel("Problem size (bits)")
    ax.set_ylabel("Mean runtime (s)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(plots_dir / "runtime_scaling.png", dpi=300)
    plt.close()

    target_noise = 0.2

    nearest_noise = (
        df["noise_p"]
        .iloc[(df["noise_p"] - target_noise).abs().argsort()[:1]]
        .values[0]
    )

    cap_df = df[df["noise_p"] == nearest_noise]

    for metric in ["bit_acc_mean", "full_acc_mean"]:

        fig, ax = plt.subplots(figsize=(7,5))

        for model in ["hopfield", "qam"]:
            m = cap_df[cap_df["model"] == model]

            grouped = (
                m.groupby("capacity_ratio")[metric]
                 .mean()
                 .reset_index()
            )

            ax.plot(
                grouped["capacity_ratio"],
                grouped[metric],
                marker="o",
                label=MODELS[model]
            )

        ax.set_title(
            f"{metric.replace('_',' ').title()} vs Capacity Ratio\nNoise={nearest_noise:.2f}"
        )

        ax.set_xlabel("Capacity ratio")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(plots_dir / f"{metric}_capacity_effect.png", dpi=300)
        plt.close()

    for model in ["hopfield", "qam"]:

        heat = (
            df[df["model"] == model]
            .pivot_table(
                index="noise_p",
                columns="n_bits",
                values="full_acc_mean"
            )
        )

        fig, ax = plt.subplots(figsize=(7,6))

        im = ax.imshow(
            heat.values,
            aspect="auto",
            origin="lower"
        )

        ax.set_title(f"{MODELS[model]} Full Recall Accuracy")
        ax.set_xlabel("Bits")
        ax.set_ylabel("Noise")

        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels(heat.columns)

        ax.set_yticks(range(len(heat.index))[::2])
        ax.set_yticklabels(
            np.round(heat.index[::2], 2)
        )

        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(plots_dir / f"{model}_heatmap.png", dpi=300)
        plt.close()

    print(f"Saved plots under {plots_dir}")


if __name__ == "__main__":
    main()
