"""Critical noise vs capacity (full-accuracy floor 0.50).

``plot_inference_paper_figures.py`` uses 0.90 instead. The two CSVs are
not the same quantity.
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.paths import results_subdir


ACCURACY_THRESHOLD = 0.50


def main():
    summary_dir = results_subdir("noise_robustness_inference", "summary")
    plots_dir = results_subdir("noise_robustness_inference", "plots")
    csv_path = summary_dir / "noise_inference_summary.csv"

    df = pd.read_csv(csv_path)

    results = []

    grouped = df.groupby(
        ["model", "n_bits", "capacity_ratio"]
    )

    for (model, n_bits, capacity_ratio), group in grouped:

        group = group.sort_values("noise_p")

        valid = group[
            group["full_acc_mean"] >= ACCURACY_THRESHOLD
        ]

        if len(valid) == 0:
            threshold = 0.0
        else:
            threshold = valid["noise_p"].max()

        results.append(
            {
                "model": model,
                "n_bits": n_bits,
                "capacity_ratio": capacity_ratio,
                "critical_noise_threshold": threshold,
            }
        )

    threshold_df = pd.DataFrame(results)

    out_csv = summary_dir / "critical_noise_thresholds.csv"
    threshold_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    plt.figure(figsize=(10, 6))

    for model in threshold_df["model"].unique():

        subset = threshold_df[
            threshold_df["model"] == model
        ]

        avg = (
            subset.groupby("capacity_ratio")
            ["critical_noise_threshold"]
            .mean()
            .reset_index()
        )

        plt.plot(
            avg["capacity_ratio"],
            avg["critical_noise_threshold"],
            marker="o",
            linewidth=2,
            label=model.upper(),
        )

    plt.xlabel("Capacity Ratio (m / n)")
    plt.ylabel("Critical Noise Threshold")
    plt.title(
        "Critical Noise Threshold vs Capacity Ratio"
    )

    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    out_png = plots_dir / "critical_noise_threshold_vs_capacity_ratio.png"
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
