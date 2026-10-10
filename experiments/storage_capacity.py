"""Accuracy vs number of stored patterns (RY-linear QAM vs Hopfield).

Recall uses a 15% fixed-count bit flip.
"""

from pennylane import numpy as np
import numpy as onp
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy.stats import wilcoxon

from src.hopfield import Hopfield
from src.qam import train_ry_linear_qam, qam_recall
from src.data_generation import corrupt_fixed_count
from src.metrics import full_and_bitwise_accuracy, bootstrap_mean_ci
from src.paths import results_subdir

# ----------------------------
# Config
# ----------------------------
n_bits_list = [8, 12, 16]
depth_policy = lambda n: min(8, n)
trials_per_m = 5
hopfield_max_factor = 1.0
qam_max_factor = 1.0
train_steps = 100
lr = 0.05
ansatz_type = "RY_linear"
recall_noise = 0.15
bootstrap_samples = 2000
reduced_scale_for_large_n = True

EXPERIMENT_NAME = "storage_capacity"


def confidence_interval(data, confidence=0.95):
    from scipy.stats import sem, t
    a = np.array(data)
    n = len(a)
    m = np.mean(a)
    h = sem(a) * t.ppf((1+confidence)/2., n-1)
    return m, h


def bootstrap_ci(data, n_boot=bootstrap_samples, confidence=0.95):
    return bootstrap_mean_ci(data, n_boot=n_boot, confidence=confidence, rng=onp.random)


def save_clean_plot():
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def main():
    summary_dir = results_subdir(EXPERIMENT_NAME, "summary")
    plots_dir = results_subdir(EXPERIMENT_NAME, "plots")

    all_results = []

    for n_bits in n_bits_list:
        print(f"\n=== n_bits = {n_bits} ===")
        max_m_hopfield = int(n_bits * hopfield_max_factor)
        max_m_qam = int(n_bits * qam_max_factor)

        max_capacity = max(max_m_hopfield, max_m_qam)

        # n=16: skip some m values past 10 to cut runtime; keep the drop-off region.
        if reduced_scale_for_large_n and n_bits >= 16:
            m_values = list(range(1, 11)) + [12, 14, 16]
        else:
            m_values = range(1, max_capacity + 1)

        for m in tqdm(m_values, desc=f"Patterns m sweep for n_bits={n_bits}"):
            print(f"Running storage capacity trial | n_bits={n_bits} | m={m}")
            hop_full, hop_bit = [], []
            qam_full, qam_bit = [], []
            hop_train_times, hop_infer_times = [], []
            qam_train_times, qam_infer_times, qam_param_counts = [], [], []

            for seed in range(trials_per_m):
                onp.random.seed(seed)
                patterns = onp.random.choice([-1,1], size=(m, n_bits))

                # Hopfield
                hop = Hopfield(n_bits)
                t0 = time.time()
                hop.train(patterns)
                hop_train_times.append(time.time() - t0)

                t0 = time.time()
                recalled = np.array([
                    hop.recall(corrupt_fixed_count(p, recall_noise, rng=onp.random))
                    for p in patterns
                ])
                hop_infer_times.append((time.time() - t0)/m)
                full, bit = full_and_bitwise_accuracy(recalled, patterns)
                hop_full.append(full)
                hop_bit.append(bit)

                # QAM only if m <= max_m_qam
                if m <= max_m_qam:
                    depth = depth_policy(n_bits)
                    params, circuit, train_time = train_ry_linear_qam(
                        patterns, n_bits, depth, steps=train_steps, lr=lr, return_tuple=True
                    )
                    qam_train_times.append(train_time)
                    t0_infer = time.time()
                    recalled_qam = np.array([
                        qam_recall(corrupt_fixed_count(p, recall_noise, rng=onp.random), circuit, params)
                        for p in patterns
                    ])
                    qam_infer_times.append((time.time() - t0_infer)/m)
                    qam_full_acc, qam_bit_acc = full_and_bitwise_accuracy(recalled_qam, patterns)
                    qam_full.append(qam_full_acc)
                    qam_bit.append(qam_bit_acc)
                    qam_param_counts.append(params.size)

            # Aggregate results and compute 95% CI
            p_value_full = None
            p_value_bit = None
            hop_full_mean, hop_full_ci_low, hop_full_ci_high = bootstrap_ci(hop_full)
            hop_bit_mean, hop_bit_ci_low, hop_bit_ci_high = bootstrap_ci(hop_bit)
            qam_full_mean, qam_full_ci_low, qam_full_ci_high = (None, None, None)
            qam_bit_mean, qam_bit_ci_low, qam_bit_ci_high = (None, None, None)
            qam_train_mean, _, _ = (None, None, None)
            qam_infer_mean, _, _ = (None, None, None)
            qam_params_mean = None
            if m <= max_m_qam and len(qam_full)>0:
                qam_full_mean, qam_full_ci_low, qam_full_ci_high = bootstrap_ci(qam_full)
                qam_bit_mean, qam_bit_ci_low, qam_bit_ci_high = bootstrap_ci(qam_bit)
                qam_train_mean, _, _ = bootstrap_ci(qam_train_times)
                qam_infer_mean, _, _ = bootstrap_ci(qam_infer_times)
                qam_params_mean = np.mean(qam_param_counts)
            if m <= max_m_qam and len(qam_full) > 0:
                try:
                    p_value_full = wilcoxon(hop_full, qam_full).pvalue
                    p_value_bit = wilcoxon(hop_bit, qam_bit).pvalue
                except Exception:
                    p_value_full = None
                    p_value_bit = None
            all_results.append({
                "model":"Hopfield",
                "n_bits":n_bits,
                "m":m,
                "capacity_ratio":m / n_bits,
                "full_acc_mean":hop_full_mean,
                "full_acc_ci_low":hop_full_ci_low,
                "full_acc_ci_high":hop_full_ci_high,
                "bit_acc_mean":hop_bit_mean,
                "bit_acc_ci_low":hop_bit_ci_low,
                "bit_acc_ci_high":hop_bit_ci_high,
                "train_time_s_mean":np.mean(hop_train_times),
                "infer_time_s_mean":np.mean(hop_infer_times),
                "params_count":n_bits**2,
                "recall_noise":recall_noise,
                "p_value_full_vs_qam":p_value_full,
                "p_value_bit_vs_qam":p_value_bit,
                "extra_notes":"Associative recall from corrupted inputs"
            })
            if m <= max_m_qam:
                all_results.append({
                    "model":"QAM",
                    "n_bits":n_bits,
                    "m":m,
                    "capacity_ratio":m / n_bits,
                    "full_acc_mean":qam_full_mean,
                    "full_acc_ci_low":qam_full_ci_low,
                    "full_acc_ci_high":qam_full_ci_high,
                    "bit_acc_mean":qam_bit_mean,
                    "bit_acc_ci_low":qam_bit_ci_low,
                    "bit_acc_ci_high":qam_bit_ci_high,
                    "train_time_s_mean":qam_train_mean,
                    "infer_time_s_mean":qam_infer_mean,
                    "params_count":qam_params_mean,
                    "recall_noise":recall_noise,
                    "p_value_full_vs_qam":p_value_full,
                    "p_value_bit_vs_qam":p_value_bit,
                    "extra_notes":f"Variational QAM | depth={depth} | steps={train_steps} | lr={lr}"
                })

    df = pd.DataFrame(all_results)
    csv_file = summary_dir / "storage_capacity_sweep.csv"
    df.to_csv(csv_file, index=False)
    print(f"CSV saved at {csv_file}")

    plt.figure(figsize=(7,5))
    for model in ["Hopfield", "QAM"]:
        for n_bits in sorted(df["n_bits"].unique()):
            sub = df[(df["model"] == model) & (df["n_bits"] == n_bits)]
            sub = sub.sort_values("m")

            plt.errorbar(
                sub["m"],
                sub["full_acc_mean"],
                yerr=[
                    sub["full_acc_mean"] - sub["full_acc_ci_low"],
                    sub["full_acc_ci_high"] - sub["full_acc_mean"]
                ],
                marker="o",
                label=f"{model} (n={n_bits})"
            )
    plt.xlabel("Number of Stored Patterns m")
    plt.ylabel("Full-pattern Recall Accuracy")
    plt.title("Storage Capacity Sweep")
    plt.legend()
    save_clean_plot()
    plt.savefig(plots_dir / "storage_capacity_full_acc.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7,5))
    for model in ["Hopfield", "QAM"]:
        for n_bits in sorted(df["n_bits"].unique()):
            sub = df[(df["model"] == model) & (df["n_bits"] == n_bits)]
            sub = sub.sort_values("m")

            plt.errorbar(
                sub["m"],
                sub["bit_acc_mean"],
                yerr=[
                    sub["bit_acc_mean"] - sub["bit_acc_ci_low"],
                    sub["bit_acc_ci_high"] - sub["bit_acc_mean"]
                ],
                marker="o",
                label=f"{model} (n={n_bits})"
            )
    plt.xlabel("Number of Stored Patterns m")
    plt.ylabel("Bitwise Recall Accuracy")
    plt.title("Storage Capacity Sweep (Bitwise)")
    plt.legend()
    save_clean_plot()
    plt.savefig(plots_dir / "storage_capacity_bit_acc.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7,5))

    for model in ["Hopfield", "QAM"]:
        sub = df[df["model"] == model].copy()
        sub = sub.dropna(subset=["full_acc_mean"])
        sub["capacity_ratio"] = sub["m"] / sub["n_bits"]
        sub = sub.sort_values("capacity_ratio")

        grouped = sub.groupby("capacity_ratio").agg({
            "full_acc_mean": "mean",
            "full_acc_ci_low": "mean",
            "full_acc_ci_high": "mean"
        }).reset_index()

        plt.errorbar(
            grouped["capacity_ratio"],
            grouped["full_acc_mean"],
            yerr=[
                grouped["full_acc_mean"] - grouped["full_acc_ci_low"],
                grouped["full_acc_ci_high"] - grouped["full_acc_mean"]
            ],
            marker="o",
            label=model
        )

    plt.xlabel("Capacity Ratio (m / n_bits)")
    plt.ylabel("Full Recall Accuracy")
    plt.title("Storage Capacity: Hopfield vs QAM")
    plt.legend()
    save_clean_plot()
    plt.savefig(plots_dir / "capacity_ratio_plot.png", dpi=300)
    plt.close()

    phase_rows = []
    for model in ["Hopfield", "QAM"]:
        for n_bits in sorted(df["n_bits"].dropna().unique()):
            sub = df[(df["model"] == model) & (df["n_bits"] == n_bits)].copy()
            sub = sub.dropna(subset=["full_acc_mean"])
            if len(sub) == 0:
                continue
            sub["capacity_ratio"] = sub["m"] / sub["n_bits"]
            sub = sub.sort_values("capacity_ratio")

            crit_ratio = None
            for _, row in sub.iterrows():
                if row["full_acc_mean"] < 0.75:
                    crit_ratio = row["capacity_ratio"]
                    break

            phase_rows.append({
                "model": model,
                "n_bits": int(n_bits),
                "critical_capacity_ratio": crit_ratio
            })

    phase_df = pd.DataFrame(phase_rows)
    phase_csv = summary_dir / "phase_transition.csv"
    phase_df.to_csv(phase_csv, index=False)
    print(f"Phase transition data saved at {phase_csv}")

    plt.figure(figsize=(7,5))

    for model in ["Hopfield", "QAM"]:
        sub = phase_df[phase_df["model"] == model]
        plt.plot(
            sub["n_bits"],
            sub["critical_capacity_ratio"],
            marker="o",
            label=model
        )

    plt.xlabel("Pattern Length (n_bits)")
    plt.ylabel("Critical Capacity Ratio")
    plt.title("Critical Capacity Threshold")
    plt.legend()
    save_clean_plot()
    plt.savefig(plots_dir / "critical_capacity_threshold.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7,5))
    for model in ["Hopfield", "QAM"]:
        for n_bits in sorted(df["n_bits"].unique()):
            sub = df[(df["model"] == model) & (df["n_bits"] == n_bits)]
            sub = sub.dropna(subset=["train_time_s_mean"])

            if len(sub) == 0:
                continue

            plt.plot(
                sub["m"],
                sub["train_time_s_mean"],
                marker="o",
                label=f"{model} (n={n_bits})"
            )

    plt.xlabel("Number of Stored Patterns m")
    plt.ylabel("Training Time (s)")
    plt.title("Training Time vs Patterns")
    plt.legend()
    plt.yscale("log")
    save_clean_plot()
    plt.savefig(plots_dir / "train_time_vs_m.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7,5))
    for model in ["Hopfield", "QAM"]:
        for n_bits in sorted(df["n_bits"].unique()):
            sub = df[(df["model"] == model) & (df["n_bits"] == n_bits)]
            sub = sub.dropna(subset=["infer_time_s_mean"])

            if len(sub) == 0:
                continue

            plt.plot(
                sub["m"],
                sub["infer_time_s_mean"],
                marker="o",
                label=f"{model} (n={n_bits})"
            )

    plt.xlabel("Number of Stored Patterns m")
    plt.ylabel("Inference Time per Pattern (s)")
    plt.title("Inference Time vs Patterns")
    plt.legend()
    plt.yscale("log")
    save_clean_plot()
    plt.savefig(plots_dir / "infer_time_vs_m.png", dpi=300)
    plt.close()

    significance_df = df[[
        "n_bits",
        "m",
        "capacity_ratio",
        "p_value_full_vs_qam",
        "p_value_bit_vs_qam"
    ]].drop_duplicates()

    significance_df.to_csv(
        summary_dir / "statistical_significance_summary.csv",
        index=False
    )

    summary_txt = summary_dir / "experiment_summary.txt"
    with open(summary_txt, "w") as f:
        f.write("Storage Capacity Sweep Summary\n")
        f.write("================================\n")
        f.write(f"n_bits_list: {n_bits_list}\n")
        f.write(f"trials_per_m: {trials_per_m}\n")
        f.write(f"train_steps: {train_steps}\n")
        f.write(f"recall_noise: {recall_noise}\n")
        f.write(f"bootstrap_samples: {bootstrap_samples}\n")
        f.write("\nNotes:\n")
        f.write("- Reduced runtime scaling used for n_bits >= 16 while preserving phase-transition behavior.\n")
        f.write("- Statistical significance computed with Wilcoxon signed-rank tests.\n")
        f.write("- Confidence intervals estimated via bootstrap resampling.\n")

    print(f"Experiment summary saved at {summary_txt}")
    print("Storage capacity sweep complete.")


if __name__ == "__main__":
    main()
