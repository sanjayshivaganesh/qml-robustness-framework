"""QAM training loss vs step (same RY-linear circuit as the capacity sweep).

If loss barely moves after step 100, ``train_steps=100`` in the capacity
experiment is a reasonable budget.
"""

import numpy as onp
import pandas as pd
import matplotlib.pyplot as plt

from src.qam import train_ry_linear_qam_record_loss
from src.paths import results_subdir

CONFIGS = [
    (8, 4),
    (12, 8),
    (16, 12),
]

TRAIN_STEPS = 300
LR = 0.05
DEPTH_POLICY = lambda n: min(8, n)
SEEDS = [0, 1, 2]

EXPERIMENT_NAME = "convergence"


def main():
    raw_dir = results_subdir(EXPERIMENT_NAME, "raw")
    summary_dir = results_subdir(EXPERIMENT_NAME, "summary")
    plots_dir = results_subdir(EXPERIMENT_NAME, "plots")

    summary_rows = []

    for n_bits, m in CONFIGS:

        print("\n" + "=" * 70)
        print(f"CONVERGENCE STUDY | n_bits={n_bits} | m={m}")
        print("=" * 70)

        depth = DEPTH_POLICY(n_bits)

        all_losses = []

        for seed in SEEDS:

            print(f"\nRunning seed {seed}")

            onp.random.seed(seed)

            patterns = onp.random.choice(
                [-1, 1],
                size=(m, n_bits)
            )

            _, _, losses = train_ry_linear_qam_record_loss(
                patterns,
                n_bits,
                depth,
                steps=TRAIN_STEPS,
                lr=LR,
            )

            all_losses.append(losses)

        all_losses = onp.array(all_losses)

        mean_loss = all_losses.mean(axis=0)
        std_loss = all_losses.std(axis=0)

        df = pd.DataFrame({
            "step": range(len(mean_loss)),
            "mean_loss": mean_loss,
            "std_loss": std_loss
        })

        df.to_csv(
            raw_dir / f"convergence_n{n_bits}_m{m}.csv",
            index=False
        )

        plt.figure(figsize=(7, 5))
        plt.plot(df["step"], df["mean_loss"])
        plt.fill_between(
            df["step"],
            df["mean_loss"] - df["std_loss"],
            df["mean_loss"] + df["std_loss"],
            alpha=0.2
        )

        plt.axvline(100, linestyle="--")

        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title(f"QAM Convergence (n={n_bits}, m={m})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            plots_dir / f"convergence_n{n_bits}_m{m}.png",
            dpi=300
        )

        plt.close()

        loss_100 = float(mean_loss[100])
        loss_final = float(mean_loss[-1])

        post100_improvement_pct = (
            (loss_100 - loss_final)
            / max(abs(loss_100), 1e-12)
            * 100
        )

        summary_rows.append({
            "n_bits": n_bits,
            "m": m,
            "depth": depth,
            "loss_step_0": float(mean_loss[0]),
            "loss_step_50": float(mean_loss[50]),
            "loss_step_100": loss_100,
            "loss_step_200": float(mean_loss[200]),
            "loss_final": loss_final,
            "post100_improvement_pct": post100_improvement_pct
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        summary_dir / "convergence_summary.csv",
        index=False
    )

    print("\n====================================================")
    print("Interpretation")
    print("====================================================")
    print("If post100_improvement_pct is consistently below")
    print("roughly 5-10%, train_steps=100 is justified.")
    print("If values are much larger, increase training budget.")


if __name__ == "__main__":
    main()
