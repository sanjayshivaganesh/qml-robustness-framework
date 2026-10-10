"""Run experiments from the repo root. Does nothing unless ``--run`` / ``--only``.

Re-running overwrites CSVs in that experiment's results directory.
"""

import argparse
import subprocess
import sys

from src.paths import PROJECT_ROOT

EXPERIMENTS = [
    ("noise_robustness_sweep", "experiments.noise_robustness_sweep"),
    ("noise_robustness_inference", "experiments.noise_robustness_inference"),
    ("storage_capacity", "experiments.storage_capacity"),
    ("scaling", "experiments.scaling"),
    ("circuit_depth", "experiments.circuit_depth"),
    ("convergence", "experiments.convergence"),
    ("statistical_significance", "experiments.statistical_significance"),
]


def main():
    parser = argparse.ArgumentParser(
        description="List or run associative-memory experiments."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run every experiment sequentially (very expensive).",
    )
    parser.add_argument(
        "--only",
        choices=[name for name, _ in EXPERIMENTS],
        help="Run a single named experiment.",
    )
    args = parser.parse_args()

    print("Experiments (run from the project root after `pip install -e .`):\n")
    for name, module in EXPERIMENTS:
        print(f"  {name}")
        print(f"    python -m {module}")
    print()

    if not args.run and args.only is None:
        print("Pass --run to execute all experiments, or --only NAME for one.")
        print("Warning: noise_robustness_inference and statistical_significance")
        print("are especially long-running.")
        return

    selected = EXPERIMENTS
    if args.only is not None:
        selected = [(n, m) for n, m in EXPERIMENTS if n == args.only]

    for name, module in selected:
        print(f"\n=== Running {name} ===")
        subprocess.run(
            [sys.executable, "-m", module],
            cwd=str(PROJECT_ROOT),
            check=True,
        )


if __name__ == "__main__":
    main()
