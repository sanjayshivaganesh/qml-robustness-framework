"""Validate or run one benchmark request file.

``python -m benchmark`` loads a configuration through
``benchmark.config_loader`` and, on ``run``, calls ``benchmark.runner``.
It does not call an experiment ``main`` and it does not write under the
archived results tree.
"""

from __future__ import annotations

import argparse
import sys

from benchmark.log import configure_logging, log_event, log_failure
from benchmark.config_loader import (
    PATTERN_GENERATOR,
    PROTOCOL_PERTURBATION,
    ConfigNotExecutable,
    ConfigValidationError,
    load_benchmark_config,
)
from benchmark.protocols.noise_robustness_sweep import is_historical_noise_sweep_request
from benchmark.protocols.scaling import is_historical_scaling_request
from benchmark.protocols.statistical_significance import is_historical_request
from benchmark.protocols.circuit_depth import (
    is_historical_circuit_depth_request,
)
from benchmark.protocols.noise_robustness_inference import (
    is_historical_inference_request,
)
from benchmark.protocols.storage_capacity import (
    is_historical_request as is_historical_storage_request,
)
from benchmark.output import ArchivedResultsRefused
from benchmark.registry import UnknownModel, get
from benchmark.runner import UnknownProtocol, run_benchmark

EXIT_OK = 0
EXIT_NOT_EXECUTABLE = 1
EXIT_INVALID = 2

EXAMPLES = """
examples:
  python -m benchmark validate configs/noise_robustness_sweep.smoke.json
  python -m benchmark run configs/noise_robustness_sweep.smoke.json --dry-run
  python -m benchmark run configs/noise_robustness_sweep.smoke.json --output-dir runs/noise_robustness_sweep_smoke
  python -m benchmark validate configs/noise_robustness_sweep.json
  python -m benchmark run configs/noise_robustness_sweep.json --dry-run
"""


def build_parser():
    """Return the argument parser. Parsing does not load a configuration."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmark",
        description=(
            "Validate or run a benchmark configuration. "
            "This entry point does not call an experiment main."
        ),
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate",
        help="Load a configuration and report whether the current runner can start it.",
        description=(
            "Load a configuration and report whether the current runner can start it. "
            "Does not train, does not construct a circuit, and does not write a result."
        ),
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    validate.add_argument("config", help="Path to a benchmark request file.")

    run = commands.add_parser(
        "run",
        help="Run an executable configuration.",
        description=(
            "Print the planned experiment, then run it through the benchmark runner. "
            "A configuration path is required. --dry-run stops after the plan."
        ),
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run.add_argument("config", help="Path to a benchmark request file.")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan. Do not train or write a result.",
    )
    run.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Parent directory for this run. Overrides the destination in the file. "
            "A new run directory is created inside it. "
            "A path that resolves to the archived results tree is refused."
        ),
    )
    run.add_argument(
        "--full-grid",
        action="store_true",
        help=(
            "Start the historical grid in the request. "
            "The noise sweep is 4 widths, 5 rates, 30 seeds, and 300 steps. "
            "Scaling is 5 widths, 4 noise levels, 30 trials, and 200 steps. "
            "Statistical significance is 3 widths, 4 capacities, 2 noise levels, "
            "30 seeds, and 100 epochs. "
            "Storage capacity is 33 memory loads, 5 seeds, and 100 steps. "
            "Circuit depth is 3 cases, 7 depths, 5 seeds, and 100 steps. "
            "Inference is 9 widths, 5 capacities, 16 noise levels, and 300 trials "
            "(216000 tasks). "
            "A smoke request rejects this flag. "
            "Does not call an experiment main and does not write under results/."
        ),
    )
    run.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Write into the destination itself when that directory already has files. "
            "This does not permit the archived results tree."
        ),
    )
    return parser


def format_plan(config, output_dir=None, *, overwrite=False):
    """Text for the experiment that a successful load describes."""
    if config.protocol == "scaling":
        return _format_scaling_plan(config, output_dir, overwrite=overwrite)
    if config.protocol == "statistical_significance":
        return _format_statistical_plan(config, output_dir, overwrite=overwrite)
    if config.protocol == "storage_capacity":
        return _format_storage_plan(config, output_dir, overwrite=overwrite)
    if config.protocol == "circuit_depth":
        return _format_circuit_depth_plan(config, output_dir, overwrite=overwrite)
    if config.protocol == "noise_robustness_inference":
        return _format_inference_plan(config, output_dir, overwrite=overwrite)
    return _format_noise_plan(config, output_dir, overwrite=overwrite)


def _format_scaling_plan(config, output_dir, *, overwrite):
    if output_dir is not None:
        destination = str(output_dir)
    else:
        resolved = config.resolved_output_dir()
        destination = "in-memory" if resolved is None else str(resolved)
    lines = [
        f"experiment_id: {config.experiment_id}",
        f"protocol: {config.protocol}",
        f"models: {' '.join(config.procedure_ids)}",
        f"dataset.n_bits_list: {' '.join(str(n) for n in config.n_bits_list)}",
        "dataset.n_patterns: min(max(1, n_bits // 2), 6)",
        f"dataset.pattern_generator: pennylane.numpy.random",
        f"perturbation.levels: {' '.join(str(level) for level in config.noise_levels)}",
        f"trial_ids: {' '.join(str(trial) for trial in config.trial_ids)}",
        f"seed_base: {config.seed_base}",
        "seed_formula: 123 + trial_id + n_bits * 100",
        f"steps: {config.steps}",
        f"lr: {config.lr}",
        "depth_rule: min(6, n_bits)",
        "training_schedule: train_once_per_trial_then_score_each_noise",
        f"grid: {config.grid}",
        f"output_destination: {destination}",
        f"overwrite: {'yes' if overwrite else 'no'}",
        (
            "run_directory: the destination itself"
            if overwrite
            else "run_directory: a new subdirectory of output_destination"
        ),
        f"executable: {'yes' if config.executable else 'no'}",
    ]
    for blocker in config.execution_blockers:
        lines.append(f"blocker: {blocker}")
    return "\n".join(lines)


def _format_noise_plan(config, output_dir=None, *, overwrite=False):
    """Text for the noise-sweep experiment that a successful load describes.

    ``output_dir`` is the directory a run would use. None means the
    destination stored on the configuration, which may itself be in-memory.
    """
    if output_dir is not None:
        destination = str(output_dir)
    else:
        resolved = config.resolved_output_dir()
        destination = "in-memory" if resolved is None else str(resolved)
    lines = [
        f"experiment_id: {config.experiment_id}",
        f"protocol: {config.protocol}",
        f"models: {' '.join(config.procedure_ids)}",
        f"dataset.n_bits_list: {' '.join(str(n) for n in config.n_bits_list)}",
        f"dataset.capacity_ratio: {config.capacity_ratio}",
        f"dataset.pattern_generator: {PATTERN_GENERATOR}",
        f"perturbation: {PROTOCOL_PERTURBATION}",
        f"perturbation.levels: {' '.join(str(level) for level in config.corruption_levels)}",
        f"trial_count: {len(config.trial_seeds)}",
        f"trial_seeds: {' '.join(str(seed) for seed in config.trial_seeds)}",
        f"train_steps: {config.train_steps}",
        f"grid: {config.grid}",
        f"output_destination: {destination}",
        f"overwrite: {'yes' if overwrite else 'no'}",
        (
            "run_directory: the destination itself"
            if overwrite
            else "run_directory: a new subdirectory of output_destination"
        ),
        f"executable: {'yes' if config.executable else 'no'}",
    ]
    for blocker in config.execution_blockers:
        lines.append(f"blocker: {blocker}")
    return "\n".join(lines)


def main(argv=None):
    """Parse ``argv`` and return a process status. ``None`` reads ``sys.argv``."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args.config)
        return _run(
            args.config,
            dry_run=args.dry_run,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            full_grid=args.full_grid,
        )
    except ConfigValidationError as exc:
        log_failure("configuration_invalid", exc)
        print(str(exc), file=sys.stderr)
        return EXIT_INVALID
    except (ConfigNotExecutable, UnknownProtocol, UnknownModel, ArchivedResultsRefused) as exc:
        log_failure("execution_refused", exc, error_type=type(exc).__name__)
        print(str(exc), file=sys.stderr)
        return EXIT_NOT_EXECUTABLE


def _validate(path):
    config = load_benchmark_config(path)
    _resolve_models(config)
    log_event(
        "configuration_validated",
        experiment_id=config.experiment_id,
        protocol=config.protocol,
        executable=config.executable,
        trials=len(_trial_ids(config)),
    )
    print(format_plan(config))
    return EXIT_OK


def _is_historical_config(config):
    """True when the file is that protocol's module grid. A smoke file returns false."""
    try:
        request = config.to_runner_request(allow_historical_grid=True)
    except ConfigNotExecutable:
        return False
    if config.protocol == "scaling":
        return is_historical_scaling_request(request)
    if config.protocol == "statistical_significance":
        return is_historical_request(request)
    if config.protocol == "storage_capacity":
        return is_historical_storage_request(request)
    if config.protocol == "circuit_depth":
        return is_historical_circuit_depth_request(request)
    if config.protocol == "noise_robustness_inference":
        return is_historical_inference_request(request)
    return is_historical_noise_sweep_request(request)


def _run(path, *, dry_run, output_dir, overwrite, full_grid):
    config = load_benchmark_config(path)
    _resolve_models(config)
    print(format_plan(config, output_dir, overwrite=overwrite))
    historical = _is_historical_config(config)
    log_event(
        "configuration_validated",
        experiment_id=config.experiment_id,
        protocol=config.protocol,
        executable=config.executable,
        trials=len(_trial_ids(config)),
        dry_run=dry_run,
        full_grid=full_grid,
    )
    if full_grid and not historical:
        log_event(
            "configuration_not_executable",
            experiment_id=config.experiment_id,
            blockers="--full-grid applies only to the historical request for that protocol",
        )
        print(
            "--full-grid applies only to the historical request for that protocol",
            file=sys.stderr,
        )
        return EXIT_NOT_EXECUTABLE
    if not config.executable and not historical:
        log_event(
            "configuration_not_executable",
            experiment_id=config.experiment_id,
            blockers="; ".join(config.execution_blockers),
        )
        return EXIT_NOT_EXECUTABLE
    if dry_run:
        log_event("dry_run", experiment_id=config.experiment_id)
        print("dry_run: configuration was not executed")
        if historical:
            print("full_grid: not started")
        return EXIT_OK
    if historical and not full_grid:
        log_event(
            "configuration_not_executable",
            experiment_id=config.experiment_id,
            blockers=(
                "the historical grid is not started without --full-grid. "
                + "; ".join(config.execution_blockers)
            ),
        )
        print(
            "the historical grid is not started without --full-grid",
            file=sys.stderr,
        )
        return EXIT_NOT_EXECUTABLE
    if historical:
        result = run_benchmark(
            config,
            output_dir=output_dir,
            overwrite=overwrite,
            allow_historical_grid=True,
        )
    else:
        result = run_benchmark(config, output_dir=output_dir, overwrite=overwrite)
    print("run_complete: yes")
    print(f"output_dir: {result['output_dir']}")
    print(f"procedure_ids: {' '.join(result['procedure_ids'])}")
    print(f"comparable_to_archive: {str(result['comparable_to_archive']).lower()}")
    return EXIT_OK


def _format_statistical_plan(config, output_dir, *, overwrite):
    if output_dir is not None:
        destination = str(output_dir)
    else:
        resolved = config.resolved_output_dir()
        destination = "in-memory" if resolved is None else str(resolved)
    lines = [
        f"experiment_id: {config.experiment_id}",
        f"protocol: {config.protocol}",
        f"models: {' '.join(config.procedure_ids)}",
        f"dataset.n_bits_list: {' '.join(str(n) for n in config.n_bits_list)}",
        f"dataset.capacity_ratios: {' '.join(str(value) for value in config.capacity_ratios)}",
        "dataset.n_patterns: max(1, int(capacity_ratio * n_bits))",
        "dataset.pattern_generator: numpy.random",
        f"perturbation.levels: {' '.join(str(level) for level in config.corruption_levels)}",
        f"seeds: {' '.join(str(seed) for seed in config.seeds)}",
        "seed_formula: the task seed itself",
        "qam: experiments.statistical_significance.build_qam",
        "epochs: 100",
        "depth: 10",
        "lr: 0.02",
        f"bootstrap_samples: {config.bootstrap_samples}",
        "bootstrap_seeded: false",
        "comparison_groups: n_bits capacity_ratio noise",
        "training_schedule: train_each_model_inside_each_task",
        f"grid: {config.grid}",
        f"output_destination: {destination}",
        f"overwrite: {'yes' if overwrite else 'no'}",
        (
            "run_directory: the destination itself"
            if overwrite
            else "run_directory: a new subdirectory of output_destination"
        ),
        f"executable: {'yes' if config.executable else 'no'}",
    ]
    for blocker in config.execution_blockers:
        lines.append(f"blocker: {blocker}")
    return "\n".join(lines)


def _format_storage_plan(config, output_dir, *, overwrite):
    if output_dir is not None:
        destination = str(output_dir)
    else:
        resolved = config.resolved_output_dir()
        destination = "in-memory" if resolved is None else str(resolved)
    width_text = " ".join(
        f"{n_bits}:{','.join(str(m) for m in m_values)}" for n_bits, m_values in config.widths
    )
    cells = sum(len(m_values) for _n_bits, m_values in config.widths)
    lines = [
        f"experiment_id: {config.experiment_id}",
        f"protocol: {config.protocol}",
        f"models: {' '.join(config.procedure_ids)}",
        f"dataset.widths: {width_text}",
        "dataset.m_rule: range(1, n+1), except n>=16 uses 1..10, 12, 14, 16",
        "dataset.pattern_generator: numpy.random",
        f"perturbation.level: {config.recall_noise}",
        "perturbation: fixed_count",
        f"seeds: {' '.join(str(seed) for seed in config.seeds)}",
        "seed_formula: the trial index in range(trials_per_m)",
        "qam: src.qam.train_ry_linear_qam",
        f"train_steps: {config.train_steps}",
        "depth_rule: min(8, n_bits)",
        f"lr: {config.lr}",
        f"bootstrap_samples: {config.bootstrap_samples}",
        "bootstrap_separate_seed: false",
        "critical_threshold: 0.75",
        f"cells: {cells}",
        f"grid: {config.grid}",
        f"output_destination: {destination}",
        f"overwrite: {'yes' if overwrite else 'no'}",
        (
            "run_directory: the destination itself"
            if overwrite
            else "run_directory: a new subdirectory of output_destination"
        ),
        f"executable: {'yes' if config.executable else 'no'}",
    ]
    for blocker in config.execution_blockers:
        lines.append(f"blocker: {blocker}")
    return "\n".join(lines)


def _format_circuit_depth_plan(config, output_dir, *, overwrite):
    if output_dir is not None:
        destination = str(output_dir)
    else:
        resolved = config.resolved_output_dir()
        destination = "in-memory" if resolved is None else str(resolved)
    cases = " ".join(f"{n_bits}:{m}" for n_bits, m in config.cases)
    depths = " ".join(str(depth) for depth in config.depths)
    lines = [
        f"experiment_id: {config.experiment_id}",
        f"protocol: {config.protocol}",
        f"models: {' '.join(config.procedure_ids)}",
        "return_tuple: false",
        "recall: inline sign of circuit expectations; zeros become +1",
        f"dataset.cases: {cases}",
        f"dataset.depths: {depths}",
        "depth_definition: ansatz layer count passed to train_ry_linear_qam",
        "depth_rule_not_applied: min(8, n_bits)",
        f"seeds: {' '.join(str(seed) for seed in config.seeds)}",
        "seed_call: numpy.random.seed, reapplied for every (n, depth)",
        f"train_steps: {config.train_steps}",
        f"lr: {config.lr}",
        f"recall_noise: {config.recall_noise}",
        "perturbation: fixed_count after training",
        "aggregation: mean over the five seeds",
        "bootstrap: no",
        f"cells: {len(config.cases) * len(config.depths)}",
        f"grid: {config.grid}",
        f"output_destination: {destination}",
        (
            "run_directory: the destination itself"
            if overwrite
            else "run_directory: a new subdirectory of output_destination"
        ),
        f"executable: {'yes' if config.executable else 'no'}",
    ]
    for blocker in config.execution_blockers:
        lines.append(f"blocker: {blocker}")
    return "\n".join(lines)


def _format_inference_plan(config, output_dir, *, overwrite):
    if output_dir is not None:
        destination = str(output_dir)
    else:
        resolved = config.resolved_output_dir()
        destination = "in-memory" if resolved is None else str(resolved)
    tasks = (
        len(config.n_bits_list)
        * len(config.capacity_ratios)
        * len(config.noise_levels)
        * len(config.trial_ids)
    )
    lines = [
        f"experiment_id: {config.experiment_id}",
        f"protocol: {config.protocol}",
        "models: hopfield_async_sequential hamming_nearest",
        "csv_model_labels: hopfield qam",
        "hamming_nearest: classical nearest neighbour, variational false",
        "training: no",
        f"dataset.n_bits_list: {' '.join(str(n) for n in config.n_bits_list)}",
        f"dataset.capacity_ratios: {' '.join(str(ratio) for ratio in config.capacity_ratios)}",
        f"dataset.noise_levels: {len(config.noise_levels)}",
        "n_patterns: max(1, int(n_bits * capacity_ratio))",
        "seed_formula: 42 + trial_id + int(n_bits * 1000) + int(capacity_ratio * 100) + int(noise_p * 1000)",
        f"trials_per_config: {len(config.trial_ids)}",
        "perturbation: inline Bernoulli inside run_trial",
        "aggregation: aggregate_results, 200 unseeded resamples",
        f"tasks: {tasks}",
        f"grid: {config.grid}",
        f"output_destination: {destination}",
        (
            "run_directory: the destination itself"
            if overwrite
            else "run_directory: a new subdirectory of output_destination"
        ),
        f"executable: {'yes' if config.executable else 'no'}",
    ]
    for blocker in config.execution_blockers:
        lines.append(f"blocker: {blocker}")
    return "\n".join(lines)


def _trial_ids(config):
    if hasattr(config, "seeds"):
        return config.seeds
    if hasattr(config, "trial_ids"):
        return config.trial_ids
    return config.trial_seeds


def _resolve_models(config):
    """Look up each procedure id. Does not call a factory."""
    for procedure_id in config.procedure_ids:
        get(procedure_id)


if __name__ == "__main__":
    sys.exit(main())
