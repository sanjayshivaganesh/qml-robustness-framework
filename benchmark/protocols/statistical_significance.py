"""Statistical-significance protocol.

Each task is one ``(n_bits, capacity_ratio, noise, seed)`` cell. The task
reseeds NumPy and PennyLane NumPy, draws patterns, trains synchronous
Hopfield, scores Bernoulli queries, then builds and trains the PauliX QAM
and scores a second Bernoulli pass. That is the order in
``experiments.statistical_significance.run_trial``.

After the tasks, groups of ``(n_bits, capacity_ratio, noise)`` are
summarised with ``bootstrap_diff`` and a Wilcoxon test. ``bootstrap_diff``
is called with no seed. Cliff's delta is computed with the shared helper
after those draws. This module does not call ``main`` and it does not plot.

The default request is one historical cell. The 720-task module grid starts
only when ``allow_historical_grid`` is true.
"""

from __future__ import annotations

import numpy as np
from pennylane import numpy as pnp

import experiments.statistical_significance as historical
from benchmark.adapters import hopfield_synchronous, qam_paulix
from benchmark.data.corruption import apply_bernoulli
from benchmark.log import configure_logging, log_event, log_failure
from benchmark.output import (
    allocate_run_directory,
    begin_run,
    complete_run,
    create_run_dir,
    fail_run,
    write_csv_table,
    write_json,
)
from benchmark.registry import get, prepare
from benchmark.scoring import WILCOXON_FAILURE, selected

PROTOCOL = "statistical_significance"
HOPFIELD_ID = hopfield_synchronous.identity.procedure_id
QAM_ID = qam_paulix.identity.procedure_id
PROCEDURE_IDS = (HOPFIELD_ID, QAM_ID)
PATTERN_RULE = "max_one_or_int_capacity_times_n_bits"
PATTERN_GENERATOR = "numpy.random"
TRAINING_SCHEDULE = "train_each_model_inside_each_task"
COMPARISON_GROUPS = ("n_bits", "capacity_ratio", "noise")
PAIRED_DIFFERENCE = "qam_minus_hopfield"
CI_PERCENTILES = (2.5, 97.5)
WILCOXON_SENTINEL = WILCOXON_FAILURE["statistical_significance"]
BOOTSTRAP_SEEDED = False
HISTORICAL_PROCESS = "multiprocessing_pool"
FRAMEWORK_PROCESS = "single_process"

MAX_SMOKE_WIDTHS = 1
MAX_SMOKE_CAPACITIES = 1
MAX_SMOKE_NOISE_LEVELS = 1
MAX_SMOKE_SEEDS = 1

TRIAL_COLUMNS = (
    "protocol",
    "grid",
    "comparable_to_archive",
    "n_bits",
    "capacity_ratio",
    "noise",
    "seed",
    "m",
    "hopfield_procedure_id",
    "qam_procedure_id",
    "hopfield_full",
    "hopfield_bit",
    "qam_full",
    "qam_bit",
)

SUMMARY_COLUMNS = (
    "protocol",
    "grid",
    "comparable_to_archive",
    "n_bits",
    "capacity_ratio",
    "noise",
    "sample_size",
    "historical_sample_size",
    "bootstrap_samples",
    "bootstrap_seeded",
    "mean_hopfield_full",
    "mean_qam_full",
    "mean_diff_full",
    "full_ci_2p5",
    "full_ci_97p5",
    "full_p_value",
    "mean_hopfield_bit",
    "mean_qam_bit",
    "mean_diff_bit",
    "bit_ci_2p5",
    "bit_ci_97p5",
    "bit_p_value",
)

EFFECT_COLUMNS = (
    "protocol",
    "grid",
    "comparable_to_archive",
    "n_bits",
    "capacity_ratio",
    "noise",
    "sample_size",
    "delta_full",
    "full_magnitude",
    "delta_bit",
    "bit_magnitude",
    "x_procedure_id",
    "y_procedure_id",
    "computed_by_experiment_main",
)

_RESULT_FILES = (
    "manifest.json",
    "configuration.json",
    "trials.csv",
    "trials.json",
    "summary.csv",
    "summary.json",
    "effect_sizes.csv",
    "effect_sizes.json",
)


class HistoricalGridRefused(RuntimeError):
    """The 720-task statistical-significance grid is not executed here."""


def n_patterns_for(n_bits, capacity_ratio):
    """``max(1, int(capacity_ratio * n_bits))``, the module expression."""
    return max(1, int(capacity_ratio * n_bits))


def historical_sample_size():
    """Seeds per comparison group in the module grid."""
    return len(tuple(historical.SEEDS))


def historical_task_count():
    """Cells in the module grid. This does not run them."""
    return len(tuple(historical_request().tasks()))


class StatisticalRequest:
    """One grid of cells. Building a request does not train."""

    def __init__(
        self,
        n_bits_list,
        capacity_ratios,
        corruption_levels,
        seeds,
        bootstrap_samples,
        grid="smoke",
    ):
        self.n_bits_list = tuple(int(n) for n in n_bits_list)
        self.capacity_ratios = tuple(float(value) for value in capacity_ratios)
        self.corruption_levels = tuple(float(value) for value in corruption_levels)
        self.seeds = tuple(int(seed) for seed in seeds)
        self.bootstrap_samples = int(bootstrap_samples)
        self.grid = grid

    def tasks(self):
        """Module order: width, capacity, noise, then seed.

        Noise is part of the task. A second noise level is a second
        training, because ``run_trial`` reseeds and trains inside the cell.
        """
        for n_bits in self.n_bits_list:
            for capacity in self.capacity_ratios:
                for noise in self.corruption_levels:
                    for seed in self.seeds:
                        yield int(n_bits), float(capacity), float(noise), int(seed)


def historical_request():
    """The module grid. Building it does not train and does not call ``main``."""
    return StatisticalRequest(
        n_bits_list=tuple(int(n) for n in historical.N_BITS_LIST),
        capacity_ratios=tuple(float(value) for value in historical.CAPACITY_RATIOS),
        corruption_levels=tuple(float(value) for value in historical.CORRUPTION_LEVELS),
        seeds=tuple(int(seed) for seed in historical.SEEDS),
        bootstrap_samples=int(historical.BOOTSTRAP_SAMPLES),
        grid="full",
    )


def is_historical_request(request: StatisticalRequest) -> bool:
    expected = historical_request()
    return (
        request.n_bits_list == expected.n_bits_list
        and request.capacity_ratios == expected.capacity_ratios
        and request.corruption_levels == expected.corruption_levels
        and request.seeds == expected.seeds
        and int(request.bootstrap_samples) == expected.bootstrap_samples
        and request.grid == "full"
    )


def lightweight_request():
    """The first module cell: ``n=8``, capacity ``0.2``, noise ``0.1``, seed ``0``.

    Depth, epoch count, learning rate, and the bootstrap count stay the
    module constants. The reduced setting is the number of cells.
    """
    return StatisticalRequest(
        n_bits_list=(int(historical.N_BITS_LIST[0]),),
        capacity_ratios=(float(historical.CAPACITY_RATIOS[0]),),
        corruption_levels=(float(historical.CORRUPTION_LEVELS[0]),),
        seeds=(int(historical.SEEDS[0]),),
        bootstrap_samples=int(historical.BOOTSTRAP_SAMPLES),
        grid="smoke",
    )


def run_statistical_significance(
    request: StatisticalRequest,
    *,
    output_dir=None,
    overwrite: bool = False,
    experiment_id: str | None = None,
    configuration: dict | None = None,
    allow_historical_grid: bool = False,
):
    """Run the cell schedule and return the result mapping.

    ``allow_historical_grid`` starts the module grid. The default stays on
    the one-cell guard. A failure leaves ``execution_status`` ``failed``
    and is re-raised. The run is not resumed.
    """
    configure_logging()
    try:
        if allow_historical_grid:
            if not is_historical_request(request):
                raise HistoricalGridRefused(
                    "allow_historical_grid starts only the statistical-significance module grid."
                )
        else:
            _refuse_unless_smoke(request)
    except Exception as exc:
        log_failure("request_refused", exc, protocol=PROTOCOL)
        raise

    identity = experiment_id or PROTOCOL
    n_tasks = len(tuple(request.tasks()))
    if output_dir is None:
        log_event(
            "run_start",
            experiment_id=identity,
            tasks=n_tasks,
            output="in-memory",
            historical_grid=allow_historical_grid,
            resumable=False,
        )
        try:
            payload = _execute(request)
        except BaseException as exc:
            log_failure("run_failed", exc, experiment_id=identity, output="in-memory")
            raise
        log_event("run_complete", experiment_id=identity, status="in-memory", resumable=False)
        payload["output_dir"] = None
        payload["execution_status"] = None
        return payload

    try:
        if overwrite:
            run_dir = create_run_dir(output_dir, overwrite=True)
            run_id = run_dir.name
        else:
            run_dir, run_id = allocate_run_directory(output_dir, identity)
    except Exception as exc:
        log_failure("output_refused", exc, experiment_id=identity)
        raise
    log_event(
        "run_start",
        run_id=run_id,
        experiment_id=identity,
        tasks=n_tasks,
        output_dir=run_dir,
        historical_grid=allow_historical_grid,
        resumable=False,
    )
    record = configuration if configuration is not None else _request_configuration(request)
    manifest = begin_run(
        run_dir,
        run_id=run_id,
        experiment_id=identity,
        configuration=record,
        models=PROCEDURE_IDS,
        dataset=_dataset_settings(request),
        perturbation=_perturbation_settings(request),
        fields=_manifest_fields(request),
    )
    try:
        payload = _execute(request)
        log_event("persist_start", run_id=run_id, directory=run_dir)
        _write_tables(run_dir, payload)
        log_event("persist_complete", run_id=run_id, directory=run_dir)
        manifest = complete_run(
            run_dir,
            manifest,
            files=_RESULT_FILES,
            fields={
                "smoke_tasks": None if allow_historical_grid else n_tasks,
                "historical_tasks": historical_task_count(),
            },
        )
    except BaseException as exc:
        log_failure("run_failed", exc, run_id=run_id, experiment_id=identity)
        try:
            fail_run(run_dir, manifest, exc)
        except Exception as write_error:
            log_failure("manifest_update_failed", write_error, run_id=run_id)
        raise
    payload["output_dir"] = str(run_dir)
    payload["run_id"] = run_id
    payload["execution_status"] = manifest["execution_status"]
    log_event(
        "run_complete",
        run_id=run_id,
        status=manifest["execution_status"],
        resumable=False,
    )
    return payload


def rng_metadata(request: StatisticalRequest) -> dict:
    """Name the seeds this request applies. This is not a reproducibility certificate."""
    applied = [
        {
            "n_bits": n_bits,
            "capacity_ratio": capacity,
            "noise": noise,
            "value": seed,
            "calls": ["numpy.random.seed", "pennylane.numpy.random.seed"],
        }
        for n_bits, capacity, noise, seed in request.tasks()
    ]
    return {
        "scope": PROTOCOL,
        "exact_reproducibility_claimed": False,
        "confidence_intervals_reproducible": False,
        "p_values_reproducible": False,
        "family": "legacy_global",
        "generator": "numpy.random",
        "also_seeded": "pennylane.numpy.random",
        "default_rng_used": False,
        "configured_seeds": {
            "role": "configured",
            "values": list(request.seeds),
            "formula": "the task seed itself",
            "reused_across_cells": True,
            "source": "request.seeds",
        },
        "seeds_applied": [
            {
                "role": "applied",
                "when": "start of each task, before the pattern draw",
                "values": applied,
            }
        ],
        "internal_resets": [],
        "unseeded_procedures": [
            {
                "role": "unseeded",
                "name": "experiments.statistical_significance.bootstrap_diff",
                "seeded": False,
                "samples": int(request.bootstrap_samples),
                "detail": (
                    "The helper draws numpy.random.randint and has no seed call. "
                    "This runner does not add one. Historical main calls it on the "
                    "parent process after the pool joins, so worker draws do not "
                    "advance that parent stream. This runner is single-process, so "
                    "the call continues from the last task. A recorded seed does "
                    "not reproduce the archived percentile interval."
                ),
            }
        ],
        "state_not_captured": [
            "The legacy RandomState is not serialized.",
            "Historical bootstrap_diff runs on the pool parent, not on a worker.",
            "A single-process run leaves the stream at the end of the last task.",
            "Wilcoxon is a function of the score columns. The smoke group does not have 30 scores.",
            "A recorded task seed does not by itself reproduce the archived summary.",
        ],
    }


def summarize_groups(trial_rows, bootstrap_samples, *, grid):
    """Paired intervals, Wilcoxon, and Cliff's delta for each comparison group.

    ``bootstrap_diff`` is not seeded here. Full-accuracy resampling runs
    before bit-accuracy resampling, which is the order in ``main``.
    Cliff's delta does not draw. Rows inside a group keep the order they
    were appended. Groups are emitted in sorted key order, which is the
    default pandas group order for these three columns.
    """
    grouped = {}
    for row in trial_rows:
        key = (int(row["n_bits"]), float(row["capacity_ratio"]), float(row["noise"]))
        grouped.setdefault(key, []).append(row)

    pair_boot = selected(PROTOCOL, "paired_bootstrap_diff")
    wilcoxon = selected(PROTOCOL, "wilcoxon_pvalue")
    cliffs = selected(PROTOCOL, "cliffs_hopfield_vs_qam")
    summaries = []
    effects = []
    for n_bits, capacity, noise in sorted(grouped):
        members = grouped[(n_bits, capacity, noise)]
        hopfield_full = np.asarray([float(row["hopfield_full"]) for row in members], dtype=float)
        qam_full = np.asarray([float(row["qam_full"]) for row in members], dtype=float)
        hopfield_bit = np.asarray([float(row["hopfield_bit"]) for row in members], dtype=float)
        qam_bit = np.asarray([float(row["qam_bit"]) for row in members], dtype=float)

        full_diffs = pair_boot(qam_full, hopfield_full, n=int(bootstrap_samples))
        full_low, full_high = np.percentile(full_diffs, list(CI_PERCENTILES))
        full_p = wilcoxon(qam_full, hopfield_full, on_failure=WILCOXON_SENTINEL)

        bit_diffs = pair_boot(qam_bit, hopfield_bit, n=int(bootstrap_samples))
        bit_low, bit_high = np.percentile(bit_diffs, list(CI_PERCENTILES))
        bit_p = wilcoxon(qam_bit, hopfield_bit, on_failure=WILCOXON_SENTINEL)

        delta_full, full_magnitude = cliffs(hopfield_full, qam_full)
        delta_bit, bit_magnitude = cliffs(hopfield_bit, qam_bit)
        sample_size = len(members)
        summaries.append(
            {
                "protocol": PROTOCOL,
                "grid": grid,
                "comparable_to_archive": False,
                "n_bits": n_bits,
                "capacity_ratio": capacity,
                "noise": noise,
                "sample_size": sample_size,
                "historical_sample_size": historical_sample_size(),
                "bootstrap_samples": int(bootstrap_samples),
                "bootstrap_seeded": BOOTSTRAP_SEEDED,
                "mean_hopfield_full": float(hopfield_full.mean()),
                "mean_qam_full": float(qam_full.mean()),
                "mean_diff_full": float(qam_full.mean() - hopfield_full.mean()),
                "full_ci_2p5": float(full_low),
                "full_ci_97p5": float(full_high),
                "full_p_value": _json_number(full_p),
                "mean_hopfield_bit": float(hopfield_bit.mean()),
                "mean_qam_bit": float(qam_bit.mean()),
                "mean_diff_bit": float(qam_bit.mean() - hopfield_bit.mean()),
                "bit_ci_2p5": float(bit_low),
                "bit_ci_97p5": float(bit_high),
                "bit_p_value": _json_number(bit_p),
            }
        )
        effects.append(
            {
                "protocol": PROTOCOL,
                "grid": grid,
                "comparable_to_archive": False,
                "n_bits": n_bits,
                "capacity_ratio": capacity,
                "noise": noise,
                "sample_size": sample_size,
                "delta_full": float(delta_full),
                "full_magnitude": full_magnitude,
                "delta_bit": float(delta_bit),
                "bit_magnitude": bit_magnitude,
                "x_procedure_id": HOPFIELD_ID,
                "y_procedure_id": QAM_ID,
                "computed_by_experiment_main": False,
            }
        )
    return summaries, effects


def _manifest_fields(request: StatisticalRequest) -> dict:
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "main_invoked": False,
        "plotting": False,
        "procedure_ids": list(PROCEDURE_IDS),
        "historical_module": "experiments.statistical_significance",
        "historical_qam": "experiments.statistical_significance.build_qam",
        "historical_epochs": int(historical.EPOCHS),
        "historical_depth": int(historical.DEPTH),
        "historical_lr": float(historical.LR),
        "historical_tasks": historical_task_count(),
        "historical_sample_size": historical_sample_size(),
        "historical_grid": request.grid == "full",
        "training_schedule": TRAINING_SCHEDULE,
        "pattern_rule": PATTERN_RULE,
        "comparison_groups": list(COMPARISON_GROUPS),
        "paired_difference": PAIRED_DIFFERENCE,
        "bootstrap_samples": int(request.bootstrap_samples),
        "bootstrap_seeded": BOOTSTRAP_SEEDED,
        "ci_percentiles": list(CI_PERCENTILES),
        "wilcoxon_failure_value": WILCOXON_SENTINEL,
        "alpha_decision": None,
        "process_model": FRAMEWORK_PROCESS,
        "historical_process_model": HISTORICAL_PROCESS,
        "rng": rng_metadata(request),
    }


def _request_configuration(request: StatisticalRequest) -> dict:
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "n_bits_list": list(request.n_bits_list),
        "capacity_ratios": list(request.capacity_ratios),
        "corruption_levels": list(request.corruption_levels),
        "seeds": list(request.seeds),
        "bootstrap_samples": int(request.bootstrap_samples),
    }


def _dataset_settings(request: StatisticalRequest) -> dict:
    return {
        "n_bits_list": list(request.n_bits_list),
        "capacity_ratios": list(request.capacity_ratios),
        "pattern_generator": PATTERN_GENERATOR,
        "pattern_rule": PATTERN_RULE,
    }


def _perturbation_settings(request: StatisticalRequest) -> dict:
    return {
        "procedure_id": "bernoulli",
        "implementation": "src.data_generation.corrupt_bernoulli",
        "levels": list(request.corruption_levels),
        "when": "each_stored_pattern_inside_hopfield_then_inside_qam",
    }


def _refuse_unless_smoke(request: StatisticalRequest) -> None:
    if request.grid != "smoke":
        raise HistoricalGridRefused(
            "only grid='smoke' is implemented unless allow_historical_grid is set"
        )
    if int(request.bootstrap_samples) != int(historical.BOOTSTRAP_SAMPLES):
        raise HistoricalGridRefused(
            "bootstrap_samples stays "
            f"{int(historical.BOOTSTRAP_SAMPLES)}. This runner does not seed a shorter resample."
        )
    if (
        not request.n_bits_list
        or not request.capacity_ratios
        or not request.corruption_levels
        or not request.seeds
    ):
        raise ValueError("n_bits_list, capacity_ratios, corruption_levels, and seeds must be non-empty")
    if len(request.n_bits_list) > MAX_SMOKE_WIDTHS:
        raise HistoricalGridRefused(f"smoke widths must be at most {MAX_SMOKE_WIDTHS}")
    if len(request.capacity_ratios) > MAX_SMOKE_CAPACITIES:
        raise HistoricalGridRefused(
            f"smoke capacities must be at most {MAX_SMOKE_CAPACITIES}"
        )
    if len(request.corruption_levels) > MAX_SMOKE_NOISE_LEVELS:
        raise HistoricalGridRefused(
            f"smoke noise levels must be at most {MAX_SMOKE_NOISE_LEVELS}"
        )
    if len(request.seeds) > MAX_SMOKE_SEEDS:
        raise HistoricalGridRefused(f"smoke seeds must be at most {MAX_SMOKE_SEEDS}")
    historical_axes = historical_request()
    if request.n_bits_list[0] not in historical_axes.n_bits_list:
        raise HistoricalGridRefused("smoke n_bits must be one of the module lengths")
    if request.capacity_ratios[0] not in historical_axes.capacity_ratios:
        raise HistoricalGridRefused("smoke capacity must be one of the module ratios")
    if request.corruption_levels[0] not in historical_axes.corruption_levels:
        raise HistoricalGridRefused("smoke noise must be one of the module levels")
    if request.seeds[0] not in historical_axes.seeds:
        raise HistoricalGridRefused("smoke seed must be one of the module seeds")
    if len(tuple(request.tasks())) == historical_task_count():
        raise HistoricalGridRefused(
            "the historical statistical-significance grid is not started here"
        )


def _execute(request: StatisticalRequest):
    specs = list(request.tasks())
    total = len(specs)
    log_event(
        "execution_start",
        tasks=total,
        bootstrap_samples=int(request.bootstrap_samples),
        bootstrap_seeded=BOOTSTRAP_SEEDED,
    )
    trials = []
    for index, (n_bits, capacity, noise, seed) in enumerate(specs, start=1):
        log_event(
            "trial_start",
            progress=f"{index}/{total}",
            n_bits=n_bits,
            capacity_ratio=capacity,
            noise=noise,
            seed=seed,
        )
        trials.append(_run_task(n_bits, capacity, noise, seed, grid=request.grid))
        log_event(
            "trial_complete",
            progress=f"{index}/{total}",
            n_bits=n_bits,
            capacity_ratio=capacity,
            noise=noise,
            seed=seed,
        )
    log_event(
        "statistics_start",
        groups="n_bits,capacity_ratio,noise",
        bootstrap_samples=int(request.bootstrap_samples),
        bootstrap_seeded=BOOTSTRAP_SEEDED,
    )
    summaries, effects = summarize_groups(
        trials, request.bootstrap_samples, grid=request.grid
    )
    log_event("statistics_complete", groups=len(summaries), resumable=False)
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "comparable_to_archive": False,
        "main_invoked": False,
        "procedure_ids": list(PROCEDURE_IDS),
        "trials": trials,
        "summaries": summaries,
        "effect_sizes": effects,
    }


def _run_task(n_bits, capacity, noise, seed, *, grid):
    np.random.seed(seed)
    pnp.random.seed(seed)
    count = n_patterns_for(n_bits, capacity)
    patterns = np.random.choice([-1, 1], size=(count, n_bits))
    accuracy = selected(PROTOCOL, "pattern_accuracy_pair")

    hopfield_factory, hopfield_args, hopfield_kwargs = prepare(
        HOPFIELD_ID, {"n_bits": n_bits, "steps": 10}
    )
    hopfield = hopfield_factory(*hopfield_args, **hopfield_kwargs)
    log_event("model_construct", procedure_id=HOPFIELD_ID, n_bits=n_bits, seed=seed)
    get(HOPFIELD_ID).adapter.train(hopfield, patterns)
    log_event("model_train", procedure_id=HOPFIELD_ID, seed=seed)
    hopfield_full_scores = []
    hopfield_bit_scores = []
    hopfield_recalled = []
    hopfield_queries = []
    for pattern in patterns:
        query = apply_bernoulli(pattern, noise)
        recalled = np.asarray(get(HOPFIELD_ID).adapter.recall(hopfield, query))
        full, bit = accuracy(recalled, pattern)
        hopfield_full_scores.append(full)
        hopfield_bit_scores.append(bit)
        hopfield_recalled.append(_vector(recalled))
        hopfield_queries.append(_vector(query))
    log_event(
        "model_evaluate",
        procedure_id=HOPFIELD_ID,
        seed=seed,
        noise=float(noise),
        queries=count,
    )

    qam_factory, qam_args, qam_kwargs = prepare(QAM_ID, {"n_bits": n_bits})
    circuit = qam_factory(*qam_args, **qam_kwargs)
    log_event(
        "model_construct",
        procedure_id=QAM_ID,
        n_bits=n_bits,
        seed=seed,
        depth=int(historical.DEPTH),
        epochs=int(historical.EPOCHS),
    )
    params = get(QAM_ID).adapter.train(patterns, circuit)
    log_event("model_train", procedure_id=QAM_ID, seed=seed, epochs=int(historical.EPOCHS))
    qam_full_scores = []
    qam_bit_scores = []
    qam_recalled = []
    qam_queries = []
    for pattern in patterns:
        query = apply_bernoulli(pattern, noise)
        recalled = np.asarray(get(QAM_ID).adapter.recall(params, circuit, query))
        full, bit = accuracy(recalled, pattern)
        qam_full_scores.append(full)
        qam_bit_scores.append(bit)
        qam_recalled.append(_vector(recalled))
        qam_queries.append(_vector(query))
    log_event(
        "model_evaluate",
        procedure_id=QAM_ID,
        seed=seed,
        noise=float(noise),
        queries=count,
    )

    return {
        "protocol": PROTOCOL,
        "grid": grid,
        "comparable_to_archive": False,
        "n_bits": int(n_bits),
        "capacity_ratio": float(capacity),
        "noise": float(noise),
        "seed": int(seed),
        "m": int(count),
        "hopfield_procedure_id": HOPFIELD_ID,
        "qam_procedure_id": QAM_ID,
        "hopfield_full": float(np.mean(hopfield_full_scores)),
        "hopfield_bit": float(np.mean(hopfield_bit_scores)),
        "qam_full": float(np.mean(qam_full_scores)),
        "qam_bit": float(np.mean(qam_bit_scores)),
        "patterns": _matrix(patterns),
        "hopfield_queries": hopfield_queries,
        "hopfield_recalled": hopfield_recalled,
        "qam_queries": qam_queries,
        "qam_recalled": qam_recalled,
        "depth": int(historical.DEPTH),
        "epochs": int(historical.EPOCHS),
        "lr": float(historical.LR),
    }


def _write_tables(run_dir, payload):
    write_csv_table(run_dir, "trials.csv", payload["trials"], TRIAL_COLUMNS)
    write_json(run_dir, "trials.json", payload["trials"])
    write_csv_table(run_dir, "summary.csv", payload["summaries"], SUMMARY_COLUMNS)
    write_json(
        run_dir,
        "summary.json",
        {
            "comparison_groups": list(COMPARISON_GROUPS),
            "paired_difference": PAIRED_DIFFERENCE,
            "bootstrap_seeded": BOOTSTRAP_SEEDED,
            "confidence_intervals_reproducible": False,
            "p_values_reproducible": False,
            "alpha_decision": None,
            "rows": payload["summaries"],
        },
    )
    write_csv_table(run_dir, "effect_sizes.csv", payload["effect_sizes"], EFFECT_COLUMNS)
    write_json(
        run_dir,
        "effect_sizes.json",
        {
            "implementation": "src.metrics.cliffs_delta",
            "x_procedure_id": HOPFIELD_ID,
            "y_procedure_id": QAM_ID,
            "computed_by_experiment_main": False,
            "rows": payload["effect_sizes"],
        },
    )


def _matrix(values):
    return np.asarray(values).astype(int).tolist()


def _vector(values):
    return np.asarray(values, dtype=float).tolist()


def _json_number(value):
    if value is None:
        return None
    return float(value)
