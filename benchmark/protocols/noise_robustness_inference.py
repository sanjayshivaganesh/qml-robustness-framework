"""Inference protocol.

Each task is one ``(n_bits, capacity_ratio, noise_p, trial_id)``. The task
body is ``experiments.noise_robustness_inference.run_trial``. That function
seeds NumPy, draws patterns, picks one stored target, applies its inline
Bernoulli mask, recalls with sequential Hopfield, and recalls with Hamming
nearest neighbour.

Rows with ``model="qam"`` are that nearest-neighbour recall. Their
procedure id is ``hamming_nearest``. They are not a variational circuit.
Rows with ``model="hopfield"`` are ``hopfield_async_sequential``.

The summary is ``aggregate_results``. Its 200-resample bootstrap is
unseeded. Its Gaussian columns divide by ``sqrt(TRIALS_PER_CONFIG)``,
which is 300. This module does not seed that bootstrap, does not start
the historical process pool, does not call ``main``, and does not plot.

The default request is one historical cell. The 216,000-task grid starts
only when ``allow_historical_grid`` is true.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import experiments.noise_robustness_inference as historical
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
from benchmark.registry import get
from src.hopfield import hopfield_async_sequential_recall
from src.qam import nearest_neighbor_recall

PROTOCOL = "noise_robustness_inference"
HOPFIELD_ID = "hopfield_async_sequential"
HAMMING_ID = "hamming_nearest"
PROCEDURE_IDS = (HOPFIELD_ID, HAMMING_ID)
CSV_MODEL_LABEL = {
    HOPFIELD_ID: "hopfield",
    HAMMING_ID: "qam",
}
LABEL_TO_PROCEDURE = {label: procedure for procedure, label in CSV_MODEL_LABEL.items()}
PATTERN_GENERATOR = "numpy.random"
TRAINING_SCHEDULE = "no_training"
SEED_FORMULA = (
    "42 + trial_id + int(n_bits * 1000) + int(capacity_ratio * 100) "
    "+ int(noise_p * 1000)"
)
FRAMEWORK_PROCESS = "single_process"
HISTORICAL_PROCESS = "spawn_pool"
PROGRESS_STRIDE = 50

MAX_SMOKE_WIDTHS = 1
MAX_SMOKE_CAPACITIES = 1
MAX_SMOKE_NOISE_LEVELS = 1

RAW_COLUMNS = (
    "protocol",
    "grid",
    "comparable_to_archive",
    "procedure_id",
    "model",
    "n_bits",
    "capacity_ratio",
    "noise_p",
    "trial",
    "bit_accuracy",
    "full_accuracy",
    "hamming_error",
    "input_bit_accuracy",
    "input_hamming_error",
    "recovery_gain",
    "runtime",
    "energy",
    "energy_drop",
)

SUMMARY_COLUMNS = (
    "protocol",
    "grid",
    "comparable_to_archive",
    "procedure_id",
    "model",
    "n_bits",
    "capacity_ratio",
    "noise_p",
    "sample_size",
    "bit_acc_mean",
    "bit_acc_std",
    "full_acc_mean",
    "full_acc_std",
    "hamming_error_mean",
    "hamming_error_std",
    "input_bit_acc_mean",
    "input_hamming_error_mean",
    "recovery_gain_mean",
    "recovery_gain_std",
    "runtime_mean",
    "runtime_std",
    "energy_mean",
    "energy_std",
    "energy_drop_mean",
    "energy_drop_std",
    "bit_ci_low",
    "bit_ci_high",
    "full_ci_low",
    "full_ci_high",
    "bit_acc_ci95",
    "full_acc_ci95",
    "runtime_ci95",
    "recovery_gain_ci95",
    "hamming_error_ci95",
)

_RESULT_FILES = (
    "manifest.json",
    "configuration.json",
    "rows.csv",
    "rows.json",
    "summary.csv",
    "summary.json",
)


class HistoricalGridRefused(RuntimeError):
    """The 216,000-task inference grid is not executed here."""


class InferenceRequest:
    """Widths, capacities, noise levels, and trial ids. Building it does not recall."""

    def __init__(
        self,
        n_bits_list,
        capacity_ratios,
        noise_levels,
        trial_ids,
        grid="smoke",
    ):
        self.n_bits_list = tuple(int(n_bits) for n_bits in n_bits_list)
        self.capacity_ratios = tuple(float(ratio) for ratio in capacity_ratios)
        self.noise_levels = tuple(float(level) for level in noise_levels)
        self.trial_ids = tuple(int(trial) for trial in trial_ids)
        self.grid = grid

    def tasks(self):
        """``(n_bits, capacity_ratio, noise_p, trial_id)`` in module order."""
        for n_bits in self.n_bits_list:
            for capacity_ratio in self.capacity_ratios:
                for noise_p in self.noise_levels:
                    for trial_id in self.trial_ids:
                        yield int(n_bits), float(capacity_ratio), float(noise_p), int(trial_id)


def historical_noise_levels():
    return tuple(float(level) for level in historical.NOISE_LEVELS)


def pattern_count(n_bits, capacity_ratio):
    """``max(1, int(n_bits * capacity_ratio))``, the count inside ``run_trial``."""
    return max(1, int(n_bits * capacity_ratio))


def task_seed(n_bits, capacity_ratio, noise_p, trial_id):
    """The integer ``run_trial`` passes to ``numpy.random.seed``."""
    return (
        int(historical.RANDOM_SEED_BASE)
        + int(trial_id)
        + int(n_bits * 1000)
        + int(capacity_ratio * 100)
        + int(noise_p * 1000)
    )


def historical_request():
    """The module grid. Building it does not recall and does not call ``main``."""
    return InferenceRequest(
        n_bits_list=tuple(int(n_bits) for n_bits in historical.N_BITS_LIST),
        capacity_ratios=tuple(float(ratio) for ratio in historical.CAPACITY_RATIOS),
        noise_levels=historical_noise_levels(),
        trial_ids=tuple(range(int(historical.TRIALS_PER_CONFIG))),
        grid="full",
    )


def historical_task_count():
    """Tasks in the module grid. This does not build or run the task list."""
    return (
        len(historical.N_BITS_LIST)
        * len(historical.CAPACITY_RATIOS)
        * len(historical.NOISE_LEVELS)
        * int(historical.TRIALS_PER_CONFIG)
    )


def is_historical_inference_request(request: InferenceRequest) -> bool:
    expected = historical_request()
    return (
        request.n_bits_list == expected.n_bits_list
        and request.capacity_ratios == expected.capacity_ratios
        and request.noise_levels == expected.noise_levels
        and request.trial_ids == expected.trial_ids
        and request.grid == "full"
    )


def lightweight_request():
    """One module cell: ``n=8``, capacity ``0.1``, the second noise level, trials ``0..299``.

    The noise value is ``NOISE_LEVELS[1]``. The other fifteen levels, the
    other capacities, and the other widths are not scored. The trial count
    stays 300 because ``aggregate_results`` divides its Gaussian columns
    by that constant.
    """
    return InferenceRequest(
        n_bits_list=(int(historical.N_BITS_LIST[0]),),
        capacity_ratios=(float(historical.CAPACITY_RATIOS[0]),),
        noise_levels=(historical_noise_levels()[1],),
        trial_ids=tuple(range(int(historical.TRIALS_PER_CONFIG))),
        grid="smoke",
    )


def run_inference(
    request: InferenceRequest,
    *,
    output_dir=None,
    overwrite: bool = False,
    experiment_id: str | None = None,
    configuration: dict | None = None,
    allow_historical_grid: bool = False,
):
    """Run the inference schedule and return the result mapping.

    ``allow_historical_grid`` starts the module grid. The default stays on
    the one-cell guard. A failure leaves ``execution_status`` ``failed``
    and is re-raised. The run is not resumed.
    """
    configure_logging()
    try:
        _require_registered_implementations()
        if allow_historical_grid:
            if not is_historical_inference_request(request):
                raise HistoricalGridRefused(
                    "allow_historical_grid starts only the inference module grid."
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


def rng_metadata(request: InferenceRequest) -> dict:
    """Name the seeds this request applies. This is not a reproducibility certificate."""
    return {
        "scope": PROTOCOL,
        "exact_reproducibility_claimed": False,
        "family": "legacy_global",
        "generator": PATTERN_GENERATOR,
        "pennylane_seed_call": False,
        "default_rng_used": False,
        "configured_seeds": {
            "role": "configured",
            "base": int(historical.RANDOM_SEED_BASE),
            "formula": SEED_FORMULA,
            "reapplied_for_every_task": True,
            "source": "experiments.noise_robustness_inference.run_trial",
        },
        "seeds_applied": [
            {
                "role": "applied",
                "when": "start of run_trial, before the pattern draw",
                "call": "numpy.random.seed",
                "example": {
                    "n_bits": request.n_bits_list[0],
                    "capacity_ratio": request.capacity_ratios[0],
                    "noise_p": request.noise_levels[0],
                    "trial_id": request.trial_ids[0],
                    "value": task_seed(
                        request.n_bits_list[0],
                        request.capacity_ratios[0],
                        request.noise_levels[0],
                        request.trial_ids[0],
                    ),
                },
            }
        ],
        "internal_resets": [],
        "unseeded_procedures": [
            {
                "name": "experiments.noise_robustness_inference.aggregate_results",
                "draws": "numpy.random.choice, 200 resamples, no seed call",
                "separate_seed": False,
            }
        ],
        "state_not_captured": [
            "The legacy RandomState is not serialized.",
            "Historical main draws the bootstrap on the pool parent, which did not run the tasks.",
            "This single-process runner draws the bootstrap after the last task.",
            "Those two streams are not the same position.",
            "Runtime columns are wall-clock measurements.",
            "A recorded seed formula does not by itself reproduce the archived summary percentiles.",
        ],
    }


def _manifest_fields(request: InferenceRequest) -> dict:
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "main_invoked": False,
        "plotting": False,
        "training": False,
        "procedure_ids": list(PROCEDURE_IDS),
        "csv_model_labels": dict(CSV_MODEL_LABEL),
        "historical_module": "experiments.noise_robustness_inference",
        "historical_trial": "experiments.noise_robustness_inference.run_trial",
        "historical_summary": "experiments.noise_robustness_inference.aggregate_results",
        "historical_tasks": historical_task_count(),
        "historical_trials_per_config": int(historical.TRIALS_PER_CONFIG),
        "historical_grid": request.grid == "full",
        "training_schedule": TRAINING_SCHEDULE,
        "bootstrap_samples": 200,
        "bootstrap_seeded": False,
        "gaussian_denominator": "TRIALS_PER_CONFIG",
        "process_model": FRAMEWORK_PROCESS,
        "historical_process_model": HISTORICAL_PROCESS,
        "historical_pool": {
            "context": "spawn",
            "workers": "min(12, os.cpu_count() or 1)",
            "order": "imap_unordered",
            "chunksize": 50,
        },
        "rng": rng_metadata(request),
    }


def _request_configuration(request: InferenceRequest) -> dict:
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "n_bits_list": list(request.n_bits_list),
        "capacity_ratios": list(request.capacity_ratios),
        "noise_levels": list(request.noise_levels),
        "trial_ids": list(request.trial_ids),
    }


def _dataset_settings(request: InferenceRequest) -> dict:
    return {
        "n_bits_list": list(request.n_bits_list),
        "capacity_ratios": list(request.capacity_ratios),
        "noise_levels": list(request.noise_levels),
        "pattern_generator": PATTERN_GENERATOR,
        "n_patterns": "max(1, int(n_bits * capacity_ratio))",
        "query": "one stored pattern per trial",
    }


def _perturbation_settings(request: InferenceRequest) -> dict:
    return {
        "procedure_id": "inline_bernoulli",
        "implementation": "experiments.noise_robustness_inference.run_trial",
        "levels": list(request.noise_levels),
        "rng": "numpy.random",
        "when": "after the target index, inside run_trial",
    }


def _require_registered_implementations():
    hopfield = get(HOPFIELD_ID)
    hamming = get(HAMMING_ID)
    if hopfield.implementation is not hopfield_async_sequential_recall:
        raise HistoricalGridRefused(
            "hopfield_async_sequential is not hopfield_async_sequential_recall."
        )
    if hamming.implementation is not nearest_neighbor_recall:
        raise HistoricalGridRefused(
            "hamming_nearest is not nearest_neighbor_recall."
        )


def _refuse_unless_smoke(request: InferenceRequest) -> None:
    if request.grid != "smoke":
        raise HistoricalGridRefused(
            "only grid='smoke' is implemented unless allow_historical_grid is set"
        )
    if request.trial_ids != tuple(range(int(historical.TRIALS_PER_CONFIG))):
        raise HistoricalGridRefused(
            f"trial ids stay 0..{int(historical.TRIALS_PER_CONFIG) - 1}."
        )
    if len(request.n_bits_list) > MAX_SMOKE_WIDTHS:
        raise HistoricalGridRefused(f"smoke widths must be at most {MAX_SMOKE_WIDTHS}")
    if len(request.capacity_ratios) > MAX_SMOKE_CAPACITIES:
        raise HistoricalGridRefused(
            f"smoke capacities must be at most {MAX_SMOKE_CAPACITIES}"
        )
    if len(request.noise_levels) > MAX_SMOKE_NOISE_LEVELS:
        raise HistoricalGridRefused(
            f"smoke noise levels must be at most {MAX_SMOKE_NOISE_LEVELS}"
        )
    if not request.n_bits_list or not request.capacity_ratios or not request.noise_levels:
        raise ValueError("widths, capacities, and noise levels must be non-empty")
    historical_widths = tuple(int(n_bits) for n_bits in historical.N_BITS_LIST)
    historical_capacities = tuple(float(ratio) for ratio in historical.CAPACITY_RATIOS)
    if request.n_bits_list[0] not in historical_widths:
        raise HistoricalGridRefused("smoke width must be one of the module widths")
    if request.capacity_ratios[0] not in historical_capacities:
        raise HistoricalGridRefused("smoke capacity must be one of the module ratios")
    if request.noise_levels[0] not in historical_noise_levels():
        raise HistoricalGridRefused("smoke noise must be one of the module levels")
    if len(tuple(request.tasks())) == historical_task_count():
        raise HistoricalGridRefused("the historical inference grid is not started here")


def _execute(request: InferenceRequest):
    specs = list(request.tasks())
    total = len(specs)
    log_event(
        "execution_start",
        tasks=total,
        trials_per_config=len(request.trial_ids),
        training=False,
        procedures=" ".join(PROCEDURE_IDS),
        csv_model_labels="hopfield qam",
    )
    rows = []
    raw_records = []
    for index, task in enumerate(specs, start=1):
        task_rows = historical.run_trial(task)
        if [record["model"] for record in task_rows] != ["hopfield", "qam"]:
            raise RuntimeError("run_trial did not return hopfield then qam")
        raw_records.extend(task_rows)
        for record in task_rows:
            procedure_id = LABEL_TO_PROCEDURE[record["model"]]
            rows.append(_annotate(request, record, procedure_id))
        if index % PROGRESS_STRIDE == 0 or index == total:
            log_event(
                "task_progress",
                progress=f"{index}/{total}",
                tasks=total,
                procedures=" ".join(PROCEDURE_IDS),
            )
    log_event(
        "aggregate_start",
        implementation="experiments.noise_robustness_inference.aggregate_results",
        bootstrap_samples=200,
        bootstrap_seeded=False,
        groups=len(request.n_bits_list)
        * len(request.capacity_ratios)
        * len(request.noise_levels)
        * len(PROCEDURE_IDS),
    )
    summary = _summarize(request, raw_records)
    log_event("aggregate_complete", rows=len(summary), resumable=False)
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "comparable_to_archive": False,
        "main_invoked": False,
        "training": False,
        "procedure_ids": list(PROCEDURE_IDS),
        "csv_model_labels": dict(CSV_MODEL_LABEL),
        "rows": rows,
        "summary": summary,
    }


def _annotate(request, record, procedure_id):
    annotated = {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "comparable_to_archive": False,
        "procedure_id": procedure_id,
    }
    for key, value in record.items():
        annotated[key] = _plain(value)
    return annotated


def _summarize(request, raw_records):
    frame = pd.DataFrame(raw_records)
    summary = historical.aggregate_results(frame)
    counts = {}
    for row in raw_records:
        key = _group_key(row)
        counts[key] = counts.get(key, 0) + 1
    records = []
    for item in summary.to_dict(orient="records"):
        plain = {key: _plain(value) for key, value in item.items()}
        procedure_id = LABEL_TO_PROCEDURE[plain["model"]]
        records.append(
            {
                "protocol": PROTOCOL,
                "grid": request.grid,
                "comparable_to_archive": False,
                "procedure_id": procedure_id,
                "sample_size": counts[_group_key(plain)],
                **plain,
            }
        )
    return records


def _group_key(row):
    return (
        row["model"],
        _plain(row["n_bits"]),
        _plain(row["capacity_ratio"]),
        _plain(row["noise_p"]),
    )


def _plain(value):
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if number != number:
            return None
        return number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def _write_tables(run_dir, payload):
    write_csv_table(run_dir, "rows.csv", payload["rows"], RAW_COLUMNS)
    write_json(run_dir, "rows.json", payload["rows"])
    write_csv_table(run_dir, "summary.csv", payload["summary"], SUMMARY_COLUMNS)
    write_json(run_dir, "summary.json", payload["summary"])
