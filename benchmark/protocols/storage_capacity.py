"""Storage-capacity protocol.

Each cell is one ``(n_bits, m)``. Every cell reseeds ``numpy.random`` to
``0 .. trials_per_m-1``, draws patterns, trains synchronous Hopfield, scores
fixed-count queries, then trains the RY-linear QAM and scores a second
fixed-count pass. That is the order in ``experiments.storage_capacity.main``.

After the five trials, means and percentile intervals come from
``bootstrap_ci``, which passes the global NumPy stream into
``bootstrap_mean_ci``. There is no separate bootstrap seed. Wilcoxon receives
Hopfield scores then QAM scores, and a failure stores ``None``. The critical
ratio is the first ``m / n_bits``, in ascending order, whose mean full
accuracy is strictly below ``0.75``.

This module does not call ``main`` and it does not plot. The default request
is one historical cell. The 33-cell module grid starts only when
``allow_historical_grid`` is true.
"""

from __future__ import annotations

import time

import numpy as np
from pennylane import numpy as pnp

import experiments.storage_capacity as historical
from benchmark.adapters import hopfield_synchronous, qam_ry_linear
from benchmark.data.corruption import apply_fixed_count
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

PROTOCOL = "storage_capacity"
HOPFIELD_ID = hopfield_synchronous.identity.procedure_id
QAM_ID = qam_ry_linear.identity.procedure_id
PROCEDURE_IDS = (HOPFIELD_ID, QAM_ID)
HOPFIELD_LABEL = "Hopfield"
QAM_LABEL = "QAM"
PATTERN_GENERATOR = "numpy.random"
M_RULE = "range_1_through_n_or_reduced_list_at_n_ge_16"
TRAINING_SCHEDULE = "reseed_each_trial_then_hopfield_then_qam"
CRITICAL_THRESHOLD = 0.75
WILCOXON_SENTINEL = WILCOXON_FAILURE["storage_capacity"]
N_AT_LEAST_16_M = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16)
FRAMEWORK_PROCESS = "single_process"
HISTORICAL_PROCESS = "single_process"

MAX_SMOKE_WIDTHS = 1
MAX_SMOKE_M_VALUES = 1

SWEEP_COLUMNS = (
    "protocol",
    "grid",
    "comparable_to_archive",
    "procedure_id",
    "historical_model_label",
    "n_bits",
    "m",
    "capacity_ratio",
    "full_acc_mean",
    "full_acc_ci_low",
    "full_acc_ci_high",
    "bit_acc_mean",
    "bit_acc_ci_low",
    "bit_acc_ci_high",
    "train_time_s_mean",
    "infer_time_s_mean",
    "params_count",
    "recall_noise",
    "p_value_full_vs_qam",
    "p_value_bit_vs_qam",
    "extra_notes",
    "sample_size",
    "bootstrap_samples",
)

PHASE_COLUMNS = (
    "protocol",
    "grid",
    "comparable_to_archive",
    "procedure_id",
    "historical_model_label",
    "n_bits",
    "critical_capacity_ratio",
)

SIGNIFICANCE_COLUMNS = (
    "protocol",
    "grid",
    "comparable_to_archive",
    "n_bits",
    "m",
    "capacity_ratio",
    "p_value_full_vs_qam",
    "p_value_bit_vs_qam",
    "sample_size",
)

_RESULT_FILES = (
    "manifest.json",
    "configuration.json",
    "sweep.csv",
    "sweep.json",
    "phase_transition.csv",
    "phase_transition.json",
    "significance.csv",
    "significance.json",
    "trials.json",
)


class HistoricalGridRefused(RuntimeError):
    """The 33-cell storage-capacity grid is not executed here."""


class StorageCapacityRequest:
    """Widths, memory loads, and the trial budget. Building it does not train."""

    def __init__(
        self,
        widths,
        seeds,
        train_steps,
        lr,
        recall_noise,
        bootstrap_samples,
        hopfield_max_factor=1.0,
        qam_max_factor=1.0,
        reduced_scale_for_large_n=True,
        grid="smoke",
    ):
        self.widths = tuple(
            (int(n_bits), tuple(int(m) for m in m_values)) for n_bits, m_values in widths
        )
        self.seeds = tuple(int(seed) for seed in seeds)
        self.train_steps = int(train_steps)
        self.lr = float(lr)
        self.recall_noise = float(recall_noise)
        self.bootstrap_samples = int(bootstrap_samples)
        self.hopfield_max_factor = float(hopfield_max_factor)
        self.qam_max_factor = float(qam_max_factor)
        self.reduced_scale_for_large_n = bool(reduced_scale_for_large_n)
        self.grid = grid

    def cells(self):
        """``(n_bits, m)`` in module order. Each cell repeats ``seeds``."""
        for n_bits, m_values in self.widths:
            for m in m_values:
                yield int(n_bits), int(m)


def m_values_for(
    n_bits,
    *,
    reduced_scale_for_large_n=True,
    hopfield_max_factor=1.0,
    qam_max_factor=1.0,
):
    """The ``m`` loop in ``main``.

    ``n >= 16`` with the module flag uses the literal list ``1..10, 12, 14, 16``.
    Other widths use ``range(1, max_capacity + 1)``.
    """
    max_m_hopfield = int(n_bits * hopfield_max_factor)
    max_m_qam = int(n_bits * qam_max_factor)
    max_capacity = max(max_m_hopfield, max_m_qam)
    if reduced_scale_for_large_n and n_bits >= 16:
        return N_AT_LEAST_16_M
    return tuple(range(1, max_capacity + 1))


def n_flips_for(n_bits, recall_noise):
    """``max(1, int(n_bits * recall_noise))``, the fixed-count count."""
    return max(1, int(n_bits * recall_noise))


def depth_for(n_bits):
    """``depth_policy`` from the module, ``min(8, n_bits)``."""
    return int(historical.depth_policy(n_bits))


def historical_request():
    """The module grid. Building it does not train and does not call ``main``."""
    widths = tuple(
        (int(n_bits), m_values_for(int(n_bits))) for n_bits in historical.n_bits_list
    )
    return StorageCapacityRequest(
        widths=widths,
        seeds=tuple(range(int(historical.trials_per_m))),
        train_steps=int(historical.train_steps),
        lr=float(historical.lr),
        recall_noise=float(historical.recall_noise),
        bootstrap_samples=int(historical.bootstrap_samples),
        hopfield_max_factor=float(historical.hopfield_max_factor),
        qam_max_factor=float(historical.qam_max_factor),
        reduced_scale_for_large_n=bool(historical.reduced_scale_for_large_n),
        grid="full",
    )


def historical_cell_count():
    """Cells in the module grid. This does not run them."""
    return len(tuple(historical_request().cells()))


def is_historical_request(request: StorageCapacityRequest) -> bool:
    expected = historical_request()
    return (
        request.widths == expected.widths
        and request.seeds == expected.seeds
        and int(request.train_steps) == expected.train_steps
        and float(request.lr) == expected.lr
        and float(request.recall_noise) == expected.recall_noise
        and int(request.bootstrap_samples) == expected.bootstrap_samples
        and float(request.hopfield_max_factor) == expected.hopfield_max_factor
        and float(request.qam_max_factor) == expected.qam_max_factor
        and bool(request.reduced_scale_for_large_n) is True
        and request.grid == "full"
    )


def lightweight_request():
    """The first module cell: ``n=8``, ``m=1``, seeds ``0..4``, 100 steps.

    The memory-load rule at width 8 is still ``1..8``. This request scores
    ``m=1`` only. Depth, step count, noise, trial count, and the bootstrap
    count stay the module constants.
    """
    return StorageCapacityRequest(
        widths=((int(historical.n_bits_list[0]), (1,)),),
        seeds=tuple(range(int(historical.trials_per_m))),
        train_steps=int(historical.train_steps),
        lr=float(historical.lr),
        recall_noise=float(historical.recall_noise),
        bootstrap_samples=int(historical.bootstrap_samples),
        grid="smoke",
    )


def run_storage_capacity(
    request: StorageCapacityRequest,
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
                    "allow_historical_grid starts only the storage-capacity module grid."
                )
        else:
            _refuse_unless_smoke(request)
    except Exception as exc:
        log_failure("request_refused", exc, protocol=PROTOCOL)
        raise

    identity = experiment_id or PROTOCOL
    n_cells = len(tuple(request.cells()))
    if output_dir is None:
        log_event(
            "run_start",
            experiment_id=identity,
            cells=n_cells,
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
        cells=n_cells,
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
                "smoke_cells": None if allow_historical_grid else n_cells,
                "historical_cells": historical_cell_count(),
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


def rng_metadata(request: StorageCapacityRequest) -> dict:
    """Name the seeds this request applies. This is not a reproducibility certificate."""
    applied = [
        {
            "n_bits": n_bits,
            "m": m,
            "value": seed,
            "call": "numpy.random.seed",
        }
        for n_bits, m in request.cells()
        for seed in request.seeds
    ]
    return {
        "scope": PROTOCOL,
        "exact_reproducibility_claimed": False,
        "family": "legacy_global",
        "generator": PATTERN_GENERATOR,
        "pennylane_seed_call": False,
        "default_rng_used": False,
        "configured_seeds": {
            "role": "configured",
            "values": list(request.seeds),
            "formula": "the trial index in range(trials_per_m)",
            "reapplied_for_every_cell": True,
            "source": "request.seeds",
        },
        "seeds_applied": [
            {
                "role": "applied",
                "when": "start of each trial, before the pattern draw",
                "values": applied,
            }
        ],
        "internal_resets": [],
        "bootstrap": {
            "role": "continues_trial_stream",
            "name": "experiments.storage_capacity.bootstrap_ci",
            "implementation": "src.metrics.bootstrap_mean_ci",
            "rng": "numpy.random",
            "separate_seed": False,
            "samples": int(request.bootstrap_samples),
            "detail": (
                "bootstrap_ci passes numpy.random into bootstrap_mean_ci and "
                "does not call seed. This runner does not add a seed. The draws "
                "start after the last trial of that cell."
            ),
        },
        "unseeded_procedures": [
            {
                "role": "not_called",
                "name": "experiments.storage_capacity.confidence_interval",
                "detail": "The student-t helper has no caller in main, and this runner does not call it.",
            }
        ],
        "state_not_captured": [
            "The legacy RandomState is not serialized.",
            "RY-linear init draws pennylane.numpy.random.randn on that same stream.",
            "Fixed-count indexes use numpy.random through the rng argument.",
            "Bootstrap resamples continue after the last trial of the cell.",
            "Train-time and inference-time means are wall-clock measurements.",
            "A recorded seed list does not by itself reproduce the archived sweep.",
        ],
    }


def summarize_cell(
    *,
    hopfield_full,
    hopfield_bit,
    hopfield_train_times,
    hopfield_infer_times,
    qam_full,
    qam_bit,
    qam_train_times,
    qam_infer_times,
    qam_param_counts,
    bootstrap_samples,
):
    """Bootstrap, then Wilcoxon, in the order ``main`` uses.

    Hopfield train and inference means are ``numpy.mean``. QAM train and
    inference means are the bootstrap mean. The bootstrap calls are not
    seeded here. ``qam_*`` may be empty when that model is outside the cell.
    """
    interval = selected(PROTOCOL, "bootstrap_mean_ci")
    hopfield_full_stats = interval(hopfield_full, rng=np.random, n_boot=int(bootstrap_samples))
    hopfield_bit_stats = interval(hopfield_bit, rng=np.random, n_boot=int(bootstrap_samples))
    qam_full_stats = None
    qam_bit_stats = None
    qam_train_mean = None
    qam_infer_mean = None
    qam_params_mean = None
    if len(qam_full) > 0:
        qam_full_stats = interval(qam_full, rng=np.random, n_boot=int(bootstrap_samples))
        qam_bit_stats = interval(qam_bit, rng=np.random, n_boot=int(bootstrap_samples))
        qam_train_mean = interval(qam_train_times, rng=np.random, n_boot=int(bootstrap_samples))[0]
        qam_infer_mean = interval(qam_infer_times, rng=np.random, n_boot=int(bootstrap_samples))[0]
        qam_params_mean = float(np.mean(np.asarray(qam_param_counts, dtype=float)))
    p_full = None
    p_bit = None
    if len(qam_full) > 0:
        wilcoxon = selected(PROTOCOL, "wilcoxon_pvalue")
        p_full = wilcoxon(hopfield_full, qam_full, on_failure=WILCOXON_SENTINEL)
        p_bit = wilcoxon(hopfield_bit, qam_bit, on_failure=WILCOXON_SENTINEL)
    return {
        "hopfield_full": hopfield_full_stats,
        "hopfield_bit": hopfield_bit_stats,
        "hopfield_train_mean": float(np.mean(np.asarray(hopfield_train_times, dtype=float))),
        "hopfield_infer_mean": float(np.mean(np.asarray(hopfield_infer_times, dtype=float))),
        "qam_full": qam_full_stats,
        "qam_bit": qam_bit_stats,
        "qam_train_mean": None if qam_train_mean is None else float(qam_train_mean),
        "qam_infer_mean": None if qam_infer_mean is None else float(qam_infer_mean),
        "qam_params_mean": qam_params_mean,
        "p_value_full": _json_number(p_full),
        "p_value_bit": _json_number(p_bit),
    }


def phase_transition_rows(sweep_rows, *, grid):
    """First capacity ratio whose mean full accuracy is strictly below 0.75.

    Rows are scanned in ascending ``capacity_ratio``, which is ``m / n_bits``.
    ``None`` when every stored mean stays at or above the threshold.
    """
    critical = selected(PROTOCOL, "critical_capacity_ratio")
    produced = []
    for label, procedure_id in ((HOPFIELD_LABEL, HOPFIELD_ID), (QAM_LABEL, QAM_ID)):
        widths = sorted(
            {
                int(row["n_bits"])
                for row in sweep_rows
                if row["historical_model_label"] == label and row["full_acc_mean"] is not None
            }
        )
        for n_bits in widths:
            members = [
                row
                for row in sweep_rows
                if row["historical_model_label"] == label
                and int(row["n_bits"]) == n_bits
                and row["full_acc_mean"] is not None
            ]
            members.sort(key=lambda row: float(row["capacity_ratio"]))
            ratio = critical(
                [float(row["capacity_ratio"]) for row in members],
                [float(row["full_acc_mean"]) for row in members],
                threshold=CRITICAL_THRESHOLD,
            )
            produced.append(
                {
                    "protocol": PROTOCOL,
                    "grid": grid,
                    "comparable_to_archive": False,
                    "procedure_id": procedure_id,
                    "historical_model_label": label,
                    "n_bits": n_bits,
                    "critical_capacity_ratio": None if ratio is None else float(ratio),
                }
            )
    return produced


def significance_rows(sweep_rows, *, grid):
    """One row per ``(n_bits, m)`` after dropping repeated p-value columns.

    ``main`` keeps the first duplicate. Hopfield is appended first, so the
    retained p-values are the pair stored on both model rows.
    """
    seen = set()
    produced = []
    for row in sweep_rows:
        key = (
            int(row["n_bits"]),
            int(row["m"]),
            float(row["capacity_ratio"]),
            row["p_value_full_vs_qam"],
            row["p_value_bit_vs_qam"],
        )
        if key in seen:
            continue
        seen.add(key)
        produced.append(
            {
                "protocol": PROTOCOL,
                "grid": grid,
                "comparable_to_archive": False,
                "n_bits": key[0],
                "m": key[1],
                "capacity_ratio": key[2],
                "p_value_full_vs_qam": key[3],
                "p_value_bit_vs_qam": key[4],
                "sample_size": int(row["sample_size"]),
            }
        )
    return produced


def _manifest_fields(request: StorageCapacityRequest) -> dict:
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "main_invoked": False,
        "plotting": False,
        "procedure_ids": list(PROCEDURE_IDS),
        "historical_module": "experiments.storage_capacity",
        "historical_qam": "src.qam.train_ry_linear_qam",
        "historical_train_steps": int(historical.train_steps),
        "historical_cells": historical_cell_count(),
        "historical_trials_per_cell": int(historical.trials_per_m),
        "historical_grid": request.grid == "full",
        "training_schedule": TRAINING_SCHEDULE,
        "m_rule": M_RULE,
        "critical_threshold": CRITICAL_THRESHOLD,
        "critical_comparison": "strictly_below",
        "bootstrap_samples": int(request.bootstrap_samples),
        "bootstrap_separate_seed": False,
        "wilcoxon_argument_order": ["hopfield", "qam"],
        "wilcoxon_failure_value": WILCOXON_SENTINEL,
        "process_model": FRAMEWORK_PROCESS,
        "historical_process_model": HISTORICAL_PROCESS,
        "rng": rng_metadata(request),
    }


def _request_configuration(request: StorageCapacityRequest) -> dict:
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "widths": [
            {"n_bits": n_bits, "m_values": list(m_values)} for n_bits, m_values in request.widths
        ],
        "seeds": list(request.seeds),
        "train_steps": int(request.train_steps),
        "lr": float(request.lr),
        "recall_noise": float(request.recall_noise),
        "bootstrap_samples": int(request.bootstrap_samples),
    }


def _dataset_settings(request: StorageCapacityRequest) -> dict:
    return {
        "widths": [
            {"n_bits": n_bits, "m_values": list(m_values)} for n_bits, m_values in request.widths
        ],
        "pattern_generator": PATTERN_GENERATOR,
        "m_rule": M_RULE,
    }


def _perturbation_settings(request: StorageCapacityRequest) -> dict:
    return {
        "procedure_id": "fixed_count",
        "implementation": "src.data_generation.corrupt_fixed_count",
        "level": float(request.recall_noise),
        "rng": "numpy.random",
    }


def _refuse_unless_smoke(request: StorageCapacityRequest) -> None:
    if request.grid != "smoke":
        raise HistoricalGridRefused(
            "only grid='smoke' is implemented unless allow_historical_grid is set"
        )
    if int(request.train_steps) != int(historical.train_steps):
        raise HistoricalGridRefused(
            f"train_steps stays {int(historical.train_steps)}. "
            "A shorter budget would change the RY-linear training."
        )
    if request.seeds != tuple(range(int(historical.trials_per_m))):
        raise HistoricalGridRefused(
            f"seeds stay {tuple(range(int(historical.trials_per_m)))}. "
            "Wilcoxon and the bootstrap are computed on that trial count."
        )
    if int(request.bootstrap_samples) != int(historical.bootstrap_samples):
        raise HistoricalGridRefused(
            f"bootstrap_samples stays {int(historical.bootstrap_samples)}."
        )
    if float(request.lr) != float(historical.lr):
        raise HistoricalGridRefused(f"lr stays {float(historical.lr)}.")
    if float(request.recall_noise) != float(historical.recall_noise):
        raise HistoricalGridRefused(f"recall_noise stays {float(historical.recall_noise)}.")
    if float(request.hopfield_max_factor) != float(historical.hopfield_max_factor):
        raise HistoricalGridRefused("hopfield_max_factor stays 1.0.")
    if float(request.qam_max_factor) != float(historical.qam_max_factor):
        raise HistoricalGridRefused("qam_max_factor stays 1.0.")
    if request.reduced_scale_for_large_n is not True:
        raise HistoricalGridRefused(
            "reduced_scale_for_large_n stays true. That flag is the module's n>=16 list."
        )
    if len(request.widths) > MAX_SMOKE_WIDTHS:
        raise HistoricalGridRefused(f"smoke widths must be at most {MAX_SMOKE_WIDTHS}")
    if not request.widths:
        raise ValueError("widths must be non-empty")
    n_bits, m_values = request.widths[0]
    if len(m_values) > MAX_SMOKE_M_VALUES:
        raise HistoricalGridRefused(f"smoke m values must be at most {MAX_SMOKE_M_VALUES}")
    full_m = m_values_for(n_bits)
    if n_bits not in tuple(int(value) for value in historical.n_bits_list):
        raise HistoricalGridRefused("smoke n_bits must be one of the module lengths")
    if not m_values or any(m not in full_m for m in m_values):
        raise HistoricalGridRefused("smoke m must be taken from the module list for that width")
    if len(tuple(request.cells())) == historical_cell_count():
        raise HistoricalGridRefused("the historical storage-capacity grid is not started here")


def _execute(request: StorageCapacityRequest):
    specs = list(request.cells())
    total = len(specs)
    log_event(
        "execution_start",
        cells=total,
        trials_per_cell=len(request.seeds),
        train_steps=int(request.train_steps),
        bootstrap_samples=int(request.bootstrap_samples),
        bootstrap_separate_seed=False,
    )
    sweep = []
    trials = []
    for index, (n_bits, m) in enumerate(specs, start=1):
        log_event(
            "cell_start",
            progress=f"{index}/{total}",
            n_bits=n_bits,
            m=m,
        )
        cell_sweep, cell_trials = _run_cell(request, n_bits, m)
        sweep.extend(cell_sweep)
        trials.extend(cell_trials)
        log_event(
            "cell_complete",
            progress=f"{index}/{total}",
            n_bits=n_bits,
            m=m,
        )
    phases = phase_transition_rows(sweep, grid=request.grid)
    significance = significance_rows(sweep, grid=request.grid)
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "comparable_to_archive": False,
        "main_invoked": False,
        "procedure_ids": list(PROCEDURE_IDS),
        "sweep": sweep,
        "phase_transition": phases,
        "significance": significance,
        "trials": trials,
    }


def _run_cell(request: StorageCapacityRequest, n_bits, m):
    depth = depth_for(n_bits)
    max_m_qam = int(n_bits * request.qam_max_factor)
    score = selected(PROTOCOL, "batch_recall")
    hopfield_adapter = get(HOPFIELD_ID).adapter
    qam_adapter = get(QAM_ID).adapter
    hop_full, hop_bit = [], []
    hop_train_times, hop_infer_times = [], []
    qam_full, qam_bit = [], []
    qam_train_times, qam_infer_times, qam_param_counts = [], [], []
    trials = []
    run_qam = m <= max_m_qam
    for seed_index, seed in enumerate(request.seeds, start=1):
        log_event(
            "trial_start",
            progress=f"{seed_index}/{len(request.seeds)}",
            n_bits=n_bits,
            m=m,
            seed=seed,
        )
        np.random.seed(seed)
        patterns = np.random.choice([-1, 1], size=(m, n_bits))
        factory, args, kwargs = prepare(HOPFIELD_ID, {"n_bits": n_bits, "steps": 10})
        hopfield = factory(*args, **kwargs)
        log_event("model_construct", procedure_id=HOPFIELD_ID, n_bits=n_bits, m=m, seed=seed)
        started = time.time()
        hopfield_adapter.train(hopfield, patterns)
        hop_train_times.append(time.time() - started)
        log_event("model_train", procedure_id=HOPFIELD_ID, n_bits=n_bits, m=m, seed=seed)
        started = time.time()
        recalled = pnp.array(
            [
                hopfield_adapter.recall(
                    hopfield, apply_fixed_count(pattern, request.recall_noise, rng=np.random)
                )
                for pattern in patterns
            ]
        )
        hop_infer_times.append((time.time() - started) / m)
        full, bit = score(recalled, patterns)
        hop_full.append(float(full))
        hop_bit.append(float(bit))
        log_event(
            "model_evaluate",
            procedure_id=HOPFIELD_ID,
            n_bits=n_bits,
            m=m,
            seed=seed,
            queries=m,
        )
        qam_params = None
        qam_full_score = None
        qam_bit_score = None
        if run_qam:
            prepare(QAM_ID, {"n_bits": n_bits, "depth": depth, "return_tuple": True})
            log_event(
                "model_construct",
                procedure_id=QAM_ID,
                n_bits=n_bits,
                m=m,
                seed=seed,
                depth=depth,
                steps=int(request.train_steps),
            )
            params, circuit, train_time = qam_adapter.train(
                patterns,
                n_bits,
                depth,
                steps=int(request.train_steps),
                lr=float(request.lr),
                return_tuple=True,
            )
            qam_train_times.append(float(train_time))
            log_event(
                "model_train",
                procedure_id=QAM_ID,
                n_bits=n_bits,
                m=m,
                seed=seed,
                steps=int(request.train_steps),
            )
            started = time.time()
            recalled_qam = pnp.array(
                [
                    qam_adapter.recall(
                        apply_fixed_count(pattern, request.recall_noise, rng=np.random),
                        circuit,
                        params,
                    )
                    for pattern in patterns
                ]
            )
            qam_infer_times.append((time.time() - started) / m)
            qam_full_score, qam_bit_score = score(recalled_qam, patterns)
            qam_full.append(float(qam_full_score))
            qam_bit.append(float(qam_bit_score))
            qam_param_counts.append(int(np.asarray(params).size))
            qam_params = np.asarray(params, dtype=float).tolist()
            log_event(
                "model_evaluate",
                procedure_id=QAM_ID,
                n_bits=n_bits,
                m=m,
                seed=seed,
                queries=m,
            )
        trials.append(
            {
                "n_bits": int(n_bits),
                "m": int(m),
                "seed": int(seed),
                "depth": int(depth),
                "train_steps": int(request.train_steps),
                "patterns": np.asarray(patterns).astype(int).tolist(),
                "hopfield_full": float(full),
                "hopfield_bit": float(bit),
                "qam_full": None if qam_full_score is None else float(qam_full_score),
                "qam_bit": None if qam_bit_score is None else float(qam_bit_score),
                "qam_params": qam_params,
                "qam_procedure_id": QAM_ID if run_qam else None,
                "hopfield_procedure_id": HOPFIELD_ID,
            }
        )
        log_event(
            "trial_complete",
            progress=f"{seed_index}/{len(request.seeds)}",
            n_bits=n_bits,
            m=m,
            seed=seed,
        )
    log_event(
        "statistics_start",
        n_bits=n_bits,
        m=m,
        bootstrap_samples=int(request.bootstrap_samples),
        bootstrap_separate_seed=False,
        sample_size=len(request.seeds),
    )
    stats = summarize_cell(
        hopfield_full=hop_full,
        hopfield_bit=hop_bit,
        hopfield_train_times=hop_train_times,
        hopfield_infer_times=hop_infer_times,
        qam_full=qam_full,
        qam_bit=qam_bit,
        qam_train_times=qam_train_times,
        qam_infer_times=qam_infer_times,
        qam_param_counts=qam_param_counts,
        bootstrap_samples=int(request.bootstrap_samples),
    )
    log_event("statistics_complete", n_bits=n_bits, m=m, resumable=False)
    ratio = m / n_bits
    notes = "Associative recall from corrupted inputs"
    qam_notes = (
        f"Variational QAM | depth={depth} | steps={int(request.train_steps)} | lr={float(request.lr)}"
    )
    sweep = [
        _sweep_row(
            request,
            procedure_id=HOPFIELD_ID,
            label=HOPFIELD_LABEL,
            n_bits=n_bits,
            m=m,
            ratio=ratio,
            full_stats=stats["hopfield_full"],
            bit_stats=stats["hopfield_bit"],
            train_mean=stats["hopfield_train_mean"],
            infer_mean=stats["hopfield_infer_mean"],
            params_count=n_bits ** 2,
            p_full=stats["p_value_full"],
            p_bit=stats["p_value_bit"],
            notes=notes,
        )
    ]
    if run_qam:
        sweep.append(
            _sweep_row(
                request,
                procedure_id=QAM_ID,
                label=QAM_LABEL,
                n_bits=n_bits,
                m=m,
                ratio=ratio,
                full_stats=stats["qam_full"],
                bit_stats=stats["qam_bit"],
                train_mean=stats["qam_train_mean"],
                infer_mean=stats["qam_infer_mean"],
                params_count=stats["qam_params_mean"],
                p_full=stats["p_value_full"],
                p_bit=stats["p_value_bit"],
                notes=qam_notes,
            )
        )
    return sweep, trials


def _sweep_row(
    request,
    *,
    procedure_id,
    label,
    n_bits,
    m,
    ratio,
    full_stats,
    bit_stats,
    train_mean,
    infer_mean,
    params_count,
    p_full,
    p_bit,
    notes,
):
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "comparable_to_archive": False,
        "procedure_id": procedure_id,
        "historical_model_label": label,
        "n_bits": int(n_bits),
        "m": int(m),
        "capacity_ratio": float(ratio),
        "full_acc_mean": float(full_stats[0]),
        "full_acc_ci_low": float(full_stats[1]),
        "full_acc_ci_high": float(full_stats[2]),
        "bit_acc_mean": float(bit_stats[0]),
        "bit_acc_ci_low": float(bit_stats[1]),
        "bit_acc_ci_high": float(bit_stats[2]),
        "train_time_s_mean": float(train_mean),
        "infer_time_s_mean": float(infer_mean),
        "params_count": float(params_count),
        "recall_noise": float(request.recall_noise),
        "p_value_full_vs_qam": p_full,
        "p_value_bit_vs_qam": p_bit,
        "extra_notes": notes,
        "sample_size": len(request.seeds),
        "bootstrap_samples": int(request.bootstrap_samples),
    }


def _write_tables(run_dir, payload):
    write_csv_table(run_dir, "sweep.csv", payload["sweep"], SWEEP_COLUMNS)
    write_json(run_dir, "sweep.json", payload["sweep"])
    write_csv_table(run_dir, "phase_transition.csv", payload["phase_transition"], PHASE_COLUMNS)
    write_json(
        run_dir,
        "phase_transition.json",
        {
            "threshold": CRITICAL_THRESHOLD,
            "comparison": "strictly_below",
            "scan": "capacity_ratio ascending",
            "rows": payload["phase_transition"],
        },
    )
    write_csv_table(run_dir, "significance.csv", payload["significance"], SIGNIFICANCE_COLUMNS)
    write_json(
        run_dir,
        "significance.json",
        {
            "argument_order": ["hopfield", "qam"],
            "on_failure": WILCOXON_SENTINEL,
            "rows": payload["significance"],
        },
    )
    write_json(run_dir, "trials.json", payload["trials"])


def _json_number(value):
    if value is None:
        return None
    return float(value)
