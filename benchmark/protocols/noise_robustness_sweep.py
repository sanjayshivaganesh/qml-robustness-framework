"""Small paired noise-sweep run. Does not call the historical ``main``.

Order for one cell, matching ``experiments.noise_robustness_sweep.run_trial``:

1. Set both global seed functions to the trial seed.
2. Draw ``m = max(1, int(capacity_ratio * n_bits))`` bipolar patterns.
3. Train synchronous Hopfield and recall every Bernoulli-corrupted pattern.
4. Construct Rot full-batch QAM. That constructor reseeds the shared stream
   to 42 and draws parameters. Train, then Bernoulli-corrupt and recall again.

The two arms therefore do not see the same queries. A shared trial seed is
not a shared random sequence after step 4.

This module does not plot, does not start a process pool, and does not run
the historical grid (4 widths, 5 rates, 30 seeds, 300 Adam steps).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pennylane import numpy as pnp

import experiments.noise_robustness_sweep as historical
from benchmark.adapters import hopfield_synchronous, qam_rot_fullbatch
from benchmark.data.corruption import BERNOULLI, CorruptionSpec, perturb
from benchmark.log import configure_logging, log_event, log_failure
from benchmark.registry import get, prepare
from benchmark.data.patterns import GENERATOR_NUMPY_GLOBAL, draw_bipolar_patterns
from benchmark.output import (
    allocate_run_directory,
    begin_run,
    complete_run,
    create_run_dir,
    fail_run,
    write_csv_table,
    write_json,
)
from benchmark.scoring import selected

PROTOCOL = "noise_robustness_sweep"
HOPFIELD_ID = hopfield_synchronous.identity.procedure_id
QAM_ROT_FULLBATCH_ID = qam_rot_fullbatch.identity.procedure_id
PROCEDURE_IDS = (HOPFIELD_ID, QAM_ROT_FULLBATCH_ID)

MAX_SMOKE_CELLS = 4
MAX_SMOKE_TRAIN_STEPS = 5

RNG_LIMITATIONS = (
    "numpy.random.seed and pennylane.numpy.random.seed are two functions on one legacy stream. A Generator from default_rng is a different stream.",
    "The trial seed is applied before the pattern draw and the Hopfield Bernoulli draws. It does not survive QAM construction.",
    "experiments.noise_robustness_sweep.QAM.__init__ calls pennylane.numpy.random.seed(42) and then draws parameters. QAM Bernoulli draws continue from that reset, not from the trial seed.",
    "This runner is single-process. The historical main uses a multiprocessing pool. Seeds are set inside each trial; wall-clock columns are not an acceptance criterion.",
    "Smoke train_steps is not the historical 300. The archived CSV is a 30-trial aggregate with no per-trial rows. comparable_to_archive is false.",
)

TRIAL_COLUMNS = (
    "protocol",
    "grid",
    "comparable_to_archive",
    "procedure_id",
    "source",
    "trial_seed",
    "n_bits",
    "n_patterns",
    "capacity_ratio",
    "pattern_generator",
    "perturbation_id",
    "perturbation_level",
    "full_accuracy",
    "bit_accuracy",
    "train_time_seconds",
    "mean_recall_time_per_bit",
    "constructor_reseed",
    "train_steps",
)

AGGREGATE_COLUMNS = (
    "protocol",
    "grid",
    "comparable_to_archive",
    "procedure_id",
    "n_bits",
    "n_patterns",
    "perturbation_id",
    "perturbation_level",
    "n_trials",
    "full_accuracy_mean",
    "full_accuracy_std",
    "full_accuracy_ci",
    "bit_accuracy_mean",
    "bit_accuracy_std",
    "bit_accuracy_ci",
    "train_time_seconds_mean",
    "mean_recall_time_per_bit_mean",
)

COLUMN_ROLES = {
    "procedure_id": "identity",
    "source": "identity",
    "trial_seed": "dataset",
    "n_bits": "dataset",
    "n_patterns": "dataset",
    "capacity_ratio": "dataset",
    "pattern_generator": "dataset",
    "perturbation_id": "perturbation",
    "perturbation_level": "perturbation",
    "full_accuracy": "metric",
    "bit_accuracy": "metric",
    "full_accuracy_mean": "metric",
    "full_accuracy_std": "metric",
    "full_accuracy_ci": "metric",
    "bit_accuracy_mean": "metric",
    "bit_accuracy_std": "metric",
    "bit_accuracy_ci": "metric",
    "train_time_seconds": "timing",
    "mean_recall_time_per_bit": "timing",
    "train_time_seconds_mean": "timing",
    "mean_recall_time_per_bit_mean": "timing",
    "constructor_reseed": "metadata",
    "train_steps": "metadata",
    "protocol": "metadata",
    "grid": "metadata",
    "comparable_to_archive": "metadata",
    "n_trials": "metadata",
}


class HistoricalGridRefused(RuntimeError):
    """The historical noise sweep is not executed by this runner."""


@dataclass(frozen=True)
class NoiseSweepRequest:
    """Explicit smoke parameters. Not a YAML document and not the 600-cell grid."""

    n_bits_list: tuple[int, ...]
    corruption_rates: tuple[float, ...]
    seeds: tuple[int, ...]
    capacity_ratio: float = 0.3
    train_steps: int = 1
    grid: str = "smoke"

    def cells(self):
        for n_bits in self.n_bits_list:
            for corruption in self.corruption_rates:
                for seed in self.seeds:
                    yield int(n_bits), float(corruption), int(seed)


def historical_noise_sweep_request():
    """The module grid: 4 widths, 5 Bernoulli rates, seeds ``0..29``, 300 Adam steps.

    Building this request does not train and does not call ``main``.
    ``run_noise_sweep`` starts it only when ``allow_historical_grid`` is true.
    """
    return NoiseSweepRequest(
        n_bits_list=tuple(int(n_bits) for n_bits in historical.n_bits_list),
        corruption_rates=tuple(float(rate) for rate in historical.corruption_rates),
        seeds=tuple(range(int(historical.trials))),
        capacity_ratio=float(historical.capacity_ratio),
        train_steps=int(historical.train_steps),
        grid="full",
    )


def is_historical_noise_sweep_request(request: NoiseSweepRequest) -> bool:
    """True when every scientific field matches ``historical_noise_sweep_request``."""
    expected = historical_noise_sweep_request()
    return (
        tuple(request.n_bits_list) == expected.n_bits_list
        and tuple(request.corruption_rates) == expected.corruption_rates
        and tuple(request.seeds) == expected.seeds
        and float(request.capacity_ratio) == expected.capacity_ratio
        and int(request.train_steps) == expected.train_steps
        and request.grid == expected.grid
    )


def lightweight_request():
    """Two trials, one pattern of length 2, one Adam step, no bit flips.

    ``capacity_ratio`` stays 0.3, so ``m = max(1, int(0.3 * 2))`` is 1.
    Corruption 0 still draws a Bernoulli mask and flips nothing.
    """
    return NoiseSweepRequest(
        n_bits_list=(2,),
        corruption_rates=(0.0,),
        seeds=(0, 1),
        capacity_ratio=0.3,
        train_steps=1,
        grid="smoke",
    )


def n_patterns_for(n_bits, capacity_ratio):
    """``max(1, int(capacity_ratio * n_bits))``, the noise-sweep rule."""
    return max(1, int(capacity_ratio * n_bits))


def run_noise_sweep(
    request: NoiseSweepRequest,
    *,
    output_dir=None,
    overwrite: bool = False,
    experiment_id: str | None = None,
    configuration: dict | None = None,
    allow_historical_grid: bool = False,
):
    """Run the paired smoke and return the result mapping.

    ``allow_historical_grid`` starts the module grid of 600 cells at 300
    Adam steps. It does not call ``main`` and it does not plot. The default
    remains the smoke guard.

    ``output_dir=None`` keeps the result in memory. Otherwise a new
    ``output_dir/<run_id>`` directory is created after the archive check,
    unless ``overwrite`` is true, in which case ``output_dir`` itself is the
    run directory. ``overwrite`` still cannot select the archived results
    tree. Training starts after that check. A failure leaves
    ``execution_status`` as ``failed`` and is re-raised. The manifest
    records ``resumable`` false. This function does not resume a run.
    A process kill after the manifest is written can leave status
    ``running`` because the failure handler does not get to run.
    """
    configure_logging()
    try:
        if allow_historical_grid:
            if not is_historical_noise_sweep_request(request):
                raise HistoricalGridRefused(
                    "allow_historical_grid starts only the noise-sweep module grid: "
                    f"n_bits {list(historical.n_bits_list)}, "
                    f"corruption {list(historical.corruption_rates)}, "
                    f"seeds 0..{int(historical.trials) - 1}, "
                    f"train_steps {int(historical.train_steps)}, "
                    f"capacity_ratio {historical.capacity_ratio}, grid 'full'."
                )
        else:
            _refuse_unless_smoke(request)
    except Exception as exc:
        log_failure("request_refused", exc, protocol=PROTOCOL)
        raise
    identity = experiment_id or PROTOCOL
    n_cells = len(request.n_bits_list) * len(request.corruption_rates) * len(request.seeds)
    if output_dir is None:
        log_event(
            "run_start",
            experiment_id=identity,
            cells=n_cells,
            train_steps=request.train_steps,
            output="in-memory",
            historical_grid=allow_historical_grid,
            resumable=False,
        )
        try:
            payload = _execute(request)
        except BaseException as exc:
            log_failure("run_failed", exc, experiment_id=identity, output="in-memory")
            raise
        log_event(
            "run_complete",
            experiment_id=identity,
            status="in-memory",
            resumable=False,
        )
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
        train_steps=request.train_steps,
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
            fields=_completion_fields(payload),
        )
    except BaseException as exc:
        log_failure("run_failed", exc, run_id=run_id, experiment_id=identity)
        _record_failure(run_dir, manifest, exc)
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


def _record_failure(run_dir, manifest, exc: BaseException) -> None:
    """Write ``failed`` when the manifest can still be replaced. Re-raises nothing.

    The caller re-raises ``exc``. A write error is logged and does not replace
    ``exc``.
    """
    try:
        fail_run(run_dir, manifest, exc)
    except Exception as write_error:
        log_failure(
            "manifest_update_failed",
            write_error,
            run_id=manifest.get("run_id"),
        )


def _request_configuration(request: NoiseSweepRequest) -> dict:
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "n_bits_list": list(request.n_bits_list),
        "corruption_rates": list(request.corruption_rates),
        "trial_seeds": list(request.seeds),
        "capacity_ratio": request.capacity_ratio,
        "train_steps": request.train_steps,
    }


def _dataset_settings(request: NoiseSweepRequest) -> dict:
    return {
        "n_bits_list": list(request.n_bits_list),
        "capacity_ratio": request.capacity_ratio,
        "pattern_generator": "numpy.random",
        "pattern_rule": "max(1, int(capacity_ratio * n_bits))",
    }


def _perturbation_settings(request: NoiseSweepRequest) -> dict:
    return {
        "procedure_id": BERNOULLI,
        "levels": list(request.corruption_rates),
    }


def _manifest_fields(request: NoiseSweepRequest) -> dict:
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "main_invoked": False,
        "procedure_ids": list(PROCEDURE_IDS),
        "historical_module": "experiments.noise_robustness_sweep",
        "historical_train_steps": int(historical.train_steps),
        "historical_grid": request.grid == "full",
        "plotting": False,
        "rng_limitations": list(RNG_LIMITATIONS),
        "rng": rng_metadata(request),
    }


def rng_metadata(request: NoiseSweepRequest) -> dict:
    """Describe the seeds this smoke applies. This is not a reproducibility certificate.

    The values name the calls the runner and the Rot full-batch constructor
    already make. Nothing here seeds a generator or replaces the global stream.
    """
    trial_seeds = [int(seed) for seed in request.seeds]
    reset = qam_rot_fullbatch.rng_contract.reseed_value
    return {
        "scope": PROTOCOL,
        "exact_reproducibility_claimed": False,
        "family": "legacy_global",
        "default_rng_used": False,
        "configured_seeds": {
            "role": "configured",
            "trial_seeds": trial_seeds,
            "source": "request.seeds",
        },
        "seeds_applied": [
            {
                "role": "applied",
                "when": "start of each trial, before the pattern draw",
                "calls": ["numpy.random.seed", "pennylane.numpy.random.seed"],
                "values": trial_seeds,
            }
        ],
        "internal_resets": [
            {
                "role": "internal_reset",
                "procedure_id": QAM_ROT_FULLBATCH_ID,
                "when": (
                    "experiments.noise_robustness_sweep.QAM.__init__, "
                    "after the Hopfield queries"
                ),
                "call": "pennylane.numpy.random.seed",
                "value": reset,
                "configured": False,
                "followed_by": "numpy.random.randn for the initial parameters",
            }
        ],
        "unseeded_procedures": [
            {
                "role": "unseeded",
                "name": "gaussian_ci_population",
                "detail": (
                    "mean_std_ci is a formula on the trial scores. "
                    "This protocol does not draw a bootstrap sample."
                ),
            }
        ],
        "state_not_captured": [
            "The legacy RandomState is not serialized.",
            "Draws that follow the reset to 42 are not given a separate seed record.",
            "The run directory id uses uuid4 and is not a scientific seed.",
            "A recorded trial seed does not make the two arms share one query sequence.",
        ],
    }


_RESULT_FILES = (
    "manifest.json",
    "configuration.json",
    "trials.csv",
    "aggregates.csv",
    "trials.json",
)


def _refuse_unless_smoke(request: NoiseSweepRequest) -> None:
    if request.grid != "smoke":
        raise HistoricalGridRefused(
            "only grid='smoke' is implemented; the historical noise sweep is not run"
        )
    if request.train_steps < 1 or request.train_steps > MAX_SMOKE_TRAIN_STEPS:
        raise HistoricalGridRefused(
            f"smoke train_steps must be 1..{MAX_SMOKE_TRAIN_STEPS}; "
            f"the historical setting is {historical.train_steps}"
        )
    if request.capacity_ratio != historical.capacity_ratio:
        raise ValueError(
            f"capacity_ratio must stay {historical.capacity_ratio} "
            "so the pattern count uses the noise-sweep formula"
        )
    if not request.n_bits_list or not request.corruption_rates or not request.seeds:
        raise ValueError("n_bits_list, corruption_rates, and seeds must be non-empty")
    for n_bits in request.n_bits_list:
        if int(n_bits) < 1:
            raise ValueError("n_bits must be at least 1")
    for corruption in request.corruption_rates:
        if not 0.0 <= float(corruption) <= 1.0:
            raise ValueError("corruption rate must be between 0 and 1")
    n_cells = (
        len(request.n_bits_list) * len(request.corruption_rates) * len(request.seeds)
    )
    historical_cells = (
        len(historical.n_bits_list)
        * len(historical.corruption_rates)
        * int(historical.trials)
    )
    if n_cells > MAX_SMOKE_CELLS or n_cells == historical_cells:
        raise HistoricalGridRefused(
            f"{n_cells} cells were requested; this runner accepts at most "
            f"{MAX_SMOKE_CELLS}. The historical grid has {historical_cells} cells "
            f"and {historical.train_steps} Adam steps, and it is not run here."
        )


def _execute(request: NoiseSweepRequest):
    score_batch = selected(PROTOCOL, "batch_recall")
    interval = selected(PROTOCOL, "gaussian_ci_population")
    specs = list(request.cells())
    total = len(specs)
    log_event("execution_start", cells=total, train_steps=request.train_steps)
    cells = []
    for index, (n_bits, corruption, seed) in enumerate(specs, start=1):
        log_event(
            "trial_start",
            progress=f"{index}/{total}",
            n_bits=n_bits,
            corruption=corruption,
            seed=seed,
        )
        cell = _run_cell(
            n_bits,
            corruption,
            seed,
            capacity_ratio=request.capacity_ratio,
            train_steps=request.train_steps,
            score_batch=score_batch,
        )
        cells.append(cell)
        log_event("trial_complete", progress=f"{index}/{total}", seed=seed)
    log_event("evaluation_aggregate", cells=total)
    aggregates = _aggregate(cells, interval, request)
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "comparable_to_archive": False,
        "main_invoked": False,
        "procedure_ids": list(PROCEDURE_IDS),
        "historical_module": "experiments.noise_robustness_sweep",
        "historical_train_steps": int(historical.train_steps),
        "historical_n_bits_list": list(historical.n_bits_list),
        "historical_corruption_rates": list(historical.corruption_rates),
        "historical_trials": int(historical.trials),
        "rng_limitations": list(RNG_LIMITATIONS),
        "cells": cells,
        "aggregates": aggregates,
    }


def _run_cell(n_bits, corruption, seed, *, capacity_ratio, train_steps, score_batch):
    np.random.seed(seed)
    pnp.random.seed(seed)
    n_patterns = n_patterns_for(n_bits, capacity_ratio)
    batch = draw_bipolar_patterns(
        n_patterns,
        n_bits,
        generator=GENERATOR_NUMPY_GLOBAL,
        seed_applied=seed,
    )
    patterns = batch.patterns
    spec = CorruptionSpec(BERNOULLI, corruption)

    hopfield_adapter = get(HOPFIELD_ID).adapter
    hopfield_factory, hopfield_args, hopfield_kwargs = prepare(
        HOPFIELD_ID, {"n_bits": n_bits, "steps": 10}
    )
    hopfield = hopfield_factory(*hopfield_args, **hopfield_kwargs)
    log_event("model_construct", procedure_id=HOPFIELD_ID, n_bits=n_bits, seed=seed)
    hopfield_train_s = float(hopfield_adapter.train(hopfield, patterns))
    log_event("model_train", procedure_id=HOPFIELD_ID, seed=seed)
    hopfield_queries = [perturb(row, spec) for row in patterns]
    hopfield_recalled = []
    hopfield_times = []
    for query in hopfield_queries:
        recalled, per_bit_s = hopfield_adapter.recall_timed(
            hopfield, query.corrupted
        )
        hopfield_recalled.append(np.asarray(recalled))
        hopfield_times.append(float(per_bit_s))
    hopfield_full, hopfield_bit = score_batch(hopfield_recalled, patterns)
    log_event(
        "model_evaluate",
        procedure_id=HOPFIELD_ID,
        seed=seed,
        queries=len(hopfield_queries),
    )

    qam_adapter = get(QAM_ROT_FULLBATCH_ID).adapter
    qam_factory, qam_args, qam_kwargs = prepare(
        QAM_ROT_FULLBATCH_ID,
        {
            "n_bits": n_bits,
            "train_steps": train_steps,
            "depth_argument": 6,
            "lr": 0.05,
        },
    )
    model = qam_factory(*qam_args, **qam_kwargs)
    log_event(
        "model_construct",
        procedure_id=QAM_ROT_FULLBATCH_ID,
        n_bits=n_bits,
        seed=seed,
    )
    init_params = np.array(np.asarray(model.params), dtype=float, copy=True)
    qam_train_s = float(qam_adapter.train(model, patterns))
    log_event("model_train", procedure_id=QAM_ROT_FULLBATCH_ID, seed=seed)
    qam_queries = [perturb(row, spec) for row in patterns]
    qam_recalled = []
    qam_times = []
    for query in qam_queries:
        recalled, per_bit_s = qam_adapter.recall(model, query.corrupted)
        qam_recalled.append(np.asarray(recalled))
        qam_times.append(float(per_bit_s))
    qam_full, qam_bit = score_batch(qam_recalled, patterns)
    log_event(
        "model_evaluate",
        procedure_id=QAM_ROT_FULLBATCH_ID,
        seed=seed,
        queries=len(qam_queries),
    )

    shared = {
        "protocol": PROTOCOL,
        "grid": "smoke",
        "comparable_to_archive": False,
        "trial_seed": int(seed),
        "n_bits": int(n_bits),
        "n_patterns": int(n_patterns),
        "capacity_ratio": float(capacity_ratio),
        "pattern_generator": batch.generator,
        "patterns": _matrix(patterns),
        "perturbation_id": BERNOULLI,
        "perturbation_level": float(corruption),
    }
    arms = [
        _arm(
            shared,
            procedure_id=HOPFIELD_ID,
            source=hopfield_synchronous.identity.source,
            full_accuracy=hopfield_full,
            bit_accuracy=hopfield_bit,
            train_time_seconds=hopfield_train_s,
            mean_recall_time_per_bit=float(np.mean(hopfield_times)),
            recalled=_matrix(hopfield_recalled),
            queries=hopfield_queries,
            constructor_reseed=None,
            train_steps=None,
            init_params=None,
            depth_used=None,
            lr=None,
        ),
        _arm(
            shared,
            procedure_id=QAM_ROT_FULLBATCH_ID,
            source=qam_rot_fullbatch.identity.source,
            full_accuracy=qam_full,
            bit_accuracy=qam_bit,
            train_time_seconds=qam_train_s,
            mean_recall_time_per_bit=float(np.mean(qam_times)),
            recalled=_matrix(qam_recalled),
            queries=qam_queries,
            constructor_reseed=qam_rot_fullbatch.rng_contract.reseed_value,
            train_steps=int(model.train_steps),
            init_params=init_params.tolist(),
            depth_used=int(model.depth),
            lr=float(model.lr),
        ),
    ]
    return {**shared, "arms": arms}


def _arm(
    shared,
    *,
    procedure_id,
    source,
    full_accuracy,
    bit_accuracy,
    train_time_seconds,
    mean_recall_time_per_bit,
    recalled,
    queries,
    constructor_reseed,
    train_steps,
    init_params,
    depth_used,
    lr,
):
    return {
        "procedure_id": procedure_id,
        "source": source,
        "metrics": {
            "full_accuracy": float(full_accuracy),
            "bit_accuracy": float(bit_accuracy),
        },
        "timings": {
            "train_time_seconds": float(train_time_seconds),
            "mean_recall_time_per_bit": float(mean_recall_time_per_bit),
        },
        "recalled_patterns": recalled,
        "queries": [
            {
                "corrupted": _matrix([query.corrupted])[0],
                "n_flips": int(query.n_flips),
                "perturbation_id": query.procedure_id,
                "perturbation_level": float(query.level),
            }
            for query in queries
        ],
        "constructor_reseed": constructor_reseed,
        "train_steps": train_steps,
        "init_params": init_params,
        "depth_used": depth_used,
        "lr": lr,
        "trial_seed": shared["trial_seed"],
        "n_bits": shared["n_bits"],
        "n_patterns": shared["n_patterns"],
    }


def _aggregate(cells, interval, request):
    grouped = {}
    for cell in cells:
        for arm in cell["arms"]:
            key = (cell["n_bits"], cell["perturbation_level"], arm["procedure_id"])
            grouped.setdefault(key, []).append((cell, arm))
    rows = []
    for (n_bits, level, procedure_id), pairs in grouped.items():
        full_values = [arm["metrics"]["full_accuracy"] for _, arm in pairs]
        bit_values = [arm["metrics"]["bit_accuracy"] for _, arm in pairs]
        full_mean, full_std, full_ci = interval(full_values)
        bit_mean, bit_std, bit_ci = interval(bit_values)
        train_times = [arm["timings"]["train_time_seconds"] for _, arm in pairs]
        recall_times = [arm["timings"]["mean_recall_time_per_bit"] for _, arm in pairs]
        cell, arm = pairs[0]
        rows.append(
            {
                "protocol": PROTOCOL,
                "grid": request.grid,
                "comparable_to_archive": False,
                "procedure_id": procedure_id,
                "n_bits": n_bits,
                "n_patterns": cell["n_patterns"],
                "perturbation_id": BERNOULLI,
                "perturbation_level": level,
                "n_trials": len(pairs),
                "full_accuracy_mean": float(full_mean),
                "full_accuracy_std": float(full_std),
                "full_accuracy_ci": float(full_ci),
                "bit_accuracy_mean": float(bit_mean),
                "bit_accuracy_std": float(bit_std),
                "bit_accuracy_ci": float(bit_ci),
                "train_time_seconds_mean": float(np.mean(train_times)),
                "mean_recall_time_per_bit_mean": float(np.mean(recall_times)),
                "source": arm["source"],
            }
        )
    return rows


def _completion_fields(payload) -> dict:
    qam_arm = next(
        arm
        for arm in payload["cells"][0]["arms"]
        if arm["procedure_id"] == QAM_ROT_FULLBATCH_ID
    )
    return {"smoke_train_steps": qam_arm["train_steps"]}


def _write_tables(run_dir, payload) -> None:
    write_csv_table(run_dir, "trials.csv", _trial_rows(payload), TRIAL_COLUMNS)
    write_csv_table(run_dir, "aggregates.csv", payload["aggregates"], AGGREGATE_COLUMNS)
    write_json(run_dir, "trials.json", {"cells": payload["cells"]})


def _trial_rows(payload):
    rows = []
    for cell in payload["cells"]:
        for arm in cell["arms"]:
            rows.append(
                {
                    "protocol": payload["protocol"],
                    "grid": payload["grid"],
                    "comparable_to_archive": False,
                    "procedure_id": arm["procedure_id"],
                    "source": arm["source"],
                    "trial_seed": cell["trial_seed"],
                    "n_bits": cell["n_bits"],
                    "n_patterns": cell["n_patterns"],
                    "capacity_ratio": cell["capacity_ratio"],
                    "pattern_generator": cell["pattern_generator"],
                    "perturbation_id": cell["perturbation_id"],
                    "perturbation_level": cell["perturbation_level"],
                    "full_accuracy": arm["metrics"]["full_accuracy"],
                    "bit_accuracy": arm["metrics"]["bit_accuracy"],
                    "train_time_seconds": arm["timings"]["train_time_seconds"],
                    "mean_recall_time_per_bit": arm["timings"]["mean_recall_time_per_bit"],
                    "constructor_reseed": (
                        "" if arm["constructor_reseed"] is None else arm["constructor_reseed"]
                    ),
                    "train_steps": "" if arm["train_steps"] is None else arm["train_steps"],
                }
            )
    return rows


def _matrix(rows):
    return [[int(value) for value in np.asarray(row).tolist()] for row in rows]
