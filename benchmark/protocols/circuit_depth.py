"""Circuit-depth protocol.

Each cell is one ``(n_bits, m, depth)``. ``m`` comes from the module's
representative cases, not from a capacity formula. ``depth`` is the layer
count passed to ``train_ry_linear_qam``. It is not ``min(8, n_bits)``.

Every trial reseeds NumPy, draws patterns, and trains with
``return_tuple=False``. Recall is the three-line sign map in
``experiments.circuit_depth.main``: PennyLane ``array``, ``sign``, and
zeros mapped to ``+1``. This module does not call ``qam_recall``.

The reported numbers are the mean full accuracy, the mean bit accuracy,
and the mean train time over the five seeds. There is no bootstrap.
This module does not call ``main`` and it does not plot.

The default request is one historical cell. The 21-cell module grid starts
only when ``allow_historical_grid`` is true.
"""

from __future__ import annotations

import numpy as np
from pennylane import numpy as pnp

import experiments.circuit_depth as historical
from benchmark.adapters import qam_ry_linear
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
from benchmark.scoring import selected

PROTOCOL = "circuit_depth"
QAM_ID = qam_ry_linear.identity.procedure_id
PROCEDURE_IDS = (QAM_ID,)
PATTERN_GENERATOR = "numpy.random"
TRAINING_SCHEDULE = "reseed_each_trial_then_train_then_fixed_count"
RECALL_IMPLEMENTATION = "inline_sign_of_circuit_expectations"
NOT_APPLIED_DEPTH_RULE = "min(8, n_bits)"
FRAMEWORK_PROCESS = "single_process"
HISTORICAL_PROCESS = "single_process"

MAX_SMOKE_CASES = 1
MAX_SMOKE_DEPTHS = 1

ROW_COLUMNS = (
    "protocol",
    "grid",
    "comparable_to_archive",
    "procedure_id",
    "return_tuple",
    "n_bits",
    "m",
    "depth",
    "full_acc_mean",
    "bit_acc_mean",
    "train_time_mean",
    "params",
    "sample_size",
)

_RESULT_FILES = (
    "manifest.json",
    "configuration.json",
    "rows.csv",
    "rows.json",
    "trials.json",
)


class HistoricalGridRefused(RuntimeError):
    """The 21-cell circuit-depth grid is not executed here."""


class CircuitDepthRequest:
    """Cases, depths, and the trial budget. Building it does not train."""

    def __init__(
        self,
        cases,
        depths,
        seeds,
        train_steps,
        lr,
        recall_noise,
        return_tuple=False,
        grid="smoke",
    ):
        self.cases = tuple((int(n_bits), int(m)) for n_bits, m in cases)
        self.depths = tuple(int(depth) for depth in depths)
        self.seeds = tuple(int(seed) for seed in seeds)
        self.train_steps = int(train_steps)
        self.lr = float(lr)
        self.recall_noise = float(recall_noise)
        self.return_tuple = bool(return_tuple)
        self.grid = grid

    def cells(self):
        """``(n_bits, m, depth)`` in module order."""
        for n_bits, m in self.cases:
            for depth in self.depths:
                yield int(n_bits), int(m), int(depth)


def historical_cases():
    """``(n, representative_cases[n])`` in ``n_bits_list`` order."""
    return tuple(
        (int(n_bits), int(historical.representative_cases[n_bits]))
        for n_bits in historical.n_bits_list
    )


def historical_depths():
    return tuple(int(depth) for depth in historical.depths)


def parameter_count(depth, n_bits):
    """``depth * n_bits``, the column ``main`` writes."""
    return int(depth) * int(n_bits)


def sign_circuit_output(circuit, params, query):
    """Sign the depth-script circuit output. Zeros become ``+1``.

    These are the three lines in ``experiments.circuit_depth.main``.
    ``qam_recall`` is a different call site, on the storage-capacity path.
    """
    out = pnp.array(circuit(params, query))
    out = pnp.sign(out)
    out[out == 0] = 1
    return out


def historical_request():
    """The module grid. Building it does not train and does not call ``main``."""
    return CircuitDepthRequest(
        cases=historical_cases(),
        depths=historical_depths(),
        seeds=tuple(range(int(historical.trials))),
        train_steps=int(historical.train_steps),
        lr=float(historical.lr),
        recall_noise=float(historical.recall_noise),
        return_tuple=False,
        grid="full",
    )


def historical_cell_count():
    """Cells in the module grid. This does not run them."""
    return len(tuple(historical_request().cells()))


def is_historical_circuit_depth_request(request: CircuitDepthRequest) -> bool:
    expected = historical_request()
    return (
        request.cases == expected.cases
        and request.depths == expected.depths
        and request.seeds == expected.seeds
        and int(request.train_steps) == expected.train_steps
        and float(request.lr) == expected.lr
        and float(request.recall_noise) == expected.recall_noise
        and request.return_tuple is False
        and request.grid == "full"
    )


def lightweight_request():
    """The first module cell: ``n=8``, ``m=4``, depth ``1``, seeds ``0..4``.

    Depth ``1`` is the first value in the module list. The other six depths
    are not scored. Step count, learning rate, noise, trial count, and
    ``return_tuple=False`` stay the module settings.
    """
    n_bits, m = historical_cases()[0]
    return CircuitDepthRequest(
        cases=((n_bits, m),),
        depths=(historical_depths()[0],),
        seeds=tuple(range(int(historical.trials))),
        train_steps=int(historical.train_steps),
        lr=float(historical.lr),
        recall_noise=float(historical.recall_noise),
        return_tuple=False,
        grid="smoke",
    )


def run_circuit_depth(
    request: CircuitDepthRequest,
    *,
    output_dir=None,
    overwrite: bool = False,
    experiment_id: str | None = None,
    configuration: dict | None = None,
    allow_historical_grid: bool = False,
):
    """Run the depth schedule and return the result mapping.

    ``allow_historical_grid`` starts the module grid. The default stays on
    the one-cell guard. A failure leaves ``execution_status`` ``failed``
    and is re-raised. The run is not resumed.
    """
    configure_logging()
    try:
        if allow_historical_grid:
            if not is_historical_circuit_depth_request(request):
                raise HistoricalGridRefused(
                    "allow_historical_grid starts only the circuit-depth module grid."
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


def rng_metadata(request: CircuitDepthRequest) -> dict:
    """Name the seeds this request applies. This is not a reproducibility certificate."""
    applied = [
        {
            "n_bits": n_bits,
            "m": m,
            "depth": depth,
            "value": seed,
            "call": "numpy.random.seed",
        }
        for n_bits, m, depth in request.cells()
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
            "formula": "the trial index in range(trials)",
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
        "unseeded_procedures": [],
        "state_not_captured": [
            "The legacy RandomState is not serialized.",
            "RY-linear init draws pennylane.numpy.random.randn after the pattern choice.",
            "Fixed-count indexes are drawn after training, on numpy.random.",
            "The recorded means do not include a bootstrap.",
            "Train-time means are wall-clock measurements.",
            "A recorded seed list does not by itself reproduce the archived depth table.",
        ],
    }


def _manifest_fields(request: CircuitDepthRequest) -> dict:
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "main_invoked": False,
        "plotting": False,
        "procedure_ids": list(PROCEDURE_IDS),
        "historical_module": "experiments.circuit_depth",
        "historical_qam": "src.qam.train_ry_linear_qam",
        "return_tuple": False,
        "recall_implementation": RECALL_IMPLEMENTATION,
        "depth_rule_not_applied": NOT_APPLIED_DEPTH_RULE,
        "historical_train_steps": int(historical.train_steps),
        "historical_depths": list(historical_depths()),
        "historical_cells": historical_cell_count(),
        "historical_trials_per_cell": int(historical.trials),
        "historical_grid": request.grid == "full",
        "training_schedule": TRAINING_SCHEDULE,
        "aggregation": "mean_over_trials",
        "bootstrap_used": False,
        "process_model": FRAMEWORK_PROCESS,
        "historical_process_model": HISTORICAL_PROCESS,
        "rng": rng_metadata(request),
    }


def _request_configuration(request: CircuitDepthRequest) -> dict:
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "cases": [{"n_bits": n_bits, "m": m} for n_bits, m in request.cases],
        "depths": list(request.depths),
        "seeds": list(request.seeds),
        "train_steps": int(request.train_steps),
        "lr": float(request.lr),
        "recall_noise": float(request.recall_noise),
        "return_tuple": False,
    }


def _dataset_settings(request: CircuitDepthRequest) -> dict:
    return {
        "cases": [{"n_bits": n_bits, "m": m} for n_bits, m in request.cases],
        "depths": list(request.depths),
        "pattern_generator": PATTERN_GENERATOR,
    }


def _perturbation_settings(request: CircuitDepthRequest) -> dict:
    return {
        "procedure_id": "fixed_count",
        "implementation": "src.data_generation.corrupt_fixed_count",
        "level": float(request.recall_noise),
        "rng": "numpy.random",
        "when": "each_stored_pattern_after_training",
    }


def _refuse_unless_smoke(request: CircuitDepthRequest) -> None:
    if request.grid != "smoke":
        raise HistoricalGridRefused(
            "only grid='smoke' is implemented unless allow_historical_grid is set"
        )
    if request.return_tuple is not False:
        raise HistoricalGridRefused(
            "return_tuple stays false. True builds the storage-capacity QNode."
        )
    if int(request.train_steps) != int(historical.train_steps):
        raise HistoricalGridRefused(
            f"train_steps stays {int(historical.train_steps)}."
        )
    if request.seeds != tuple(range(int(historical.trials))):
        raise HistoricalGridRefused(
            f"seeds stay {tuple(range(int(historical.trials)))}."
        )
    if float(request.lr) != float(historical.lr):
        raise HistoricalGridRefused(f"lr stays {float(historical.lr)}.")
    if float(request.recall_noise) != float(historical.recall_noise):
        raise HistoricalGridRefused(f"recall_noise stays {float(historical.recall_noise)}.")
    if len(request.cases) > MAX_SMOKE_CASES:
        raise HistoricalGridRefused(f"smoke cases must be at most {MAX_SMOKE_CASES}")
    if len(request.depths) > MAX_SMOKE_DEPTHS:
        raise HistoricalGridRefused(f"smoke depths must be at most {MAX_SMOKE_DEPTHS}")
    if not request.cases or not request.depths:
        raise ValueError("cases and depths must be non-empty")
    if request.cases[0] not in historical_cases():
        raise HistoricalGridRefused("smoke case must be one of the module (n, m) pairs")
    if request.depths[0] not in historical_depths():
        raise HistoricalGridRefused("smoke depth must be one of the module depths")
    if len(tuple(request.cells())) == historical_cell_count():
        raise HistoricalGridRefused("the historical circuit-depth grid is not started here")


def _execute(request: CircuitDepthRequest):
    specs = list(request.cells())
    total = len(specs)
    log_event(
        "execution_start",
        cells=total,
        trials_per_cell=len(request.seeds),
        train_steps=int(request.train_steps),
        return_tuple=False,
    )
    rows = []
    trials = []
    for index, (n_bits, m, depth) in enumerate(specs, start=1):
        log_event(
            "cell_start",
            progress=f"{index}/{total}",
            n_bits=n_bits,
            m=m,
            depth=depth,
        )
        row, cell_trials = _run_cell(request, n_bits, m, depth)
        rows.append(row)
        trials.extend(cell_trials)
        log_event(
            "cell_complete",
            progress=f"{index}/{total}",
            n_bits=n_bits,
            m=m,
            depth=depth,
        )
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "comparable_to_archive": False,
        "main_invoked": False,
        "procedure_ids": list(PROCEDURE_IDS),
        "rows": rows,
        "trials": trials,
    }


def _run_cell(request: CircuitDepthRequest, n_bits, m, depth):
    score = selected(PROTOCOL, "batch_recall")
    adapter = get(QAM_ID).adapter
    full_scores = []
    bit_scores = []
    train_times = []
    trials = []
    for seed_index, seed in enumerate(request.seeds, start=1):
        log_event(
            "trial_start",
            progress=f"{seed_index}/{len(request.seeds)}",
            n_bits=n_bits,
            m=m,
            depth=depth,
            seed=seed,
        )
        np.random.seed(seed)
        patterns = np.random.choice([-1, 1], size=(m, n_bits))
        prepare(
            QAM_ID,
            {"n_bits": n_bits, "depth": depth, "return_tuple": False},
        )
        log_event(
            "model_construct",
            procedure_id=QAM_ID,
            n_bits=n_bits,
            m=m,
            depth=depth,
            seed=seed,
            return_tuple=False,
        )
        params, circuit, train_time = adapter.train(
            patterns,
            n_bits,
            depth,
            steps=int(request.train_steps),
            lr=float(request.lr),
            return_tuple=False,
        )
        train_times.append(float(train_time))
        log_event(
            "model_train",
            procedure_id=QAM_ID,
            n_bits=n_bits,
            m=m,
            depth=depth,
            seed=seed,
            steps=int(request.train_steps),
        )
        recalled = []
        for pattern in patterns:
            query = apply_fixed_count(pattern, request.recall_noise, rng=np.random)
            recalled.append(sign_circuit_output(circuit, params, query))
        full, bit = score(recalled, patterns)
        full_scores.append(float(full))
        bit_scores.append(float(bit))
        log_event(
            "model_evaluate",
            procedure_id=QAM_ID,
            n_bits=n_bits,
            m=m,
            depth=depth,
            seed=seed,
            queries=m,
        )
        trials.append(
            {
                "n_bits": int(n_bits),
                "m": int(m),
                "depth": int(depth),
                "seed": int(seed),
                "return_tuple": False,
                "procedure_id": QAM_ID,
                "train_steps": int(request.train_steps),
                "patterns": np.asarray(patterns).astype(int).tolist(),
                "full_acc": float(full),
                "bit_acc": float(bit),
                "params": np.asarray(params, dtype=float).tolist(),
                "parameter_count": parameter_count(depth, n_bits),
            }
        )
        log_event(
            "trial_complete",
            progress=f"{seed_index}/{len(request.seeds)}",
            n_bits=n_bits,
            m=m,
            depth=depth,
            seed=seed,
        )
    log_event(
        "aggregate_start",
        n_bits=n_bits,
        m=m,
        depth=depth,
        aggregation="mean_over_trials",
        bootstrap_used=False,
        sample_size=len(request.seeds),
    )
    row = {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "comparable_to_archive": False,
        "procedure_id": QAM_ID,
        "return_tuple": False,
        "n_bits": int(n_bits),
        "m": int(m),
        "depth": int(depth),
        "full_acc_mean": float(np.mean(np.asarray(full_scores, dtype=float))),
        "bit_acc_mean": float(np.mean(np.asarray(bit_scores, dtype=float))),
        "train_time_mean": float(np.mean(np.asarray(train_times, dtype=float))),
        "params": parameter_count(depth, n_bits),
        "sample_size": len(request.seeds),
    }
    log_event("aggregate_complete", n_bits=n_bits, m=m, depth=depth, resumable=False)
    return row, trials


def _write_tables(run_dir, payload):
    write_csv_table(run_dir, "rows.csv", payload["rows"], ROW_COLUMNS)
    write_json(run_dir, "rows.json", payload["rows"])
    write_json(run_dir, "trials.json", payload["trials"])
