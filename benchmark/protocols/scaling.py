"""Scaling protocol: one training per trial, then every noise level.

The models are ``HopfieldAsyncRandom`` and ``experiments.scaling.QAM``.
Pattern draws, the constructor ``randn``, minibatch draws, Bernoulli masks,
and recall permutations stay on the PennyLane NumPy legacy stream. This
module does not call ``experiments.scaling.main`` and it does not plot.

The default request is a one-width, one-trial, one-step smoke. The module
grid starts only when ``allow_historical_grid`` is true.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from pennylane import numpy as pnp

import experiments.scaling as historical
from benchmark.adapters import hopfield_async_random, qam_rot_minibatch
from benchmark.data.corruption import BERNOULLI, CorruptionSpec, perturb
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
from benchmark.scoring import batch_recall

PROTOCOL = "scaling"
HOPFIELD_ID = hopfield_async_random.identity.procedure_id
QAM_ID = qam_rot_minibatch.identity.procedure_id
PROCEDURE_IDS = (HOPFIELD_ID, QAM_ID)
PATTERN_RULE = "min_max_one_or_half_n_bits_capped_at_6"
DEPTH_RULE = "min_6_and_n_bits"
SEED_FORMULA = "123 + trial_id + n_bits * 100"
SEED_POLICY = "pennylane_numpy_seed_base_plus_trial_plus_n_times_100"
TRAINING_SCHEDULE = "train_once_per_trial_then_score_each_noise"
GENERATOR = "pennylane.numpy.random"

MAX_SMOKE_WIDTHS = 1
MAX_SMOKE_TRIALS = 1
MAX_SMOKE_NOISE_LEVELS = 1
MAX_SMOKE_STEPS = 1

TRIAL_COLUMNS = (
    "protocol",
    "grid",
    "comparable_to_archive",
    "procedure_id",
    "historical_model_label",
    "n_bits",
    "m",
    "trial_id",
    "seed",
    "noise_p",
    "full_acc",
    "bit_acc",
    "train_time_s",
    "infer_time_s",
    "params_count",
    "extra_notes",
)

_RESULT_FILES = (
    "manifest.json",
    "configuration.json",
    "trials.csv",
    "trials.json",
)


class HistoricalGridRefused(RuntimeError):
    """The historical scaling grid is not executed by this runner."""


@dataclass(frozen=True)
class ScalingRequest:
    """Widths, noise levels, trial ids, and the QAM step budget."""

    n_bits_list: tuple[int, ...]
    noise_levels: tuple[float, ...]
    trial_ids: tuple[int, ...]
    steps: int
    lr: float = 0.05
    seed_base: int = 123
    grid: str = "smoke"

    def trainings(self):
        """Pairs that train. Noise levels are applied after training."""
        for n_bits in self.n_bits_list:
            for trial_id in self.trial_ids:
                yield int(n_bits), int(trial_id)


def n_patterns_for(n_bits):
    """``min(max(1, n_bits // 2), 6)``, the scaling rule."""
    return min(max(1, n_bits // 2), 6)


def trial_seed(seed_base, trial_id, n_bits):
    """``seed_base + trial_id + n_bits * 100``."""
    return int(seed_base) + int(trial_id) + int(n_bits) * 100


def historical_scaling_request():
    """The module grid. Building it does not train and does not call ``main``."""
    return ScalingRequest(
        n_bits_list=tuple(int(n_bits) for n_bits in historical.n_bits_list),
        noise_levels=tuple(float(level) for level in historical.noise_levels),
        trial_ids=tuple(range(int(historical.trials))),
        steps=int(qam_rot_minibatch.PROTOCOL_STEPS),
        lr=float(qam_rot_minibatch.PROTOCOL_LR),
        seed_base=int(historical.RANDOM_SEED_BASE),
        grid="full",
    )


def is_historical_scaling_request(request: ScalingRequest) -> bool:
    expected = historical_scaling_request()
    return (
        tuple(int(n) for n in request.n_bits_list) == expected.n_bits_list
        and tuple(float(level) for level in request.noise_levels) == expected.noise_levels
        and tuple(int(trial) for trial in request.trial_ids) == expected.trial_ids
        and int(request.steps) == expected.steps
        and float(request.lr) == expected.lr
        and int(request.seed_base) == expected.seed_base
        and request.grid == "full"
    )


def lightweight_request():
    """One historical width, one trial, one noise level, one training step.

    ``n_bits`` 4 keeps ``m = 2``. The step count is the smoke budget, not 200.
    """
    return ScalingRequest(
        n_bits_list=(4,),
        noise_levels=(0.0,),
        trial_ids=(0,),
        steps=1,
        lr=0.05,
        seed_base=123,
        grid="smoke",
    )


def run_scaling(
    request: ScalingRequest,
    *,
    output_dir=None,
    overwrite: bool = False,
    experiment_id: str | None = None,
    configuration: dict | None = None,
    allow_historical_grid: bool = False,
):
    """Run the scaling schedule and return the result mapping.

    ``allow_historical_grid`` starts the module grid. The default stays on
    the smoke guard. A failure leaves ``execution_status`` ``failed`` and
    is re-raised. The run is not resumed.
    """
    configure_logging()
    try:
        if allow_historical_grid:
            if not is_historical_scaling_request(request):
                raise HistoricalGridRefused(
                    "allow_historical_grid starts only the scaling module grid."
                )
        else:
            _refuse_unless_smoke(request)
    except Exception as exc:
        log_failure("request_refused", exc, protocol=PROTOCOL)
        raise

    identity = experiment_id or PROTOCOL
    n_trainings = len(request.n_bits_list) * len(request.trial_ids)
    if output_dir is None:
        log_event(
            "run_start",
            experiment_id=identity,
            trainings=n_trainings,
            steps=request.steps,
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
        trainings=n_trainings,
        steps=request.steps,
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
            fields={"smoke_steps": None if allow_historical_grid else int(request.steps)},
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


def rng_metadata(request: ScalingRequest) -> dict:
    """Name the seeds this request applies. This is not a reproducibility certificate."""
    applied = [
        {
            "n_bits": n_bits,
            "trial_id": trial_id,
            "value": trial_seed(request.seed_base, trial_id, n_bits),
        }
        for n_bits, trial_id in request.trainings()
    ]
    return {
        "scope": PROTOCOL,
        "exact_reproducibility_claimed": False,
        "family": "legacy_global",
        "generator": GENERATOR,
        "default_rng_used": False,
        "configured_seeds": {
            "role": "configured",
            "seed_base": int(request.seed_base),
            "formula": SEED_FORMULA,
            "source": "request.seed_base and request.trainings",
        },
        "seeds_applied": [
            {
                "role": "applied",
                "when": "start of each trial, before the pattern draw",
                "call": "pennylane.numpy.random.seed",
                "values": applied,
            }
        ],
        "internal_resets": [],
        "unseeded_procedures": [
            {
                "role": "unseeded",
                "name": "experiments.scaling.bootstrap_ci",
                "detail": (
                    "The helper draws on the global stream and has no seed call. "
                    "The historical main does not call it, and this runner does not call it."
                ),
            }
        ],
        "state_not_captured": [
            "QAM construction draws randn on the trial stream and does not reseed.",
            "Each training step draws a minibatch, and the optimizer evaluates that draw a second time.",
            "Early stopping changes how many of those draws occur.",
            "Bernoulli masks and Hopfield permutations continue on the same stream.",
            "The legacy RandomState is not serialized.",
            "A recorded trial seed does not by itself reproduce the archived CSV.",
        ],
    }


def _manifest_fields(request: ScalingRequest) -> dict:
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "main_invoked": False,
        "plotting": False,
        "procedure_ids": list(PROCEDURE_IDS),
        "historical_module": "experiments.scaling",
        "historical_steps": int(qam_rot_minibatch.PROTOCOL_STEPS),
        "historical_grid": request.grid == "full",
        "training_schedule": TRAINING_SCHEDULE,
        "pattern_rule": PATTERN_RULE,
        "depth_rule": DEPTH_RULE,
        "rng": rng_metadata(request),
    }


def _request_configuration(request: ScalingRequest) -> dict:
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "n_bits_list": list(request.n_bits_list),
        "noise_levels": list(request.noise_levels),
        "trial_ids": list(request.trial_ids),
        "steps": int(request.steps),
        "lr": float(request.lr),
        "seed_base": int(request.seed_base),
    }


def _dataset_settings(request: ScalingRequest) -> dict:
    return {
        "n_bits_list": list(request.n_bits_list),
        "pattern_generator": GENERATOR,
        "pattern_rule": PATTERN_RULE,
    }


def _perturbation_settings(request: ScalingRequest) -> dict:
    return {
        "procedure_id": BERNOULLI,
        "implementation": "corrupt_bernoulli_on_the_legacy_stream",
        "levels": list(request.noise_levels),
        "when": "each_stored_pattern_after_both_models_train",
    }


def _refuse_unless_smoke(request: ScalingRequest) -> None:
    if request.grid != "smoke":
        raise HistoricalGridRefused(
            "only grid='smoke' is implemented unless allow_historical_grid is set"
        )
    if int(request.steps) < 1 or int(request.steps) > MAX_SMOKE_STEPS:
        raise HistoricalGridRefused(
            f"smoke steps must be 1..{MAX_SMOKE_STEPS}; "
            f"the historical setting is {qam_rot_minibatch.PROTOCOL_STEPS}"
        )
    if float(request.lr) != float(qam_rot_minibatch.PROTOCOL_LR):
        raise ValueError("lr must stay 0.05")
    if int(request.seed_base) != int(historical.RANDOM_SEED_BASE):
        raise ValueError("seed_base must stay 123")
    if not request.n_bits_list or not request.noise_levels or not request.trial_ids:
        raise ValueError("n_bits_list, noise_levels, and trial_ids must be non-empty")
    if len(request.n_bits_list) > MAX_SMOKE_WIDTHS:
        raise HistoricalGridRefused(
            f"smoke widths must be at most {MAX_SMOKE_WIDTHS}"
        )
    if len(request.trial_ids) > MAX_SMOKE_TRIALS:
        raise HistoricalGridRefused(
            f"smoke trials must be at most {MAX_SMOKE_TRIALS}"
        )
    if len(request.noise_levels) > MAX_SMOKE_NOISE_LEVELS:
        raise HistoricalGridRefused(
            f"smoke noise levels must be at most {MAX_SMOKE_NOISE_LEVELS}"
        )
    historical_trainings = len(historical.n_bits_list) * int(historical.trials)
    if len(request.n_bits_list) * len(request.trial_ids) == historical_trainings:
        raise HistoricalGridRefused("the historical scaling grid is not started here")


def _execute(request: ScalingRequest):
    specs = list(request.trainings())
    total = len(specs)
    log_event(
        "execution_start",
        trainings=total,
        steps=request.steps,
        noise_levels=len(request.noise_levels),
    )
    trials = []
    for index, (n_bits, trial_id) in enumerate(specs, start=1):
        seed = trial_seed(request.seed_base, trial_id, n_bits)
        log_event(
            "trial_start",
            progress=f"{index}/{total}",
            n_bits=n_bits,
            trial_id=trial_id,
            seed=seed,
        )
        trials.append(
            _run_trial(
                n_bits,
                trial_id,
                request.noise_levels,
                steps=request.steps,
                lr=request.lr,
                seed_base=request.seed_base,
                grid=request.grid,
            )
        )
        log_event("trial_complete", progress=f"{index}/{total}", seed=seed)
    return {
        "protocol": PROTOCOL,
        "grid": request.grid,
        "comparable_to_archive": False,
        "main_invoked": False,
        "procedure_ids": list(PROCEDURE_IDS),
        "trials": trials,
    }


def _run_trial(n_bits, trial_id, noise_levels, *, steps, lr, seed_base, grid):
    seed = trial_seed(seed_base, trial_id, n_bits)
    pnp.random.seed(seed)
    count = n_patterns_for(n_bits)
    patterns = pnp.random.choice([-1, 1], size=(count, n_bits))

    hopfield_adapter = get(HOPFIELD_ID).adapter
    hopfield_factory, hopfield_args, hopfield_kwargs = prepare(
        HOPFIELD_ID, {"n_bits": n_bits, "steps": 10}
    )
    hopfield = hopfield_factory(*hopfield_args, **hopfield_kwargs)
    log_event("model_construct", procedure_id=HOPFIELD_ID, n_bits=n_bits, seed=seed)
    started = time.time()
    hopfield_adapter.train(hopfield, patterns)
    hopfield_train_s = time.time() - started
    log_event("model_train", procedure_id=HOPFIELD_ID, seed=seed)

    depth = min(6, n_bits)
    qam_adapter = get(QAM_ID).adapter
    qam_factory, qam_args, qam_kwargs = prepare(
        QAM_ID,
        {"n_bits": n_bits, "depth": depth, "steps": int(steps), "lr": float(lr)},
    )
    model = qam_factory(*qam_args, **qam_kwargs)
    log_event("model_construct", procedure_id=QAM_ID, n_bits=n_bits, seed=seed, depth=model.depth)
    initial = np.array(np.asarray(model.params), dtype=float, copy=True)
    qam_train_s = float(qam_adapter.train(model, patterns))
    log_event("model_train", procedure_id=QAM_ID, seed=seed, steps=int(model.steps))

    noise_rows = []
    for level in noise_levels:
        spec = CorruptionSpec(BERNOULLI, float(level))
        hopfield_started = time.time()
        hopfield_recalled = []
        hopfield_queries = []
        for pattern in patterns:
            query = perturb(pattern, spec)
            hopfield_queries.append(query)
            hopfield_recalled.append(np.asarray(hopfield_adapter.recall(hopfield, query.corrupted)))
        hopfield_infer_s = (time.time() - hopfield_started) / count
        hopfield_full, hopfield_bit = batch_recall(hopfield_recalled, patterns)

        qam_started = time.time()
        qam_recalled = []
        qam_queries = []
        for pattern in patterns:
            query = perturb(pattern, spec)
            qam_queries.append(query)
            qam_recalled.append(np.asarray(qam_adapter.recall(model, query.corrupted)))
        qam_infer_s = (time.time() - qam_started) / count
        qam_full, qam_bit = batch_recall(qam_recalled, patterns)
        log_event(
            "model_evaluate",
            n_bits=n_bits,
            trial_id=trial_id,
            noise_p=float(level),
            queries=count,
        )
        noise_rows.append(
            {
                "noise_p": float(level),
                "arms": [
                    _arm(
                        HOPFIELD_ID,
                        "Hopfield",
                        hopfield_full,
                        hopfield_bit,
                        hopfield_train_s,
                        hopfield_infer_s,
                        n_bits ** 2,
                        hopfield_recalled,
                        hopfield_queries,
                    ),
                    _arm(
                        QAM_ID,
                        "QAM",
                        qam_full,
                        qam_bit,
                        qam_train_s,
                        qam_infer_s,
                        int(model.param_count),
                        qam_recalled,
                        qam_queries,
                    ),
                ],
            }
        )
    return {
        "protocol": PROTOCOL,
        "grid": grid,
        "n_bits": int(n_bits),
        "m": int(count),
        "trial_id": int(trial_id),
        "seed": int(seed),
        "steps": int(model.steps),
        "depth_used": int(model.depth),
        "lr": float(model.lr),
        "patterns": _matrix(patterns),
        "init_params": initial.tolist(),
        "noise": noise_rows,
    }


def _arm(
    procedure_id,
    label,
    full_acc,
    bit_acc,
    train_time_s,
    infer_time_s,
    params_count,
    recalled,
    queries,
):
    return {
        "procedure_id": procedure_id,
        "historical_model_label": label,
        "full_acc": float(full_acc),
        "bit_acc": float(bit_acc),
        "train_time_s": float(train_time_s),
        "infer_time_s": float(infer_time_s),
        "params_count": int(params_count),
        "extra_notes": "",
        "recalled_patterns": _matrix(recalled),
        "queries": [_matrix([query.corrupted])[0] for query in queries],
    }


def _write_tables(run_dir, payload) -> None:
    write_csv_table(run_dir, "trials.csv", _trial_rows(payload), TRIAL_COLUMNS)
    write_json(run_dir, "trials.json", {"trials": payload["trials"]})


def _trial_rows(payload):
    rows = []
    for trial in payload["trials"]:
        for noise in trial["noise"]:
            for arm in noise["arms"]:
                rows.append(
                    {
                        "protocol": PROTOCOL,
                        "grid": payload["grid"],
                        "comparable_to_archive": False,
                        "procedure_id": arm["procedure_id"],
                        "historical_model_label": arm["historical_model_label"],
                        "n_bits": trial["n_bits"],
                        "m": trial["m"],
                        "trial_id": trial["trial_id"],
                        "seed": trial["seed"],
                        "noise_p": noise["noise_p"],
                        "full_acc": arm["full_acc"],
                        "bit_acc": arm["bit_acc"],
                        "train_time_s": arm["train_time_s"],
                        "infer_time_s": arm["infer_time_s"],
                        "params_count": arm["params_count"],
                        "extra_notes": arm["extra_notes"],
                    }
                )
    return rows


def _matrix(rows):
    return [[int(value) for value in np.asarray(row).tolist()] for row in rows]
