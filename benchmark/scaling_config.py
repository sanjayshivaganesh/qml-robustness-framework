"""Load a scaling request. Does not construct a model or start training.

The noise-sweep schema stays in ``benchmark.config_loader``. This parser
accepts ``protocol`` ``scaling`` only. A historical document is valid and
not executable until ``allow_historical_grid`` is passed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark.config_loader import ConfigNotExecutable, ConfigValidationError
from benchmark.output import assert_output_allowed
from benchmark.protocols.scaling import (
    DEPTH_RULE,
    GENERATOR,
    MAX_SMOKE_NOISE_LEVELS,
    MAX_SMOKE_STEPS,
    MAX_SMOKE_TRIALS,
    MAX_SMOKE_WIDTHS,
    PATTERN_RULE,
    PROTOCOL,
    SEED_FORMULA,
    SEED_POLICY,
    TRAINING_SCHEDULE,
    ScalingRequest,
    historical_scaling_request,
    is_historical_scaling_request,
    qam_rot_minibatch,
)
from src.paths import PROJECT_ROOT

SCHEMA_VERSION = 1
DOCUMENT_ROLE = "benchmark_request"
LOADER_STATUS = "not_implemented"
AUDIT_PATH = "baseline/configs/original_experiments.json"
AUDIT_KEY = "scaling"
AUDIT_ROLE = "audit_record_only"
HOPFIELD_ID = "hopfield_async_random"
QAM_ID = "qam_rot_minibatch"
HOPFIELD_SOURCE = "src.hopfield.HopfieldAsyncRandom"
QAM_SOURCE = "experiments.scaling.QAM"
OUTPUT_FILES = ("manifest.json", "trials.csv", "trials.json")
ORIGIN_HISTORICAL = "historical"
ORIGIN_DEMONSTRATION = "demonstration"

TOP_LEVEL = frozenset(
    {
        "schema_version",
        "document_role",
        "loader_status",
        "experiment",
        "historical_reference",
        "grid",
        "comparable_to_archive",
        "dataset",
        "models",
        "perturbation",
        "metrics",
        "timings",
        "seeds",
        "training_schedule",
        "declared_training_count",
        "execution",
        "output",
        "setting_origin",
    }
)


@dataclass(frozen=True)
class ScalingConfig:
    """A schema-valid scaling request.

    ``executable`` is true only for a smoke budget. The historical module
    grid stays false until ``to_runner_request(allow_historical_grid=True)``.
    """

    source_path: str
    experiment_id: str
    protocol: str
    procedure_ids: tuple[str, ...]
    grid: str
    n_bits_list: tuple[int, ...]
    noise_levels: tuple[float, ...]
    trial_ids: tuple[int, ...]
    steps: int
    lr: float
    seed_base: int
    output_destination: str | None
    executable: bool
    execution_blockers: tuple[str, ...]

    def resolved_output_dir(self):
        if self.output_destination is None:
            return None
        path = Path(self.output_destination)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    def to_runner_request(self, *, allow_historical_grid: bool = False):
        request = ScalingRequest(
            n_bits_list=self.n_bits_list,
            noise_levels=self.noise_levels,
            trial_ids=self.trial_ids,
            steps=self.steps,
            lr=self.lr,
            seed_base=self.seed_base,
            grid=self.grid,
        )
        if allow_historical_grid:
            if not is_historical_scaling_request(request):
                raise ConfigNotExecutable(
                    ("allow_historical_grid accepts only the historical scaling grid.",)
                )
            return request
        if not self.executable:
            raise ConfigNotExecutable(self.execution_blockers)
        return request


def load_scaling_document(source: Path, payload):
    """Validate ``payload`` and return a ``ScalingConfig``. Does not train."""
    parsed = _parse(payload)
    blockers = _execution_blockers(parsed)
    return ScalingConfig(
        source_path=str(source),
        experiment_id=parsed["experiment_id"],
        protocol=PROTOCOL,
        procedure_ids=(HOPFIELD_ID, QAM_ID),
        grid=parsed["grid"],
        n_bits_list=tuple(parsed["n_bits_list"]),
        noise_levels=tuple(parsed["noise_levels"]),
        trial_ids=tuple(parsed["trial_ids"]),
        steps=parsed["steps"],
        lr=parsed["lr"],
        seed_base=parsed["seed_base"],
        output_destination=parsed["destination"],
        executable=not blockers,
        execution_blockers=blockers,
    )


def _parse(payload):
    errors = []
    if not isinstance(payload, dict):
        raise ConfigValidationError(["document: expected a JSON object"])
    unknown = sorted(set(payload) - TOP_LEVEL)
    missing = sorted(TOP_LEVEL - set(payload))
    for key in unknown:
        errors.append(f"document: unknown field {key}")
    for key in missing:
        errors.append(f"document: missing {key}")
    if errors:
        raise ConfigValidationError(errors)

    _exact(payload, "schema_version", SCHEMA_VERSION, "schema_version", errors, integer=True)
    _exact_str(payload, "document_role", DOCUMENT_ROLE, "document_role", errors)
    _exact_str(payload, "loader_status", LOADER_STATUS, "loader_status", errors)
    _one_of(payload, "grid", {"smoke", "full"}, "grid", errors)
    _exact_bool(payload, "comparable_to_archive", False, "comparable_to_archive", errors)
    _exact_str(payload, "training_schedule", TRAINING_SCHEDULE, "training_schedule", errors)

    experiment = _object(payload, "experiment", errors)
    reference = _object(payload, "historical_reference", errors)
    dataset = _object(payload, "dataset", errors)
    perturbation = _object(payload, "perturbation", errors)
    seeds = _object(payload, "seeds", errors)
    execution = _object(payload, "execution", errors)
    output = _object(payload, "output", errors)
    origins = _object(payload, "setting_origin", errors)
    models = payload.get("models")

    experiment_id = ""
    if isinstance(experiment, dict):
        _reject_unknown(experiment, {"experiment_id", "protocol"}, "experiment", errors)
        experiment_id = _non_empty_str(experiment, "experiment_id", "experiment.experiment_id", errors)
        if experiment.get("protocol") != PROTOCOL:
            errors.append(f"experiment.protocol: expected {PROTOCOL!r}")

    if isinstance(reference, dict):
        _reject_unknown(reference, {"path", "experiment_key", "role"}, "historical_reference", errors)
        _exact_str(reference, "path", AUDIT_PATH, "historical_reference.path", errors)
        _exact_str(reference, "experiment_key", AUDIT_KEY, "historical_reference.experiment_key", errors)
        _exact_str(reference, "role", AUDIT_ROLE, "historical_reference.role", errors)

    audit = _read_audit(errors)
    n_bits_list = _parse_dataset(dataset, errors) if isinstance(dataset, dict) else []
    noise_levels = _parse_perturbation(perturbation, errors) if isinstance(perturbation, dict) else []
    steps, lr = _parse_models(models, errors)
    seed_base, trial_ids = _parse_seeds(seeds, errors) if isinstance(seeds, dict) else (None, [])
    if isinstance(execution, dict):
        _parse_execution(execution, errors)
    destination = _parse_output(output, errors) if isinstance(output, dict) else None
    metrics = payload.get("metrics")
    timings = payload.get("timings")
    if metrics != ["batch_recall"]:
        errors.append("metrics: expected ['batch_recall']")
    if timings != ["train_time_s", "infer_time_s"]:
        errors.append("timings: expected ['train_time_s', 'infer_time_s']")
    declared = payload.get("declared_training_count")
    if type(declared) is not int or isinstance(declared, bool) or declared < 1:
        errors.append("declared_training_count: expected an integer >= 1")
    elif declared != len(n_bits_list) * len(trial_ids):
        errors.append(
            "declared_training_count: must equal len(n_bits_list) * len(trial_ids). "
            "Noise levels are scored after training and are not extra trainings."
        )
    if audit is not None and isinstance(origins, dict):
        _check_origins(
            origins,
            n_bits_list=n_bits_list,
            noise_levels=noise_levels,
            trial_ids=trial_ids,
            steps=steps,
            grid=payload.get("grid"),
            audit=audit,
            errors=errors,
        )
    if errors:
        raise ConfigValidationError(errors)
    return {
        "experiment_id": experiment_id,
        "grid": payload["grid"],
        "n_bits_list": n_bits_list,
        "noise_levels": noise_levels,
        "trial_ids": trial_ids,
        "steps": steps,
        "lr": lr,
        "seed_base": seed_base,
        "destination": destination,
    }


def _execution_blockers(parsed):
    blockers = []
    if parsed["grid"] != "smoke":
        blockers.append("grid 'full' is not executable without allow_historical_grid.")
    if parsed["steps"] < 1 or parsed["steps"] > MAX_SMOKE_STEPS:
        blockers.append(
            f"steps {parsed['steps']} is outside the smoke range 1..{MAX_SMOKE_STEPS}. "
            f"The historical setting is {qam_rot_minibatch.PROTOCOL_STEPS}."
        )
    if len(parsed["n_bits_list"]) > MAX_SMOKE_WIDTHS:
        blockers.append(f"width count exceeds the smoke cap {MAX_SMOKE_WIDTHS}.")
    if len(parsed["trial_ids"]) > MAX_SMOKE_TRIALS:
        blockers.append(f"trial count exceeds the smoke cap {MAX_SMOKE_TRIALS}.")
    if len(parsed["noise_levels"]) > MAX_SMOKE_NOISE_LEVELS:
        blockers.append(f"noise-level count exceeds the smoke cap {MAX_SMOKE_NOISE_LEVELS}.")
    historical = historical_scaling_request()
    if (
        tuple(parsed["n_bits_list"]) == historical.n_bits_list
        and tuple(parsed["trial_ids"]) == historical.trial_ids
    ):
        blockers.append("this request is the historical scaling training grid and is not started.")
    return tuple(blockers)


def _parse_dataset(dataset, errors):
    _reject_unknown(
        dataset,
        {"pattern_generator", "alphabet", "n_bits_list", "n_patterns"},
        "dataset",
        errors,
    )
    if dataset.get("pattern_generator") != GENERATOR:
        errors.append(f"dataset.pattern_generator: expected {GENERATOR!r}")
    if dataset.get("alphabet") != [-1, 1]:
        errors.append("dataset.alphabet: expected [-1, 1]")
    rule = dataset.get("n_patterns")
    if not isinstance(rule, dict) or set(rule) != {"rule_id"}:
        errors.append("dataset.n_patterns: expected {rule_id}")
    elif rule.get("rule_id") != PATTERN_RULE:
        errors.append(f"dataset.n_patterns.rule_id: expected {PATTERN_RULE!r}")
    values = dataset.get("n_bits_list")
    if not isinstance(values, list) or not values:
        errors.append("dataset.n_bits_list: expected a non-empty list of integers")
        return []
    cleaned = []
    for value in values:
        if type(value) is not int or isinstance(value, bool) or value < 1:
            errors.append(f"dataset.n_bits_list: expected integers >= 1, got {value!r}")
        else:
            cleaned.append(value)
    return cleaned


def _parse_perturbation(perturbation, errors):
    _reject_unknown(
        perturbation,
        {"procedure_id", "implementation", "levels", "when"},
        "perturbation",
        errors,
    )
    if perturbation.get("procedure_id") != "bernoulli":
        errors.append("perturbation.procedure_id: expected 'bernoulli'")
    if perturbation.get("implementation") != "inline_rand_mask":
        errors.append(
            "perturbation.implementation: expected 'inline_rand_mask'. "
            "The runner calls corrupt_bernoulli on the same legacy stream."
        )
    if perturbation.get("when") != "each_stored_pattern_after_both_models_train":
        errors.append(
            "perturbation.when: expected 'each_stored_pattern_after_both_models_train'"
        )
    levels = perturbation.get("levels")
    if not isinstance(levels, list) or not levels:
        errors.append("perturbation.levels: expected a non-empty list of numbers")
        return []
    cleaned = []
    for level in levels:
        if isinstance(level, bool) or not isinstance(level, (int, float)) or not 0.0 <= float(level) <= 1.0:
            errors.append(f"perturbation.levels: expected numbers in [0, 1], got {level!r}")
        else:
            cleaned.append(float(level))
    return cleaned


def _parse_models(models, errors):
    if not isinstance(models, list) or len(models) != 2:
        errors.append("models: expected hopfield_async_random then qam_rot_minibatch")
        return None, None
    hopfield, qam = models
    _check_hopfield(hopfield, errors)
    return _check_qam(qam, errors)


def _check_hopfield(model, errors):
    if not isinstance(model, dict):
        errors.append("models[0]: expected an object")
        return
    _reject_unknown(
        model,
        {"procedure_id", "source", "parameters", "training", "recall"},
        "models[0]",
        errors,
    )
    if model.get("procedure_id") != HOPFIELD_ID:
        errors.append(f"models[0].procedure_id: expected {HOPFIELD_ID!r}")
    if model.get("source") != HOPFIELD_SOURCE:
        errors.append(f"models[0].source: expected {HOPFIELD_SOURCE!r}")
    parameters = model.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("steps") != 10 or type(parameters.get("steps")) is not int:
        errors.append("models[0].parameters.steps: expected the integer 10")
    if model.get("training") != {"operation": "train"}:
        errors.append("models[0].training: expected {operation: train}")
    if model.get("recall") != {"operation": "recall", "returns": ["pattern"]}:
        errors.append("models[0].recall: expected recall returning the pattern")


def _check_qam(model, errors):
    if not isinstance(model, dict):
        errors.append("models[1]: expected an object")
        return None, None
    _reject_unknown(
        model,
        {"procedure_id", "source", "parameters", "rng", "training", "recall"},
        "models[1]",
        errors,
    )
    if model.get("procedure_id") != QAM_ID:
        errors.append(
            f"models[1].procedure_id: expected {QAM_ID!r}. "
            "qam_ry_linear and qam_rot_fullbatch are different circuits."
        )
    if model.get("source") != QAM_SOURCE:
        errors.append(f"models[1].source: expected {QAM_SOURCE!r}")
    parameters = model.get("parameters")
    steps = None
    lr = None
    if not isinstance(parameters, dict):
        errors.append("models[1].parameters: expected an object")
    else:
        _reject_unknown(
            parameters,
            {
                "depth_rule_id",
                "steps",
                "lr",
                "minibatch",
                "early_stop_patience",
                "early_stop_delta",
            },
            "models[1].parameters",
            errors,
        )
        if parameters.get("depth_rule_id") != DEPTH_RULE:
            errors.append(f"models[1].parameters.depth_rule_id: expected {DEPTH_RULE!r}")
        steps = parameters.get("steps")
        if type(steps) is not int or isinstance(steps, bool) or steps < 1:
            errors.append("models[1].parameters.steps: expected an integer >= 1")
            steps = None
        lr = parameters.get("lr")
        if isinstance(lr, bool) or lr != 0.05:
            errors.append("models[1].parameters.lr: expected 0.05")
            lr = None
        if parameters.get("minibatch") != 2 or type(parameters.get("minibatch")) is not int:
            errors.append("models[1].parameters.minibatch: expected the integer 2")
        if parameters.get("early_stop_patience") != 15:
            errors.append("models[1].parameters.early_stop_patience: expected 15")
        delta = parameters.get("early_stop_delta")
        if isinstance(delta, bool) or delta != 1e-4:
            errors.append("models[1].parameters.early_stop_delta: expected 0.0001")
    rng = model.get("rng")
    if not isinstance(rng, dict):
        errors.append("models[1].rng: expected an object")
    else:
        if rng.get("family") != "legacy_global":
            errors.append("models[1].rng.family: expected 'legacy_global'")
        if rng.get("constructor_reseeds") is not False:
            errors.append("models[1].rng.constructor_reseeds: expected false")
        if rng.get("seed_policy_id") != SEED_POLICY:
            errors.append(f"models[1].rng.seed_policy_id: expected {SEED_POLICY!r}")
    if model.get("training") != {"operation": "train"}:
        errors.append("models[1].training: expected {operation: train}")
    if model.get("recall") != {"operation": "recall", "returns": ["pattern"]}:
        errors.append("models[1].recall: expected recall returning the pattern")
    return steps, lr


def _parse_seeds(seeds, errors):
    _reject_unknown(seeds, {"seed_base", "formula", "trial_ids", "seed_policy_id"}, "seeds", errors)
    if seeds.get("seed_base") != 123 or type(seeds.get("seed_base")) is not int:
        errors.append("seeds.seed_base: expected the integer 123")
    if seeds.get("formula") != SEED_FORMULA:
        errors.append(f"seeds.formula: expected {SEED_FORMULA!r}")
    if seeds.get("seed_policy_id") != SEED_POLICY:
        errors.append(f"seeds.seed_policy_id: expected {SEED_POLICY!r}")
    trial_ids = seeds.get("trial_ids")
    if not isinstance(trial_ids, list) or not trial_ids:
        errors.append("seeds.trial_ids: expected a non-empty list of integers")
        return 123, []
    cleaned = []
    for trial_id in trial_ids:
        if type(trial_id) is not int or isinstance(trial_id, bool) or trial_id < 0:
            errors.append(f"seeds.trial_ids: expected integers >= 0, got {trial_id!r}")
        else:
            cleaned.append(trial_id)
    if len(cleaned) != len(set(cleaned)):
        errors.append("seeds.trial_ids: duplicates are invalid")
    return 123, cleaned


def _parse_execution(execution, errors):
    _reject_unknown(
        execution,
        {"process_model", "max_widths", "max_trials", "max_noise_levels", "max_steps", "caps_are_authoritative_in"},
        "execution",
        errors,
    )
    if execution.get("process_model") != "single_process":
        errors.append("execution.process_model: expected 'single_process'")
    if execution.get("max_widths") != MAX_SMOKE_WIDTHS:
        errors.append(f"execution.max_widths: expected {MAX_SMOKE_WIDTHS}")
    if execution.get("max_trials") != MAX_SMOKE_TRIALS:
        errors.append(f"execution.max_trials: expected {MAX_SMOKE_TRIALS}")
    if execution.get("max_noise_levels") != MAX_SMOKE_NOISE_LEVELS:
        errors.append(f"execution.max_noise_levels: expected {MAX_SMOKE_NOISE_LEVELS}")
    if execution.get("max_steps") != MAX_SMOKE_STEPS:
        errors.append(f"execution.max_steps: expected {MAX_SMOKE_STEPS}")
    if execution.get("caps_are_authoritative_in") != "benchmark.protocols.scaling":
        errors.append("execution.caps_are_authoritative_in: expected benchmark.protocols.scaling")


def _parse_output(output, errors):
    _reject_unknown(
        output,
        {"destination", "overwrite_archived_results", "if_destination_non_empty", "files"},
        "output",
        errors,
    )
    if output.get("overwrite_archived_results") is not False:
        errors.append("output.overwrite_archived_results: expected false")
    if output.get("if_destination_non_empty") != "refuse":
        errors.append("output.if_destination_non_empty: expected 'refuse'")
    if output.get("files") != list(OUTPUT_FILES):
        errors.append(f"output.files: expected {list(OUTPUT_FILES)}")
    destination = output.get("destination")
    if destination is None:
        return None
    if not isinstance(destination, str) or not destination:
        errors.append("output.destination: expected a path string")
        return None
    path = Path(destination)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        assert_output_allowed(path)
    except Exception as exc:
        errors.append(f"output.destination: {exc}")
    return destination


def _check_origins(origins, *, n_bits_list, noise_levels, trial_ids, steps, grid, audit, errors):
    required = {
        "dataset.n_bits_list",
        "dataset.n_patterns.rule_id",
        "models[0].procedure_id",
        "models[0].parameters.steps",
        "models[1].procedure_id",
        "models[1].parameters.depth_rule_id",
        "models[1].parameters.steps",
        "models[1].parameters.lr",
        "models[1].parameters.minibatch",
        "models[1].rng.constructor_reseeds",
        "perturbation.procedure_id",
        "perturbation.levels",
        "seeds.seed_base",
        "seeds.formula",
        "seeds.trial_ids",
        "grid",
        "execution.process_model",
        "output.destination",
        "comparable_to_archive",
    }
    missing = sorted(required - set(origins))
    unknown = sorted(set(origins) - required)
    for path in missing:
        errors.append(f"setting_origin: missing {path}")
    for path in unknown:
        errors.append(f"setting_origin: unknown {path}")
    historical_bits = list(audit["n_bits_list"])
    historical_noise = [float(level) for level in audit["noise_levels"]]
    historical_trials = list(range(int(audit["trials"])))
    historical_steps = int(audit["qam"]["steps"])
    expected = {
        "dataset.n_bits_list": _origin(list(n_bits_list), historical_bits),
        "dataset.n_patterns.rule_id": ORIGIN_HISTORICAL,
        "models[0].procedure_id": ORIGIN_HISTORICAL,
        "models[0].parameters.steps": ORIGIN_HISTORICAL,
        "models[1].procedure_id": ORIGIN_HISTORICAL,
        "models[1].parameters.depth_rule_id": ORIGIN_HISTORICAL,
        "models[1].parameters.steps": _origin(steps, historical_steps),
        "models[1].parameters.lr": ORIGIN_HISTORICAL,
        "models[1].parameters.minibatch": ORIGIN_HISTORICAL,
        "models[1].rng.constructor_reseeds": ORIGIN_HISTORICAL,
        "perturbation.procedure_id": ORIGIN_HISTORICAL,
        "perturbation.levels": _origin(list(noise_levels), historical_noise),
        "seeds.seed_base": ORIGIN_HISTORICAL,
        "seeds.formula": ORIGIN_HISTORICAL,
        "seeds.trial_ids": _origin(list(trial_ids), historical_trials),
        "grid": ORIGIN_HISTORICAL if grid == "full" else ORIGIN_DEMONSTRATION,
        "execution.process_model": ORIGIN_HISTORICAL,
        "output.destination": ORIGIN_DEMONSTRATION,
        "comparable_to_archive": ORIGIN_DEMONSTRATION,
    }
    for path, want in expected.items():
        got = origins.get(path)
        if got in {ORIGIN_HISTORICAL, ORIGIN_DEMONSTRATION} and got != want:
            errors.append(
                f"setting_origin.{path}: {got!r} does not match the audit relationship; expected {want!r}"
            )


def _origin(value, audit_value):
    return ORIGIN_HISTORICAL if value == audit_value else ORIGIN_DEMONSTRATION


def _read_audit(errors):
    path = PROJECT_ROOT / AUDIT_PATH
    try:
        payload = __import__("json").loads(path.read_text(encoding="utf-8"))
        experiment = payload["experiments"][AUDIT_KEY]
        experiment["n_bits_list"]
        experiment["noise_levels"]
        experiment["trials"]
        experiment["random_seed_base"]
        experiment["qam"]["steps"]
        experiment["qam"]["lr"]
        experiment["qam"]["minibatch"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"historical_reference: cannot read scaling audit ({exc})")
        return None
    if int(experiment["random_seed_base"]) != 123:
        errors.append("historical_reference: audit seed base is not 123")
    if int(experiment["qam"]["steps"]) != int(qam_rot_minibatch.PROTOCOL_STEPS):
        errors.append("historical_reference: audit steps do not match the scaling call")
    return experiment


def _object(payload, key, errors):
    value = payload.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key}: expected an object")
        return None
    return value


def _reject_unknown(value, allowed, label, errors):
    if not isinstance(value, dict):
        return
    for key in sorted(set(value) - set(allowed)):
        errors.append(f"{label}: unknown field {key}")


def _exact(payload, key, expected, label, errors, *, integer=False):
    value = payload.get(key)
    if integer and (type(value) is not int or isinstance(value, bool)):
        errors.append(f"{label}: expected the integer {expected}")
        return
    if value != expected:
        errors.append(f"{label}: expected {expected!r}, got {value!r}")


def _exact_str(payload, key, expected, label, errors):
    if payload.get(key) != expected:
        errors.append(f"{label}: expected {expected!r}")


def _exact_bool(payload, key, expected, label, errors):
    if payload.get(key) is not expected:
        errors.append(f"{label}: expected {expected}")


def _one_of(payload, key, allowed, label, errors):
    if payload.get(key) not in allowed:
        errors.append(f"{label}: expected one of {sorted(allowed)}")


def _non_empty_str(payload, key, label, errors):
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        errors.append(f"{label}: expected a non-empty string")
        return ""
    return value
