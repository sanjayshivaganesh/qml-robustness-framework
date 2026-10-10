"""Load a circuit-depth request. Does not construct a model or train.

The noise-sweep schema stays in ``benchmark.config_loader``. This parser
accepts ``protocol`` ``circuit_depth`` only. A historical document is valid
and not executable until ``allow_historical_grid`` is passed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import experiments.circuit_depth as historical
from benchmark.config_loader import ConfigNotExecutable, ConfigValidationError
from benchmark.output import assert_output_allowed
from benchmark.protocols.circuit_depth import (
    FRAMEWORK_PROCESS,
    HISTORICAL_PROCESS,
    MAX_SMOKE_CASES,
    MAX_SMOKE_DEPTHS,
    NOT_APPLIED_DEPTH_RULE,
    PATTERN_GENERATOR,
    PROTOCOL,
    QAM_ID,
    RECALL_IMPLEMENTATION,
    TRAINING_SCHEDULE,
    CircuitDepthRequest,
    historical_cases,
    historical_cell_count,
    historical_depths,
    historical_request,
    is_historical_circuit_depth_request,
)
from src.paths import PROJECT_ROOT

SCHEMA_VERSION = 1
DOCUMENT_ROLE = "benchmark_request"
LOADER_STATUS = "not_implemented"
AUDIT_PATH = "baseline/configs/original_experiments.json"
AUDIT_KEY = "circuit_depth"
AUDIT_ROLE = "audit_record_only"
QAM_SOURCE = "src.qam.train_ry_linear_qam"
DEPTH_DEFINITION = "ansatz_layer_count"
OUTPUT_FILES = (
    "manifest.json",
    "configuration.json",
    "rows.csv",
    "rows.json",
    "trials.json",
)
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
        "aggregation",
        "seeds",
        "training_schedule",
        "declared_cell_count",
        "declared_trainings",
        "execution",
        "output",
        "setting_origin",
    }
)


@dataclass(frozen=True)
class CircuitDepthConfig:
    """A schema-valid circuit-depth request.

    ``executable`` is true only for one module cell at the module training
    budget. The 21-cell grid stays false until
    ``to_runner_request(allow_historical_grid=True)``.
    """

    source_path: str
    experiment_id: str
    protocol: str
    procedure_ids: tuple[str, ...]
    grid: str
    cases: tuple
    depths: tuple[int, ...]
    seeds: tuple[int, ...]
    train_steps: int
    lr: float
    recall_noise: float
    return_tuple: bool
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
        request = CircuitDepthRequest(
            cases=self.cases,
            depths=self.depths,
            seeds=self.seeds,
            train_steps=self.train_steps,
            lr=self.lr,
            recall_noise=self.recall_noise,
            return_tuple=self.return_tuple,
            grid=self.grid,
        )
        if allow_historical_grid:
            if not is_historical_circuit_depth_request(request):
                raise ConfigNotExecutable(
                    (
                        "allow_historical_grid accepts only the historical "
                        "circuit-depth grid.",
                    )
                )
            return request
        if not self.executable:
            raise ConfigNotExecutable(self.execution_blockers)
        return request


def load_circuit_depth_document(source: Path, payload):
    """Validate ``payload`` and return a config. Does not train."""
    parsed = _parse(payload)
    blockers = _execution_blockers(parsed)
    return CircuitDepthConfig(
        source_path=str(source),
        experiment_id=parsed["experiment_id"],
        protocol=PROTOCOL,
        procedure_ids=(QAM_ID,),
        grid=parsed["grid"],
        cases=tuple(parsed["cases"]),
        depths=tuple(parsed["depths"]),
        seeds=tuple(parsed["seeds"]),
        train_steps=parsed["train_steps"],
        lr=parsed["lr"],
        recall_noise=parsed["recall_noise"],
        return_tuple=False,
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
    aggregation = _object(payload, "aggregation", errors)
    seeds = _object(payload, "seeds", errors)
    execution = _object(payload, "execution", errors)
    output = _object(payload, "output", errors)
    origins = _object(payload, "setting_origin", errors)

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

    _read_audit(errors)

    cases = []
    depths = []
    if isinstance(dataset, dict):
        _reject_unknown(
            dataset,
            {
                "pattern_generator",
                "cases",
                "depths",
                "depth_definition",
                "depth_rule_not_applied",
            },
            "dataset",
            errors,
        )
        _exact_str(dataset, "pattern_generator", PATTERN_GENERATOR, "dataset.pattern_generator", errors)
        _exact_str(
            dataset,
            "depth_definition",
            DEPTH_DEFINITION,
            "dataset.depth_definition",
            errors,
        )
        _exact_str(
            dataset,
            "depth_rule_not_applied",
            NOT_APPLIED_DEPTH_RULE,
            "dataset.depth_rule_not_applied",
            errors,
        )
        cases = _cases(dataset.get("cases"), errors)
        depths = _depths(dataset.get("depths"), errors)

    models = payload.get("models")
    if not isinstance(models, list) or len(models) != 1:
        errors.append("models: expected one qam_ry_linear entry")
    elif isinstance(models[0], dict):
        _model(models[0], errors)
    else:
        errors.append("models[0]: expected an object")

    if isinstance(perturbation, dict):
        _perturbation(perturbation, errors)
    metrics = payload.get("metrics")
    if metrics != ["batch_recall"]:
        errors.append("metrics: expected ['batch_recall']")

    if isinstance(aggregation, dict):
        _reject_unknown(
            aggregation,
            {"method", "bootstrap_used", "parameter_count"},
            "aggregation",
            errors,
        )
        _exact_str(aggregation, "method", "mean_over_trials", "aggregation.method", errors)
        _exact_bool(aggregation, "bootstrap_used", False, "aggregation.bootstrap_used", errors)
        _exact_str(
            aggregation,
            "parameter_count",
            "depth * n_bits",
            "aggregation.parameter_count",
            errors,
        )

    seed_values = []
    if isinstance(seeds, dict):
        _reject_unknown(
            seeds,
            {"values", "generator", "reapplied_for_every_cell", "pennylane_seed_call"},
            "seeds",
            errors,
        )
        seed_values = _seed_list(seeds.get("values"), errors)
        _exact_str(seeds, "generator", "numpy.random.seed", "seeds.generator", errors)
        _exact_bool(seeds, "reapplied_for_every_cell", True, "seeds.reapplied_for_every_cell", errors)
        _exact_bool(seeds, "pennylane_seed_call", False, "seeds.pennylane_seed_call", errors)

    declared_cells = _positive_int(payload, "declared_cell_count", "declared_cell_count", errors)
    declared_trainings = _positive_int(payload, "declared_trainings", "declared_trainings", errors)
    if cases and depths and declared_cells is not None:
        expected_cells = len(cases) * len(depths)
        if declared_cells != expected_cells:
            errors.append(
                f"declared_cell_count: expected {expected_cells}, got {declared_cells}"
            )
        expected_trainings = expected_cells * len(historical_request().seeds)
        if declared_trainings is not None and declared_trainings != expected_trainings:
            errors.append(
                f"declared_trainings: expected {expected_trainings}, got {declared_trainings}"
            )

    if isinstance(execution, dict):
        _reject_unknown(
            execution,
            {
                "max_cases",
                "max_depths",
                "process_model",
                "historical_process_model",
            },
            "execution",
            errors,
        )
        _exact(execution, "max_cases", MAX_SMOKE_CASES, "execution.max_cases", errors, integer=True)
        _exact(
            execution,
            "max_depths",
            MAX_SMOKE_DEPTHS,
            "execution.max_depths",
            errors,
            integer=True,
        )
        _exact_str(
            execution,
            "process_model",
            FRAMEWORK_PROCESS,
            "execution.process_model",
            errors,
        )
        _exact_str(
            execution,
            "historical_process_model",
            HISTORICAL_PROCESS,
            "execution.historical_process_model",
            errors,
        )

    destination = None
    if isinstance(output, dict):
        _reject_unknown(output, {"destination", "files", "archive_comparison"}, "output", errors)
        destination = _non_empty_str(output, "destination", "output.destination", errors)
        if output.get("files") != list(OUTPUT_FILES):
            errors.append(f"output.files: expected {list(OUTPUT_FILES)}")
        _exact_bool(output, "archive_comparison", False, "output.archive_comparison", errors)
        if destination and "results" in Path(destination).parts:
            errors.append("output.destination: archived results are not a framework destination")

    if isinstance(origins, dict) and cases and depths:
        _origins(origins, payload.get("grid"), cases, depths, errors)

    if errors:
        raise ConfigValidationError(errors)
    return {
        "experiment_id": experiment_id,
        "grid": payload["grid"],
        "cases": cases,
        "depths": depths,
        "seeds": seed_values,
        "train_steps": int(historical.train_steps),
        "lr": float(historical.lr),
        "recall_noise": float(historical.recall_noise),
        "destination": destination,
    }


def _execution_blockers(parsed):
    blockers = []
    if parsed["grid"] != "smoke":
        blockers.append("grid is full; the 21-cell depth sweep is not started")
    if len(parsed["cases"]) > MAX_SMOKE_CASES:
        blockers.append(f"cases exceed the smoke cap of {MAX_SMOKE_CASES}")
    if len(parsed["depths"]) > MAX_SMOKE_DEPTHS:
        blockers.append(f"depths exceed the smoke cap of {MAX_SMOKE_DEPTHS}")
    cells = len(parsed["cases"]) * len(parsed["depths"])
    if cells == historical_cell_count():
        blockers.append(f"declared schedule has all {historical_cell_count()} historical cells")
    if parsed["destination"]:
        try:
            assert_output_allowed(Path(parsed["destination"]))
        except Exception as exc:
            blockers.append(str(exc))
    return tuple(blockers)


def _read_audit(errors):
    path = PROJECT_ROOT / AUDIT_PATH
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))["experiments"][AUDIT_KEY]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        errors.append(f"historical_reference: unreadable audit record ({exc})")
        return
    expected_cases = {str(n_bits): m for n_bits, m in historical_cases()}
    if audit.get("cases") != expected_cases:
        errors.append("historical_reference: audit cases differ from the module")
    if list(audit.get("depths", [])) != list(historical_depths()):
        errors.append("historical_reference: audit depths differ from the module")
    if audit.get("trials") != int(historical.trials):
        errors.append("historical_reference: audit trials differ from the module")
    if audit.get("train_steps") != int(historical.train_steps):
        errors.append("historical_reference: audit train_steps differ from the module")
    if audit.get("lr") != float(historical.lr):
        errors.append("historical_reference: audit lr differs from the module")
    if audit.get("recall_noise") != float(historical.recall_noise):
        errors.append("historical_reference: audit recall_noise differ from the module")
    qam = audit.get("qam")
    if not isinstance(qam, dict) or qam.get("return_tuple") is not False:
        errors.append("historical_reference: audit return_tuple is not false")
    if not isinstance(qam, dict) or qam.get("function") != QAM_SOURCE:
        errors.append("historical_reference: audit trainer is not train_ry_linear_qam")


def _model(model, errors):
    _reject_unknown(
        model,
        {"procedure_id", "role", "source", "recall", "parameters"},
        "models[0]",
        errors,
    )
    if model.get("procedure_id") != QAM_ID:
        errors.append(f"models[0].procedure_id: expected {QAM_ID!r}")
    _exact_str(model, "role", "only_model", "models[0].role", errors)
    _exact_str(model, "source", QAM_SOURCE, "models[0].source", errors)
    if model.get("recall") != RECALL_IMPLEMENTATION:
        errors.append(
            "models[0].recall: expected the inline sign map. "
            "qam_recall is the storage-capacity call."
        )
    parameters = model.get("parameters")
    if not isinstance(parameters, dict):
        errors.append("models[0].parameters: expected an object")
        return
    _reject_unknown(
        parameters,
        {"return_tuple", "train_steps", "lr", "init_scale", "encoding", "ansatz"},
        "models[0].parameters",
        errors,
    )
    if parameters.get("return_tuple") is not False:
        errors.append("models[0].parameters.return_tuple: expected false")
    _exact(
        parameters,
        "train_steps",
        int(historical.train_steps),
        "models[0].parameters.train_steps",
        errors,
        integer=True,
    )
    if parameters.get("lr") != float(historical.lr):
        errors.append(f"models[0].parameters.lr: expected {float(historical.lr)}")
    if parameters.get("init_scale") != 0.01:
        errors.append("models[0].parameters.init_scale: expected 0.01")
    _exact_str(parameters, "encoding", "RY(x * pi)", "models[0].parameters.encoding", errors)
    _exact_str(parameters, "ansatz", "RY + linear CNOT", "models[0].parameters.ansatz", errors)


def _perturbation(perturbation, errors):
    _reject_unknown(
        perturbation,
        {"procedure_id", "implementation", "level", "rng", "flip_rule", "when"},
        "perturbation",
        errors,
    )
    _exact_str(perturbation, "procedure_id", "fixed_count", "perturbation.procedure_id", errors)
    _exact_str(
        perturbation,
        "implementation",
        "src.data_generation.corrupt_fixed_count",
        "perturbation.implementation",
        errors,
    )
    if perturbation.get("level") != float(historical.recall_noise):
        errors.append(f"perturbation.level: expected {float(historical.recall_noise)}")
    _exact_str(perturbation, "rng", "numpy.random", "perturbation.rng", errors)
    _exact_str(
        perturbation,
        "flip_rule",
        "max(1, int(n_bits * level))",
        "perturbation.flip_rule",
        errors,
    )
    _exact_str(
        perturbation,
        "when",
        "each_stored_pattern_after_training",
        "perturbation.when",
        errors,
    )


def _cases(value, errors):
    allowed = historical_cases()
    if not isinstance(value, list) or not value:
        errors.append("dataset.cases: expected a non-empty list")
        return []
    parsed = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"dataset.cases[{index}]: expected an object")
            continue
        _reject_unknown(item, {"n_bits", "m"}, f"dataset.cases[{index}]", errors)
        n_bits = item.get("n_bits")
        m = item.get("m")
        if isinstance(n_bits, bool) or not isinstance(n_bits, int) or n_bits < 1:
            errors.append(f"dataset.cases[{index}].n_bits: expected a positive integer")
            continue
        if isinstance(m, bool) or not isinstance(m, int) or m < 1:
            errors.append(f"dataset.cases[{index}].m: expected a positive integer")
            continue
        pair = (int(n_bits), int(m))
        if pair not in allowed:
            errors.append(
                f"dataset.cases[{index}]: {(n_bits, m)} is not a module (n, m) pair"
            )
        parsed.append(pair)
    if parsed and tuple(parsed) != _ordered_subset(parsed, allowed):
        errors.append("dataset.cases: pairs must keep the module order")
    return parsed


def _depths(value, errors):
    allowed = historical_depths()
    if not isinstance(value, list) or not value:
        errors.append("dataset.depths: expected a non-empty list")
        return []
    parsed = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            errors.append(f"dataset.depths[{index}]: expected a positive integer")
            continue
        if int(item) not in allowed:
            errors.append(f"dataset.depths[{index}]: {item} is not a module depth")
        parsed.append(int(item))
    if parsed and tuple(parsed) != _ordered_subset(parsed, allowed):
        errors.append("dataset.depths: values must keep the module order")
    return parsed


def _ordered_subset(chosen, allowed):
    positions = [allowed.index(item) for item in chosen if item in allowed]
    if len(positions) != len(chosen) or positions != sorted(positions) or len(set(positions)) != len(positions):
        return None
    return tuple(chosen)


def _seed_list(value, errors):
    expected = list(range(int(historical.trials)))
    if value != expected:
        errors.append(f"seeds.values: expected {expected}")
        return []
    return expected


def _origins(origins, grid, cases, depths, errors):
    expected_keys = {
        "cases",
        "depths",
        "return_tuple",
        "train_steps",
        "lr",
        "recall_noise",
        "seeds",
        "aggregation",
        "process_model",
        "destination",
        "comparable_to_archive",
    }
    _reject_unknown(origins, expected_keys, "setting_origin", errors)
    for key in sorted(expected_keys - set(origins)):
        errors.append(f"setting_origin: missing {key}")
    cases_origin = ORIGIN_HISTORICAL if tuple(cases) == historical_cases() else ORIGIN_DEMONSTRATION
    depths_origin = ORIGIN_HISTORICAL if tuple(depths) == historical_depths() else ORIGIN_DEMONSTRATION
    _exact_str(origins, "cases", cases_origin, "setting_origin.cases", errors)
    _exact_str(origins, "depths", depths_origin, "setting_origin.depths", errors)
    for key in (
        "return_tuple",
        "train_steps",
        "lr",
        "recall_noise",
        "seeds",
        "aggregation",
        "process_model",
    ):
        _exact_str(origins, key, ORIGIN_HISTORICAL, f"setting_origin.{key}", errors)
    _exact_str(origins, "destination", ORIGIN_DEMONSTRATION, "setting_origin.destination", errors)
    _exact_str(
        origins,
        "comparable_to_archive",
        ORIGIN_DEMONSTRATION,
        "setting_origin.comparable_to_archive",
        errors,
    )
    if grid == "full" and (cases_origin != ORIGIN_HISTORICAL or depths_origin != ORIGIN_HISTORICAL):
        errors.append("grid full requires the historical cases and depths")
    if grid == "smoke" and cases_origin == ORIGIN_HISTORICAL and depths_origin == ORIGIN_HISTORICAL:
        errors.append("grid smoke cannot carry the full case and depth lists")


def _object(payload, key, errors):
    value = payload.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key}: expected an object")
        return None
    return value


def _reject_unknown(payload, allowed, label, errors):
    for key in sorted(set(payload) - set(allowed)):
        errors.append(f"{label}: unknown field {key}")


def _one_of(payload, key, allowed, label, errors):
    if payload.get(key) not in allowed:
        errors.append(f"{label}: expected one of {sorted(allowed)}")


def _exact(payload, key, expected, label, errors, *, integer=False):
    value = payload.get(key)
    if integer and (isinstance(value, bool) or not isinstance(value, int)):
        errors.append(f"{label}: expected integer {expected}")
        return
    if value != expected:
        errors.append(f"{label}: expected {expected!r}")


def _exact_str(payload, key, expected, label, errors):
    if payload.get(key) != expected:
        errors.append(f"{label}: expected {expected!r}")


def _exact_bool(payload, key, expected, label, errors):
    if payload.get(key) is not expected:
        errors.append(f"{label}: expected {expected}")


def _non_empty_str(payload, key, label, errors):
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label}: expected a non-empty string")
        return ""
    return value


def _positive_int(payload, key, label, errors):
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        errors.append(f"{label}: expected a positive integer")
        return None
    return value
