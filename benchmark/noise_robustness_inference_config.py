"""Load an inference request. Does not recall a pattern or train a model.

The noise-sweep schema stays in ``benchmark.config_loader``. This parser
accepts ``protocol`` ``noise_robustness_inference`` only. A historical
document is valid and not executable until ``allow_historical_grid`` is
passed. Loading does not call ``run_trial``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import experiments.noise_robustness_inference as historical
from benchmark.config_loader import ConfigNotExecutable, ConfigValidationError
from benchmark.output import assert_output_allowed
from benchmark.protocols.noise_robustness_inference import (
    CSV_MODEL_LABEL,
    FRAMEWORK_PROCESS,
    HAMMING_ID,
    HISTORICAL_PROCESS,
    HOPFIELD_ID,
    MAX_SMOKE_CAPACITIES,
    MAX_SMOKE_NOISE_LEVELS,
    MAX_SMOKE_WIDTHS,
    PATTERN_GENERATOR,
    PROTOCOL,
    SEED_FORMULA,
    TRAINING_SCHEDULE,
    InferenceRequest,
    historical_noise_levels,
    historical_task_count,
    is_historical_inference_request,
)
from src.paths import PROJECT_ROOT

SCHEMA_VERSION = 1
DOCUMENT_ROLE = "benchmark_request"
LOADER_STATUS = "not_implemented"
AUDIT_PATH = "baseline/configs/original_experiments.json"
AUDIT_KEY = "noise_robustness_inference"
AUDIT_ROLE = "audit_record_only"
HOPFIELD_SOURCE = "src.hopfield.hopfield_async_sequential_recall"
HAMMING_SOURCE = "src.qam.nearest_neighbor_recall"
OUTPUT_FILES = (
    "manifest.json",
    "configuration.json",
    "rows.csv",
    "rows.json",
    "summary.csv",
    "summary.json",
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
        "randomness",
        "trials_per_config",
        "training_schedule",
        "declared_task_count",
        "declared_raw_rows",
        "declared_summary_rows",
        "execution",
        "output",
        "setting_origin",
    }
)


@dataclass(frozen=True)
class InferenceConfig:
    """A schema-valid inference request.

    ``executable`` is true only for one module cell at 300 trials. The
    216,000-task grid stays false until
    ``to_runner_request(allow_historical_grid=True)``.
    """

    source_path: str
    experiment_id: str
    protocol: str
    procedure_ids: tuple[str, ...]
    grid: str
    n_bits_list: tuple[int, ...]
    capacity_ratios: tuple[float, ...]
    noise_levels: tuple[float, ...]
    trial_ids: tuple[int, ...]
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
        request = InferenceRequest(
            n_bits_list=self.n_bits_list,
            capacity_ratios=self.capacity_ratios,
            noise_levels=self.noise_levels,
            trial_ids=self.trial_ids,
            grid=self.grid,
        )
        if allow_historical_grid:
            if not is_historical_inference_request(request):
                raise ConfigNotExecutable(
                    (
                        "allow_historical_grid accepts only the historical "
                        "inference grid.",
                    )
                )
            return request
        if not self.executable:
            raise ConfigNotExecutable(self.execution_blockers)
        return request


def load_inference_document(source: Path, payload):
    """Validate ``payload`` and return a config. Does not call ``run_trial``."""
    parsed = _parse(payload)
    blockers = _execution_blockers(parsed)
    return InferenceConfig(
        source_path=str(source),
        experiment_id=parsed["experiment_id"],
        protocol=PROTOCOL,
        procedure_ids=(HOPFIELD_ID, HAMMING_ID),
        grid=parsed["grid"],
        n_bits_list=tuple(parsed["n_bits_list"]),
        capacity_ratios=tuple(parsed["capacity_ratios"]),
        noise_levels=tuple(parsed["noise_levels"]),
        trial_ids=tuple(range(int(historical.TRIALS_PER_CONFIG))),
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
    _exact(
        payload,
        "trials_per_config",
        int(historical.TRIALS_PER_CONFIG),
        "trials_per_config",
        errors,
        integer=True,
    )

    experiment = _object(payload, "experiment", errors)
    reference = _object(payload, "historical_reference", errors)
    dataset = _object(payload, "dataset", errors)
    perturbation = _object(payload, "perturbation", errors)
    aggregation = _object(payload, "aggregation", errors)
    randomness = _object(payload, "randomness", errors)
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

    widths = []
    capacities = []
    noises = []
    if isinstance(dataset, dict):
        _reject_unknown(
            dataset,
            {
                "pattern_generator",
                "n_bits_list",
                "capacity_ratios",
                "noise_levels",
                "n_patterns",
                "query",
            },
            "dataset",
            errors,
        )
        _exact_str(dataset, "pattern_generator", PATTERN_GENERATOR, "dataset.pattern_generator", errors)
        _exact_str(
            dataset,
            "n_patterns",
            "max(1, int(n_bits * capacity_ratio))",
            "dataset.n_patterns",
            errors,
        )
        _exact_str(dataset, "query", "one stored pattern per trial", "dataset.query", errors)
        widths = _int_subsequence(
            dataset.get("n_bits_list"),
            tuple(int(n_bits) for n_bits in historical.N_BITS_LIST),
            "dataset.n_bits_list",
            errors,
        )
        capacities = _float_subsequence(
            dataset.get("capacity_ratios"),
            tuple(float(ratio) for ratio in historical.CAPACITY_RATIOS),
            "dataset.capacity_ratios",
            errors,
        )
        noises = _float_subsequence(
            dataset.get("noise_levels"),
            historical_noise_levels(),
            "dataset.noise_levels",
            errors,
        )

    _models(payload.get("models"), errors)
    if isinstance(perturbation, dict):
        _perturbation(perturbation, errors)
    if payload.get("metrics") != ["compute_metrics", "recovery_gain"]:
        errors.append("metrics: expected ['compute_metrics', 'recovery_gain']")
    if isinstance(aggregation, dict):
        _aggregation(aggregation, errors)
    if isinstance(randomness, dict):
        _randomness(randomness, errors)

    tasks = None
    if widths and capacities and noises:
        tasks = len(widths) * len(capacities) * len(noises) * int(historical.TRIALS_PER_CONFIG)
    declared_tasks = _positive_int(payload, "declared_task_count", "declared_task_count", errors)
    declared_rows = _positive_int(payload, "declared_raw_rows", "declared_raw_rows", errors)
    declared_summary = _positive_int(
        payload, "declared_summary_rows", "declared_summary_rows", errors
    )
    if tasks is not None and declared_tasks is not None and declared_tasks != tasks:
        errors.append(f"declared_task_count: expected {tasks}, got {declared_tasks}")
    if tasks is not None and declared_rows is not None and declared_rows != tasks * 2:
        errors.append(f"declared_raw_rows: expected {tasks * 2}, got {declared_rows}")
    if widths and capacities and noises and declared_summary is not None:
        expected_summary = len(widths) * len(capacities) * len(noises) * 2
        if declared_summary != expected_summary:
            errors.append(
                f"declared_summary_rows: expected {expected_summary}, got {declared_summary}"
            )

    if isinstance(execution, dict):
        _reject_unknown(
            execution,
            {
                "max_widths",
                "max_capacities",
                "max_noise_levels",
                "process_model",
                "historical_process_model",
            },
            "execution",
            errors,
        )
        _exact(execution, "max_widths", MAX_SMOKE_WIDTHS, "execution.max_widths", errors, integer=True)
        _exact(
            execution,
            "max_capacities",
            MAX_SMOKE_CAPACITIES,
            "execution.max_capacities",
            errors,
            integer=True,
        )
        _exact(
            execution,
            "max_noise_levels",
            MAX_SMOKE_NOISE_LEVELS,
            "execution.max_noise_levels",
            errors,
            integer=True,
        )
        _exact_str(execution, "process_model", FRAMEWORK_PROCESS, "execution.process_model", errors)
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

    if isinstance(origins, dict) and widths and capacities and noises:
        _origins(origins, payload.get("grid"), widths, capacities, noises, errors)

    if errors:
        raise ConfigValidationError(errors)
    return {
        "experiment_id": experiment_id,
        "grid": payload["grid"],
        "n_bits_list": widths,
        "capacity_ratios": capacities,
        "noise_levels": noises,
        "destination": destination,
    }


def _execution_blockers(parsed):
    blockers = []
    if parsed["grid"] != "smoke":
        blockers.append("grid is full; the 216000-task inference grid is not started")
    if len(parsed["n_bits_list"]) > MAX_SMOKE_WIDTHS:
        blockers.append(f"widths exceed the smoke cap of {MAX_SMOKE_WIDTHS}")
    if len(parsed["capacity_ratios"]) > MAX_SMOKE_CAPACITIES:
        blockers.append(f"capacities exceed the smoke cap of {MAX_SMOKE_CAPACITIES}")
    if len(parsed["noise_levels"]) > MAX_SMOKE_NOISE_LEVELS:
        blockers.append(f"noise levels exceed the smoke cap of {MAX_SMOKE_NOISE_LEVELS}")
    tasks = (
        len(parsed["n_bits_list"])
        * len(parsed["capacity_ratios"])
        * len(parsed["noise_levels"])
        * int(historical.TRIALS_PER_CONFIG)
    )
    if tasks == historical_task_count():
        blockers.append(f"declared schedule has all {historical_task_count()} historical tasks")
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
    if list(audit.get("n_bits_list", [])) != list(historical.N_BITS_LIST):
        errors.append("historical_reference: audit widths differ from the module")
    if list(audit.get("capacity_ratios", [])) != list(historical.CAPACITY_RATIOS):
        errors.append("historical_reference: audit capacities differ from the module")
    if audit.get("trials_per_config") != int(historical.TRIALS_PER_CONFIG):
        errors.append("historical_reference: audit trial count differs from the module")
    if audit.get("task_count") != historical_task_count():
        errors.append("historical_reference: audit task_count differs from the module")
    if audit.get("raw_row_count_expected") != historical_task_count() * 2:
        errors.append("historical_reference: audit raw row count differs from the module")
    if audit.get("summary_bootstrap") != 200:
        errors.append("historical_reference: audit bootstrap count is not 200")
    labeled = audit.get("model_labeled_qam")
    if not isinstance(labeled, dict) or labeled.get("circuit") is not False:
        errors.append("historical_reference: the qam column is not marked circuit false")
    if not isinstance(labeled, dict) or labeled.get("function") != HAMMING_SOURCE:
        errors.append("historical_reference: the qam column is not nearest_neighbor_recall")
    hopfield = audit.get("hopfield")
    if not isinstance(hopfield, dict) or hopfield.get("steps") != 20:
        errors.append("historical_reference: Hopfield steps are not 20")


def _models(models, errors):
    if not isinstance(models, list) or len(models) != 2:
        errors.append("models: expected hopfield_async_sequential then hamming_nearest")
        return
    _hopfield_model(models[0], errors)
    _hamming_model(models[1], errors)


def _hopfield_model(model, errors):
    if not isinstance(model, dict):
        errors.append("models[0]: expected an object")
        return
    _reject_unknown(
        model,
        {"procedure_id", "role", "source", "csv_model_label", "parameters"},
        "models[0]",
        errors,
    )
    if model.get("procedure_id") != HOPFIELD_ID:
        errors.append(f"models[0].procedure_id: expected {HOPFIELD_ID!r}")
    _exact_str(model, "role", "sequential_async_recall", "models[0].role", errors)
    _exact_str(model, "source", HOPFIELD_SOURCE, "models[0].source", errors)
    if model.get("csv_model_label") != CSV_MODEL_LABEL[HOPFIELD_ID]:
        errors.append("models[0].csv_model_label: expected 'hopfield'")
    parameters = model.get("parameters")
    if not isinstance(parameters, dict):
        errors.append("models[0].parameters: expected an object")
        return
    _reject_unknown(parameters, {"steps", "energy"}, "models[0].parameters", errors)
    _exact(parameters, "steps", 20, "models[0].parameters.steps", errors, integer=True)
    _exact_str(
        parameters,
        "energy",
        "E = -0.5 * x^T W x",
        "models[0].parameters.energy",
        errors,
    )


def _hamming_model(model, errors):
    if not isinstance(model, dict):
        errors.append("models[1]: expected an object")
        return
    _reject_unknown(
        model,
        {
            "procedure_id",
            "role",
            "source",
            "csv_model_label",
            "variational",
            "parameters",
        },
        "models[1]",
        errors,
    )
    if model.get("procedure_id") != HAMMING_ID:
        errors.append(f"models[1].procedure_id: expected {HAMMING_ID!r}")
    _exact_str(model, "role", "classical_nearest_neighbour", "models[1].role", errors)
    _exact_str(model, "source", HAMMING_SOURCE, "models[1].source", errors)
    if model.get("csv_model_label") != "qam":
        errors.append(
            "models[1].csv_model_label: expected 'qam'. "
            "That archive label is Hamming nearest neighbour."
        )
    if model.get("variational") is not False:
        errors.append("models[1].variational: expected false")
    if model.get("parameters") != {}:
        errors.append("models[1].parameters: expected an empty object. There is no training.")


def _perturbation(perturbation, errors):
    _reject_unknown(
        perturbation,
        {"procedure_id", "implementation", "mask", "rng"},
        "perturbation",
        errors,
    )
    _exact_str(perturbation, "procedure_id", "inline_bernoulli", "perturbation.procedure_id", errors)
    _exact_str(
        perturbation,
        "implementation",
        "experiments.noise_robustness_inference.run_trial",
        "perturbation.implementation",
        errors,
    )
    _exact_str(
        perturbation,
        "mask",
        "numpy.random.rand(n_bits) < noise_p",
        "perturbation.mask",
        errors,
    )
    _exact_str(perturbation, "rng", "numpy.random", "perturbation.rng", errors)


def _aggregation(aggregation, errors):
    _reject_unknown(
        aggregation,
        {
            "implementation",
            "bootstrap_samples",
            "bootstrap_seeded",
            "gaussian_denominator",
        },
        "aggregation",
        errors,
    )
    _exact_str(
        aggregation,
        "implementation",
        "experiments.noise_robustness_inference.aggregate_results",
        "aggregation.implementation",
        errors,
    )
    _exact(aggregation, "bootstrap_samples", 200, "aggregation.bootstrap_samples", errors, integer=True)
    _exact_bool(aggregation, "bootstrap_seeded", False, "aggregation.bootstrap_seeded", errors)
    _exact_str(
        aggregation,
        "gaussian_denominator",
        "TRIALS_PER_CONFIG",
        "aggregation.gaussian_denominator",
        errors,
    )


def _randomness(randomness, errors):
    _reject_unknown(
        randomness,
        {"base", "formula", "generator", "pennylane_seed_call"},
        "randomness",
        errors,
    )
    _exact(
        randomness,
        "base",
        int(historical.RANDOM_SEED_BASE),
        "randomness.base",
        errors,
        integer=True,
    )
    _exact_str(randomness, "formula", SEED_FORMULA, "randomness.formula", errors)
    _exact_str(randomness, "generator", "numpy.random.seed", "randomness.generator", errors)
    _exact_bool(randomness, "pennylane_seed_call", False, "randomness.pennylane_seed_call", errors)


def _int_subsequence(value, allowed, label, errors):
    if not isinstance(value, list) or not value:
        errors.append(f"{label}: expected a non-empty list")
        return []
    parsed = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            errors.append(f"{label}[{index}]: expected a positive integer")
            continue
        if int(item) not in allowed:
            errors.append(f"{label}[{index}]: {item} is not a module value")
        parsed.append(int(item))
    if parsed and tuple(parsed) != _ordered(parsed, allowed):
        errors.append(f"{label}: values must keep the module order")
    return parsed


def _float_subsequence(value, allowed, label, errors):
    if not isinstance(value, list) or not value:
        errors.append(f"{label}: expected a non-empty list")
        return []
    parsed = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            errors.append(f"{label}[{index}]: expected a number")
            continue
        matched = None
        for candidate in allowed:
            if float(item) == float(candidate):
                matched = float(candidate)
                break
        if matched is None:
            errors.append(f"{label}[{index}]: {item} is not a module value")
            continue
        parsed.append(matched)
    if parsed and tuple(parsed) != _ordered(parsed, allowed):
        errors.append(f"{label}: values must keep the module order")
    return parsed


def _ordered(chosen, allowed):
    positions = [allowed.index(item) for item in chosen if item in allowed]
    if len(positions) != len(chosen) or positions != sorted(positions) or len(set(positions)) != len(positions):
        return None
    return tuple(chosen)


def _origins(origins, grid, widths, capacities, noises, errors):
    expected_keys = {
        "n_bits_list",
        "capacity_ratios",
        "noise_levels",
        "trials_per_config",
        "seed_formula",
        "models",
        "aggregation",
        "process_model",
        "destination",
        "comparable_to_archive",
    }
    _reject_unknown(origins, expected_keys, "setting_origin", errors)
    for key in sorted(expected_keys - set(origins)):
        errors.append(f"setting_origin: missing {key}")
    widths_origin = (
        ORIGIN_HISTORICAL
        if tuple(widths) == tuple(int(n) for n in historical.N_BITS_LIST)
        else ORIGIN_DEMONSTRATION
    )
    capacity_origin = (
        ORIGIN_HISTORICAL
        if tuple(capacities) == tuple(float(ratio) for ratio in historical.CAPACITY_RATIOS)
        else ORIGIN_DEMONSTRATION
    )
    noise_origin = (
        ORIGIN_HISTORICAL if tuple(noises) == historical_noise_levels() else ORIGIN_DEMONSTRATION
    )
    _exact_str(origins, "n_bits_list", widths_origin, "setting_origin.n_bits_list", errors)
    _exact_str(origins, "capacity_ratios", capacity_origin, "setting_origin.capacity_ratios", errors)
    _exact_str(origins, "noise_levels", noise_origin, "setting_origin.noise_levels", errors)
    for key in ("trials_per_config", "seed_formula", "models", "aggregation"):
        _exact_str(origins, key, ORIGIN_HISTORICAL, f"setting_origin.{key}", errors)
    _exact_str(origins, "process_model", ORIGIN_DEMONSTRATION, "setting_origin.process_model", errors)
    _exact_str(origins, "destination", ORIGIN_DEMONSTRATION, "setting_origin.destination", errors)
    _exact_str(
        origins,
        "comparable_to_archive",
        ORIGIN_DEMONSTRATION,
        "setting_origin.comparable_to_archive",
        errors,
    )
    if grid == "full" and (
        widths_origin != ORIGIN_HISTORICAL
        or capacity_origin != ORIGIN_HISTORICAL
        or noise_origin != ORIGIN_HISTORICAL
    ):
        errors.append("grid full requires the historical widths, capacities, and noise levels")
    if (
        grid == "smoke"
        and widths_origin == ORIGIN_HISTORICAL
        and capacity_origin == ORIGIN_HISTORICAL
        and noise_origin == ORIGIN_HISTORICAL
    ):
        errors.append("grid smoke cannot carry the full inference grid")


def _object(payload, key, errors):
    value = payload.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key}: expected an object")
        return None
    return value


def _reject_unknown(payload, allowed, label, errors):
    for key in sorted(set(payload) - set(allowed)):
        errors.append(f"{label}: unknown field {key}")


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


def _one_of(payload, key, allowed, label, errors):
    if payload.get(key) not in allowed:
        errors.append(f"{label}: expected one of {sorted(allowed)}")


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
