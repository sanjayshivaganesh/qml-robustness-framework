"""Load a benchmark request JSON file into a validated configuration.

This module does not train a model, build a circuit, or start a run.
A document can be schema-valid and still be outside the smoke runner's
budget. ``executable`` records that distinction. ``to_runner_request``
refuses a document the current runner would not start.

Pydantic is not a project dependency. Checks are explicit so an integer
field stays an integer: ``1.0`` is not accepted as ``1``, and a boolean is
not accepted as a number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from benchmark.adapters import ALL
from benchmark.data.corruption import BERNOULLI, FIXED_COUNT
from benchmark.output import ArchivedResultsRefused, assert_output_allowed
from benchmark.protocols.noise_robustness_sweep import (
    MAX_SMOKE_CELLS,
    MAX_SMOKE_TRAIN_STEPS,
    PROTOCOL,
    historical,
)
from benchmark.protocols.noise_robustness_sweep import (
    NoiseSweepRequest,
    is_historical_noise_sweep_request,
)
from benchmark.scoring import METRICS, NOT_SELECTED, PROTOCOL_METRICS, RUNTIME_QUANTITIES
from src.paths import PROJECT_ROOT

SCHEMA_VERSION = 1
DOCUMENT_ROLE = "benchmark_request"
LOADER_STATUS = "not_implemented"
AUDIT_PATH = "baseline/configs/original_experiments.json"
AUDIT_KEY = "noise_robustness_sweep"
AUDIT_ROLE = "audit_record_only"
GRID_SMOKE = "smoke"
GRID_FULL = "full"
KNOWN_GRIDS = frozenset({GRID_SMOKE, GRID_FULL})
PATTERN_GENERATOR = "numpy.random"
ALPHABET = (-1, 1)
PATTERN_RULE = "max_one_or_int_capacity_ratio_times_n_bits"
DEPTH_RULE = "min_depth_argument_and_n_bits"
SEED_POLICY = "legacy_global_trial_seed_then_qam_constructor_reseed_42"
PERTURBATION_WHEN = "each_stored_pattern_after_that_model_trains"
PROCESS_MODEL = "single_process"
CAPS_MODULE = "benchmark.protocols.noise_robustness_sweep"
RECALL_RETURNS = ("pattern", "wall_time_per_bit")
OUTPUT_FILES = (
    "manifest.json",
    "trials.csv",
    "aggregates.csv",
    "trials.json",
)
HOPFIELD_ID = "hopfield_synchronous"
QAM_ID = "qam_rot_fullbatch"
PROTOCOL_MODELS = (HOPFIELD_ID, QAM_ID)
REGISTERED_MODELS = frozenset(adapter.identity.procedure_id for adapter in ALL)
REGISTERED_SOURCES = {
    adapter.identity.procedure_id: adapter.identity.source for adapter in ALL
}
KNOWN_PERTURBATIONS = frozenset({BERNOULLI, FIXED_COUNT})
PROTOCOL_PERTURBATION = BERNOULLI
ORIGIN_HISTORICAL = "historical"
ORIGIN_DEMONSTRATION = "demonstration"
REQUIRED_ORIGINS = frozenset(
    {
        "dataset.pattern_generator",
        "dataset.alphabet",
        "dataset.n_bits_list",
        "dataset.n_patterns.rule_id",
        "dataset.n_patterns.capacity_ratio",
        "models[0].procedure_id",
        "models[0].parameters.steps",
        "models[1].procedure_id",
        "models[1].parameters.depth_argument",
        "models[1].parameters.depth_used_rule_id",
        "models[1].parameters.train_steps",
        "models[1].parameters.lr",
        "models[1].rng.reseed_value",
        "models[1].rng.seed_policy_id",
        "perturbation.procedure_id",
        "perturbation.levels",
        "metrics",
        "timings",
        "seeds.trial_seeds",
        "seeds.trial_count",
        "grid",
        "execution.process_model",
        "output.destination",
        "comparable_to_archive",
    }
)

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
        "declared_cell_count",
        "execution",
        "output",
        "setting_origin",
    }
)


class ConfigValidationError(ValueError):
    """The document is not a well-formed benchmark request."""

    def __init__(self, errors):
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


class ConfigNotExecutable(RuntimeError):
    """The document is valid and the current runner will not start it."""

    def __init__(self, blockers):
        self.blockers = tuple(blockers)
        super().__init__("\n".join(self.blockers))


@dataclass(frozen=True)
class BenchmarkConfig:
    """A schema-valid noise-sweep request.

    ``executable`` is true only when ``run_benchmark`` can take the same
    widths, levels, seeds, and Adam budget. False does not mean the file
    was malformed.
    """

    source_path: str
    experiment_id: str
    protocol: str
    procedure_ids: tuple[str, ...]
    grid: str
    n_bits_list: tuple[int, ...]
    capacity_ratio: float
    corruption_levels: tuple[float, ...]
    trial_seeds: tuple[int, ...]
    train_steps: int
    output_destination: str | None
    executable: bool
    execution_blockers: tuple[str, ...]

    def resolved_output_dir(self):
        """Repo-relative destination, or None for an in-memory result.

        Does not create the directory.
        """
        if self.output_destination is None:
            return None
        path = Path(self.output_destination)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    def to_runner_request(self, *, allow_historical_grid: bool = False):
        """Build a ``NoiseSweepRequest``. Refuse a non-executable document.

        ``allow_historical_grid`` accepts only the noise-sweep module grid.
        It does not make an arbitrary large request runnable.
        """
        request = NoiseSweepRequest(
            n_bits_list=self.n_bits_list,
            corruption_rates=self.corruption_levels,
            seeds=self.trial_seeds,
            capacity_ratio=self.capacity_ratio,
            train_steps=self.train_steps,
            grid=self.grid,
        )
        if allow_historical_grid:
            if not is_historical_noise_sweep_request(request):
                raise ConfigNotExecutable(
                    (
                        "allow_historical_grid accepts only the historical "
                        "noise-sweep grid recorded by the experiment module.",
                    )
                )
            return request
        if not self.executable:
            raise ConfigNotExecutable(self.execution_blockers)
        return request


def load_benchmark_config(path):
    """Read ``path`` and return a ``BenchmarkConfig``.

    Raises ``ConfigValidationError`` for malformed JSON, missing fields,
    wrong types, unknown identifiers, and output paths inside the archive.
    Does not call a model constructor.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigValidationError([f"{source}: cannot read configuration ({exc.strerror})"]) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(
            [f"{source}: malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"]
        ) from exc
    experiment = payload.get("experiment") if isinstance(payload, dict) else None
    if isinstance(experiment, dict) and experiment.get("protocol") == "scaling":
        from benchmark.scaling_config import load_scaling_document

        return load_scaling_document(source, payload)
    if isinstance(experiment, dict) and experiment.get("protocol") == "statistical_significance":
        from benchmark.statistical_significance_config import (
            load_statistical_significance_document,
        )

        return load_statistical_significance_document(source, payload)
    if isinstance(experiment, dict) and experiment.get("protocol") == "storage_capacity":
        from benchmark.storage_capacity_config import load_storage_capacity_document

        return load_storage_capacity_document(source, payload)
    if isinstance(experiment, dict) and experiment.get("protocol") == "circuit_depth":
        from benchmark.circuit_depth_config import load_circuit_depth_document

        return load_circuit_depth_document(source, payload)
    if isinstance(experiment, dict) and experiment.get("protocol") == "noise_robustness_inference":
        from benchmark.noise_robustness_inference_config import load_inference_document

        return load_inference_document(source, payload)
    parsed = _parse_document(payload)
    blockers = _execution_blockers(parsed)
    return BenchmarkConfig(
        source_path=str(source),
        experiment_id=parsed["experiment_id"],
        protocol=parsed["protocol"],
        procedure_ids=tuple(parsed["procedure_ids"]),
        grid=parsed["grid"],
        n_bits_list=tuple(parsed["n_bits_list"]),
        capacity_ratio=parsed["capacity_ratio"],
        corruption_levels=tuple(parsed["levels"]),
        trial_seeds=tuple(parsed["trial_seeds"]),
        train_steps=parsed["train_steps"],
        output_destination=parsed["destination"],
        executable=not blockers,
        execution_blockers=blockers,
    )


def _parse_document(payload):
    errors = []
    if not isinstance(payload, dict):
        raise ConfigValidationError(["document: expected a JSON object"])
    _reject_unknown(payload, TOP_LEVEL, "document", errors)
    _require(payload, TOP_LEVEL, "document", errors)

    experiment = _object(payload, "experiment", errors)
    reference = _object(payload, "historical_reference", errors)
    dataset = _object(payload, "dataset", errors)
    perturbation = _object(payload, "perturbation", errors)
    seeds = _object(payload, "seeds", errors)
    execution = _object(payload, "execution", errors)
    output = _object(payload, "output", errors)
    origins = _object(payload, "setting_origin", errors)
    models = payload.get("models")

    _exact(payload, "schema_version", SCHEMA_VERSION, "schema_version", errors, integer=True)
    _exact_str(payload, "document_role", DOCUMENT_ROLE, "document_role", errors)
    _exact_str(payload, "loader_status", LOADER_STATUS, "loader_status", errors)
    _one_of(payload, "grid", KNOWN_GRIDS, "grid", errors)
    _exact_bool(payload, "comparable_to_archive", False, "comparable_to_archive", errors)

    experiment_id = ""
    protocol = ""
    if isinstance(experiment, dict):
        _reject_unknown(experiment, {"experiment_id", "protocol"}, "experiment", errors)
        _require(experiment, {"experiment_id", "protocol"}, "experiment", errors)
        experiment_id = _non_empty_str(experiment, "experiment_id", "experiment.experiment_id", errors)
        protocol = _protocol(experiment.get("protocol"), errors)

    if isinstance(reference, dict):
        _reject_unknown(
            reference, {"path", "experiment_key", "role"}, "historical_reference", errors
        )
        _exact_str(reference, "path", AUDIT_PATH, "historical_reference.path", errors)
        _exact_str(reference, "experiment_key", AUDIT_KEY, "historical_reference.experiment_key", errors)
        _exact_str(reference, "role", AUDIT_ROLE, "historical_reference.role", errors)

    audit = _read_audit(errors)
    n_bits_list = []
    capacity_ratio = None
    if isinstance(dataset, dict):
        n_bits_list, capacity_ratio = _parse_dataset(dataset, audit, errors)
    levels = []
    if isinstance(perturbation, dict):
        levels = _parse_perturbation(perturbation, protocol, errors)
    train_steps = None
    procedure_ids = []
    if isinstance(models, list):
        train_steps = _parse_models(models, protocol, errors)
        procedure_ids = [
            model.get("procedure_id")
            for model in models
            if isinstance(model, dict) and isinstance(model.get("procedure_id"), str)
        ]
    elif "models" in payload:
        errors.append("models: expected a list of two model objects")
    metrics = _string_list(payload, "metrics", "metrics", errors)
    timings = _string_list(payload, "timings", "timings", errors)
    if protocol == PROTOCOL and isinstance(metrics, list):
        _check_metrics(metrics, errors)
    if isinstance(timings, list):
        _exact_sequence(timings, ("train_time_seconds", "mean_recall_time_per_bit"), "timings", errors)
    trial_seeds = []
    if isinstance(seeds, dict):
        trial_seeds = _parse_seeds(seeds, errors)
    declared = _integer(payload, "declared_cell_count", "declared_cell_count", errors, minimum=1)
    if isinstance(execution, dict):
        _parse_execution(execution, errors)
    destination = None
    if isinstance(output, dict):
        destination = _parse_output(output, errors)

    parsed = {
        "experiment_id": experiment_id,
        "protocol": protocol,
        "procedure_ids": procedure_ids,
        "grid": payload.get("grid"),
        "n_bits_list": n_bits_list,
        "capacity_ratio": capacity_ratio,
        "levels": levels,
        "trial_seeds": trial_seeds,
        "train_steps": train_steps,
        "destination": destination,
    }
    if declared is not None and n_bits_list is not None and levels is not None and trial_seeds is not None:
        product = len(n_bits_list) * len(levels) * len(trial_seeds)
        if declared != product:
            errors.append(
                "declared_cell_count: "
                f"{declared} does not equal len(n_bits_list) * len(levels) * len(trial_seeds) = {product}"
            )
    if isinstance(origins, dict) and audit is not None and not _has_structural_gap(parsed):
        _check_origins(origins, parsed, audit, errors)
    if errors:
        raise ConfigValidationError(errors)
    return parsed


def _has_structural_gap(parsed):
    return any(
        parsed[key] in (None, [], "")
        for key in (
            "protocol",
            "n_bits_list",
            "capacity_ratio",
            "levels",
            "trial_seeds",
            "train_steps",
        )
    )


def _parse_dataset(dataset, audit, errors):
    allowed = {"pattern_generator", "alphabet", "n_bits_list", "n_patterns"}
    _reject_unknown(dataset, allowed, "dataset", errors)
    _require(dataset, allowed, "dataset", errors)
    _exact_str(dataset, "pattern_generator", PATTERN_GENERATOR, "dataset.pattern_generator", errors)
    alphabet = dataset.get("alphabet")
    if alphabet != list(ALPHABET):
        if "alphabet" in dataset:
            errors.append(
                f"dataset.alphabet: expected [-1, 1], got {alphabet!r}. "
                "Patterns are bipolar. This value is not coerced."
            )
    n_bits_list = _integer_list(dataset, "n_bits_list", "dataset.n_bits_list", errors, minimum=1)
    patterns = _object(dataset, "n_patterns", errors)
    capacity_ratio = None
    if isinstance(patterns, dict):
        _reject_unknown(patterns, {"rule_id", "capacity_ratio"}, "dataset.n_patterns", errors)
        _require(patterns, {"rule_id", "capacity_ratio"}, "dataset.n_patterns", errors)
        rule = patterns.get("rule_id")
        if rule != PATTERN_RULE:
            if "rule_id" in patterns:
                errors.append(
                    "dataset.n_patterns.rule_id: "
                    f"unknown rule {rule!r}. The implemented rule is {PATTERN_RULE!r}. "
                    "Configuration strings are not evaluated."
                )
        capacity_ratio = _number(
            patterns, "capacity_ratio", "dataset.n_patterns.capacity_ratio", errors
        )
        if capacity_ratio is not None and audit is not None and capacity_ratio != audit["capacity_ratio"]:
            errors.append(
                "dataset.n_patterns.capacity_ratio: "
                f"{capacity_ratio!r} is not the noise-sweep value {audit['capacity_ratio']!r}. "
                "The runner does not substitute another ratio."
            )
    return n_bits_list, capacity_ratio


def _parse_perturbation(perturbation, protocol, errors):
    allowed = {"procedure_id", "levels", "when"}
    _reject_unknown(perturbation, allowed, "perturbation", errors)
    _require(perturbation, allowed, "perturbation", errors)
    procedure = perturbation.get("procedure_id")
    if procedure not in KNOWN_PERTURBATIONS:
        if "procedure_id" in perturbation:
            errors.append(
                "perturbation.procedure_id: "
                f"unknown perturbation {procedure!r}. "
                f"Registered perturbations are {sorted(KNOWN_PERTURBATIONS)}."
            )
    elif protocol == PROTOCOL and procedure != PROTOCOL_PERTURBATION:
        errors.append(
            "perturbation.procedure_id: "
            f"{procedure!r} is registered but {PROTOCOL} uses {PROTOCOL_PERTURBATION!r}. "
            "fixed_count flips max(1, int(n * level)), including at level 0."
        )
    _exact_str(perturbation, "when", PERTURBATION_WHEN, "perturbation.when", errors)
    levels = _number_list(perturbation, "levels", "perturbation.levels", errors)
    if isinstance(levels, list):
        for index, level in enumerate(levels):
            if level < 0 or level > 1:
                errors.append(
                    f"perturbation.levels[{index}]: {level!r} is outside the Bernoulli range [0, 1]"
                )
    return levels if isinstance(levels, list) else []


def _parse_models(models, protocol, errors):
    if len(models) != 2:
        errors.append(
            "models: expected [hopfield_synchronous, qam_rot_fullbatch] in that order, "
            f"got {len(models)} model(s)"
        )
        return None
    train_steps = _parse_model(models[0], 0, HOPFIELD_ID, errors)
    qam_steps = _parse_model(models[1], 1, QAM_ID, errors)
    if protocol not in ("", PROTOCOL) and protocol != PROTOCOL:
        return qam_steps
    return qam_steps if train_steps is None or qam_steps is not None else None


def _parse_model(model, index, expected_id, errors):
    prefix = f"models[{index}]"
    if not isinstance(model, dict):
        errors.append(f"{prefix}: expected a model object")
        return None
    procedure = model.get("procedure_id")
    if procedure not in REGISTERED_MODELS:
        errors.append(
            f"{prefix}.procedure_id: {procedure!r} is not a registered model. "
            f"Registered ids are {sorted(REGISTERED_MODELS)}. "
            "'qam' is not a procedure id."
        )
    elif procedure != expected_id:
        errors.append(
            f"{prefix}.procedure_id: {PROTOCOL} requires {expected_id!r} "
            f"in this position, not {procedure!r}"
        )
    source = model.get("source")
    if procedure in REGISTERED_SOURCES and source != REGISTERED_SOURCES[procedure]:
        errors.append(
            f"{prefix}.source: {source!r} does not match {procedure!r}, "
            f"whose source is {REGISTERED_SOURCES[procedure]!r}"
        )
    parameters = model.get("parameters")
    training = model.get("training")
    recall = model.get("recall")
    train_steps = None
    if expected_id == HOPFIELD_ID:
        hopfield_keys = {"procedure_id", "source", "parameters", "training", "recall"}
        _reject_unknown(model, hopfield_keys, prefix, errors)
        _require(model, hopfield_keys, prefix, errors)
        if isinstance(parameters, dict):
            _reject_unknown(parameters, {"steps"}, f"{prefix}.parameters", errors)
            _require(parameters, {"steps"}, f"{prefix}.parameters", errors)
            _exact(parameters, "steps", 10, f"{prefix}.parameters.steps", errors, integer=True)
        elif "parameters" in model:
            errors.append(f"{prefix}.parameters: expected an object with steps")
        if isinstance(model, dict) and "rng" in model:
            errors.append(f"{prefix}.rng: hopfield_synchronous does not reseed")
    else:
        qam_keys = {"procedure_id", "source", "parameters", "rng", "training", "recall"}
        _reject_unknown(model, qam_keys, prefix, errors)
        _require(model, qam_keys, prefix, errors)
        if isinstance(parameters, dict):
            qam_parameter_keys = {"depth_argument", "depth_used_rule_id", "train_steps", "lr"}
            _reject_unknown(parameters, qam_parameter_keys, f"{prefix}.parameters", errors)
            _require(parameters, qam_parameter_keys, f"{prefix}.parameters", errors)
            _exact(
                parameters,
                "depth_argument",
                6,
                f"{prefix}.parameters.depth_argument",
                errors,
                integer=True,
            )
            _exact_str(
                parameters,
                "depth_used_rule_id",
                DEPTH_RULE,
                f"{prefix}.parameters.depth_used_rule_id",
                errors,
            )
            train_steps = _integer(
                parameters,
                "train_steps",
                f"{prefix}.parameters.train_steps",
                errors,
                minimum=1,
            )
            lr = _number(parameters, "lr", f"{prefix}.parameters.lr", errors)
            if lr is not None and lr != 0.05:
                errors.append(
                    f"{prefix}.parameters.lr: {lr!r} is not the class value 0.05. "
                    "The runner does not pass a different learning rate."
                )
        elif "parameters" in model:
            errors.append(f"{prefix}.parameters: expected the qam_rot_fullbatch parameter object")
        rng = model.get("rng")
        if isinstance(rng, dict):
            _reject_unknown(
                rng,
                {"family", "constructor_reseeds", "reseed_value", "seed_policy_id"},
                f"{prefix}.rng",
                errors,
            )
            _exact_str(rng, "family", "legacy_global", f"{prefix}.rng.family", errors)
            _exact_bool(rng, "constructor_reseeds", True, f"{prefix}.rng.constructor_reseeds", errors)
            _exact(rng, "reseed_value", 42, f"{prefix}.rng.reseed_value", errors, integer=True)
            _exact_str(rng, "seed_policy_id", SEED_POLICY, f"{prefix}.rng.seed_policy_id", errors)
        elif "rng" in model:
            errors.append(f"{prefix}.rng: expected the legacy_global reseed object")
    if isinstance(training, dict):
        _exact_str(training, "operation", "train", f"{prefix}.training.operation", errors)
    elif "training" in model:
        errors.append(f"{prefix}.training: expected an object")
    if isinstance(recall, dict):
        operation = "recall_timed" if expected_id == HOPFIELD_ID else "recall"
        _exact_str(recall, "operation", operation, f"{prefix}.recall.operation", errors)
        returns = recall.get("returns")
        if returns != list(RECALL_RETURNS):
            errors.append(
                f"{prefix}.recall.returns: expected {list(RECALL_RETURNS)!r}, got {returns!r}"
            )
    elif "recall" in model:
        errors.append(f"{prefix}.recall: expected an object")
    return train_steps


def _check_metrics(metrics, errors):
    allowed = PROTOCOL_METRICS[PROTOCOL]
    for name in metrics:
        if name in RUNTIME_QUANTITIES or name in {
            "train_time_seconds",
            "mean_recall_time_per_bit",
        }:
            errors.append(
                f"metrics: {name!r} is a timing, not a metric for {PROTOCOL}"
            )
        elif name in NOT_SELECTED:
            errors.append(f"metrics: {name!r} is not selectable. {NOT_SELECTED[name]}")
        elif name not in METRICS:
            errors.append(
                f"metrics: unknown metric {name!r}. "
                f"{PROTOCOL} accepts {sorted(allowed)}."
            )
        elif name not in allowed:
            errors.append(
                f"metrics: {name!r} is registered but {PROTOCOL} does not report it. "
                f"This protocol accepts {sorted(allowed)}."
            )
    if set(metrics) != set(allowed):
        missing = sorted(set(allowed) - set(metrics))
        if missing:
            errors.append(f"metrics: missing {missing} required by {PROTOCOL}")


def _parse_seeds(seeds, errors):
    allowed = {"trial_seeds", "trial_count", "seed_policy_id"}
    _reject_unknown(seeds, allowed, "seeds", errors)
    _require(seeds, allowed, "seeds", errors)
    trial_seeds = _integer_list(seeds, "trial_seeds", "seeds.trial_seeds", errors, minimum=None)
    count = _integer(seeds, "trial_count", "seeds.trial_count", errors, minimum=1)
    _exact_str(seeds, "seed_policy_id", SEED_POLICY, "seeds.seed_policy_id", errors)
    if isinstance(trial_seeds, list) and len(trial_seeds) != len(set(trial_seeds)):
        errors.append("seeds.trial_seeds: duplicate seeds are not distinct trials")
    if isinstance(trial_seeds, list) and count is not None and count != len(trial_seeds):
        errors.append(
            "seeds.trial_count: "
            f"{count} does not equal len(trial_seeds) = {len(trial_seeds)}. "
            "trial_count is not a separate range expression."
        )
    return trial_seeds if isinstance(trial_seeds, list) else []


def _parse_execution(execution, errors):
    allowed = {"process_model", "max_cells", "max_train_steps", "caps_are_authoritative_in"}
    _reject_unknown(execution, allowed, "execution", errors)
    _require(execution, allowed, "execution", errors)
    _exact_str(execution, "process_model", PROCESS_MODEL, "execution.process_model", errors)
    _exact(execution, "max_cells", MAX_SMOKE_CELLS, "execution.max_cells", errors, integer=True)
    _exact(
        execution,
        "max_train_steps",
        MAX_SMOKE_TRAIN_STEPS,
        "execution.max_train_steps",
        errors,
        integer=True,
    )
    _exact_str(
        execution,
        "caps_are_authoritative_in",
        CAPS_MODULE,
        "execution.caps_are_authoritative_in",
        errors,
    )


def _parse_output(output, errors):
    allowed = {
        "destination",
        "overwrite_archived_results",
        "if_destination_non_empty",
        "files",
    }
    _reject_unknown(output, allowed, "output", errors)
    _require(output, allowed, "output", errors)
    _exact_bool(
        output,
        "overwrite_archived_results",
        False,
        "output.overwrite_archived_results",
        errors,
    )
    _exact_str(output, "if_destination_non_empty", "refuse", "output.if_destination_non_empty", errors)
    files = output.get("files")
    if files != list(OUTPUT_FILES):
        if "files" in output:
            errors.append(
                "output.files: expected "
                f"{list(OUTPUT_FILES)!r} with the manifest first, got {files!r}"
            )
    destination = output.get("destination", "missing")
    if destination is None:
        return None
    if not isinstance(destination, str) or not destination:
        if "destination" in output:
            errors.append(
                "output.destination: expected a path string or null, "
                f"got {destination!r}"
            )
        return None
    path = Path(destination)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        assert_output_allowed(path)
    except ArchivedResultsRefused as exc:
        errors.append(
            "output.destination: "
            f"{destination!r} resolves inside or onto the archived results tree ({exc}). "
            "overwrite_archived_results cannot allow that path."
        )
    return destination


def _check_origins(origins, parsed, audit, errors):
    _reject_unknown(origins, REQUIRED_ORIGINS, "setting_origin", errors)
    missing = REQUIRED_ORIGINS - set(origins)
    for path in sorted(missing):
        errors.append(f"setting_origin: missing {path}")
    for path, label in origins.items():
        if path not in REQUIRED_ORIGINS:
            continue
        if label not in {ORIGIN_HISTORICAL, ORIGIN_DEMONSTRATION}:
            errors.append(
                f"setting_origin.{path}: expected {ORIGIN_HISTORICAL!r} or "
                f"{ORIGIN_DEMONSTRATION!r}, got {label!r}"
            )
    expected = {
        "dataset.pattern_generator": ORIGIN_HISTORICAL,
        "dataset.alphabet": ORIGIN_HISTORICAL,
        "dataset.n_patterns.rule_id": ORIGIN_HISTORICAL,
        "dataset.n_patterns.capacity_ratio": ORIGIN_HISTORICAL,
        "models[0].procedure_id": ORIGIN_HISTORICAL,
        "models[0].parameters.steps": ORIGIN_HISTORICAL,
        "models[1].procedure_id": ORIGIN_HISTORICAL,
        "models[1].parameters.depth_argument": ORIGIN_HISTORICAL,
        "models[1].parameters.depth_used_rule_id": ORIGIN_HISTORICAL,
        "models[1].parameters.lr": ORIGIN_HISTORICAL,
        "models[1].rng.reseed_value": ORIGIN_HISTORICAL,
        "models[1].rng.seed_policy_id": ORIGIN_HISTORICAL,
        "perturbation.procedure_id": ORIGIN_HISTORICAL,
        "metrics": ORIGIN_HISTORICAL,
        "timings": ORIGIN_HISTORICAL,
        "execution.process_model": ORIGIN_DEMONSTRATION,
        "output.destination": ORIGIN_DEMONSTRATION,
        "comparable_to_archive": ORIGIN_DEMONSTRATION,
        "dataset.n_bits_list": _origin_for_match(parsed["n_bits_list"], list(audit["n_bits_list"])),
        "perturbation.levels": _origin_for_match(parsed["levels"], list(audit["corruption_rates"])),
        "seeds.trial_seeds": _origin_for_match(
            list(parsed["trial_seeds"]), list(range(int(audit["trials"])))
        ),
        "seeds.trial_count": _origin_for_match(len(parsed["trial_seeds"]), int(audit["trials"])),
        "models[1].parameters.train_steps": _origin_for_match(
            parsed["train_steps"], int(audit["qam"]["train_steps"])
        ),
        "grid": ORIGIN_HISTORICAL if parsed["grid"] == GRID_FULL else ORIGIN_DEMONSTRATION,
    }
    for path, want in expected.items():
        got = origins.get(path)
        if got in {ORIGIN_HISTORICAL, ORIGIN_DEMONSTRATION} and got != want:
            errors.append(
                f"setting_origin.{path}: {got!r} does not match the value's relationship "
                f"to the audit record; expected {want!r}"
            )


def _origin_for_match(value, audit_value):
    return ORIGIN_HISTORICAL if value == audit_value else ORIGIN_DEMONSTRATION


def _execution_blockers(parsed):
    blockers = []
    if parsed["grid"] != GRID_SMOKE:
        blockers.append(
            f"grid {parsed['grid']!r} is not executable. "
            "The current runner only starts grid 'smoke'."
        )
    steps = parsed["train_steps"]
    if steps is None or steps < 1 or steps > MAX_SMOKE_TRAIN_STEPS:
        blockers.append(
            f"train_steps {steps!r} is outside the executable range "
            f"1..{MAX_SMOKE_TRAIN_STEPS}. The historical audit value is "
            f"{historical.train_steps} and is not started by this runner."
        )
    cell_count = len(parsed["n_bits_list"]) * len(parsed["levels"]) * len(parsed["trial_seeds"])
    historical_cells = (
        len(historical.n_bits_list) * len(historical.corruption_rates) * int(historical.trials)
    )
    if cell_count > MAX_SMOKE_CELLS:
        blockers.append(
            f"cell count {cell_count} exceeds the executable cap {MAX_SMOKE_CELLS}. "
            "The cap is the protocol constant, not execution.max_cells in the file."
        )
    if cell_count == historical_cells:
        blockers.append(
            f"cell count {cell_count} is the historical {AUDIT_KEY} grid and is not started."
        )
    return tuple(blockers)


def _read_audit(errors):
    path = PROJECT_ROOT / AUDIT_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"historical_reference: cannot read {AUDIT_PATH} ({exc})")
        return None
    try:
        experiment = payload["experiments"][AUDIT_KEY]
        experiment["capacity_ratio"]
        experiment["n_bits_list"]
        experiment["corruption_rates"]
        experiment["trials"]
        experiment["hopfield"]["steps"]
        experiment["qam"]["train_steps"]
        experiment["qam"]["depth_argument"]
        experiment["qam"]["lr"]
    except (KeyError, TypeError) as exc:
        errors.append(f"historical_reference: {AUDIT_PATH} is missing {exc}")
        return None
    if experiment["capacity_ratio"] != historical.capacity_ratio:
        errors.append(
            "historical_reference: audit capacity_ratio does not match the experiment module"
        )
    if int(experiment["qam"]["train_steps"]) != int(historical.train_steps):
        errors.append(
            "historical_reference: audit train_steps does not match the experiment module"
        )
    return experiment


def _protocol(value, errors):
    if value in PROTOCOL_METRICS:
        if value != PROTOCOL:
            errors.append(
                f"experiment.protocol: {value!r} is a known protocol with no request schema. "
                f"This loader validates {PROTOCOL!r} only."
            )
            return ""
        return value
    if value is not None:
        errors.append(
            f"experiment.protocol: unsupported protocol {value!r}. "
            f"Known protocols are {sorted(PROTOCOL_METRICS)}."
        )
    return ""


def _object(payload, key, errors):
    if not isinstance(payload, dict) or key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, dict):
        errors.append(f"{key}: expected a JSON object, got {type(value).__name__}")
        return None
    return value


def _require(mapping, keys, prefix, errors):
    for key in sorted(keys - set(mapping)):
        location = key if prefix == "document" else f"{prefix}.{key}"
        errors.append(f"{location}: required field is missing")


def _reject_unknown(mapping, allowed, prefix, errors):
    if not isinstance(mapping, dict):
        return
    for key in sorted(set(mapping) - set(allowed)):
        location = key if prefix == "document" else f"{prefix}.{key}"
        errors.append(f"{location}: unknown field")


def _exact(mapping, key, expected, path, errors, *, integer):
    if not isinstance(mapping, dict) or key not in mapping:
        return
    value = mapping[key]
    if integer and not _is_int(value):
        errors.append(f"{path}: expected integer {expected!r}, got {value!r}. Values are not coerced.")
        return
    if value != expected:
        errors.append(f"{path}: expected {expected!r}, got {value!r}")


def _exact_str(mapping, key, expected, path, errors):
    if not isinstance(mapping, dict) or key not in mapping:
        return
    value = mapping[key]
    if value != expected:
        errors.append(f"{path}: expected {expected!r}, got {value!r}")


def _exact_bool(mapping, key, expected, path, errors):
    if not isinstance(mapping, dict) or key not in mapping:
        return
    value = mapping[key]
    if not isinstance(value, bool):
        errors.append(f"{path}: expected boolean {expected!r}, got {value!r}. Values are not coerced.")
        return
    if value != expected:
        errors.append(f"{path}: expected {expected!r}, got {value!r}")


def _one_of(mapping, key, allowed, path, errors):
    if not isinstance(mapping, dict) or key not in mapping:
        return
    value = mapping[key]
    if value not in allowed:
        errors.append(f"{path}: unsupported value {value!r}. Allowed values are {sorted(allowed)}.")


def _non_empty_str(mapping, key, path, errors):
    if not isinstance(mapping, dict) or key not in mapping:
        return ""
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path}: expected a non-empty string, got {value!r}")
        return ""
    return value


def _integer(mapping, key, path, errors, *, minimum):
    if not isinstance(mapping, dict) or key not in mapping:
        return None
    value = mapping[key]
    if not _is_int(value):
        errors.append(f"{path}: expected an integer, got {value!r}. Values are not coerced.")
        return None
    if minimum is not None and value < minimum:
        errors.append(f"{path}: expected an integer >= {minimum}, got {value}")
        return None
    return value


def _number(mapping, key, path, errors):
    if not isinstance(mapping, dict) or key not in mapping:
        return None
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{path}: expected a number, got {value!r}. Values are not coerced.")
        return None
    return value


def _integer_list(mapping, key, path, errors, *, minimum):
    if not isinstance(mapping, dict) or key not in mapping:
        return None
    value = mapping[key]
    if not isinstance(value, list) or not value:
        errors.append(f"{path}: expected a non-empty list of integers, got {value!r}")
        return None
    for index, item in enumerate(value):
        if not _is_int(item):
            errors.append(
                f"{path}[{index}]: expected an integer, got {item!r}. Values are not coerced."
            )
            return None
        if minimum is not None and item < minimum:
            errors.append(f"{path}[{index}]: expected an integer >= {minimum}, got {item}")
            return None
    return value


def _number_list(mapping, key, path, errors):
    if not isinstance(mapping, dict) or key not in mapping:
        return None
    value = mapping[key]
    if not isinstance(value, list) or not value:
        errors.append(f"{path}: expected a non-empty list of numbers, got {value!r}")
        return None
    cleaned = []
    ok = True
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            errors.append(f"{path}[{index}]: expected a number, got {item!r}. Values are not coerced.")
            ok = False
        else:
            cleaned.append(item)
    return cleaned if ok else None


def _string_list(mapping, key, path, errors):
    if not isinstance(mapping, dict) or key not in mapping:
        return None
    value = mapping[key]
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        errors.append(f"{path}: expected a non-empty list of strings, got {value!r}")
        return None
    return value


def _exact_sequence(value, expected, path, errors):
    if list(value) != list(expected):
        errors.append(f"{path}: expected {list(expected)!r}, got {list(value)!r}")


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)
