"""Load a storage-capacity request. Does not construct a model or train.

The noise-sweep schema stays in ``benchmark.config_loader``. This parser
accepts ``protocol`` ``storage_capacity`` only. A historical document is
valid and not executable until ``allow_historical_grid`` is passed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import experiments.storage_capacity as historical
from benchmark.config_loader import ConfigNotExecutable, ConfigValidationError
from benchmark.output import assert_output_allowed
from benchmark.protocols.storage_capacity import (
    CRITICAL_THRESHOLD,
    FRAMEWORK_PROCESS,
    HISTORICAL_PROCESS,
    M_RULE,
    MAX_SMOKE_M_VALUES,
    MAX_SMOKE_WIDTHS,
    N_AT_LEAST_16_M,
    PATTERN_GENERATOR,
    PROTOCOL,
    TRAINING_SCHEDULE,
    WILCOXON_SENTINEL,
    StorageCapacityRequest,
    historical_cell_count,
    historical_request,
    is_historical_request,
    m_values_for,
)
from src.paths import PROJECT_ROOT

SCHEMA_VERSION = 1
DOCUMENT_ROLE = "benchmark_request"
LOADER_STATUS = "not_implemented"
AUDIT_PATH = "baseline/configs/original_experiments.json"
AUDIT_KEY = "storage_capacity"
AUDIT_ROLE = "audit_record_only"
HOPFIELD_ID = "hopfield_synchronous"
QAM_ID = "qam_ry_linear"
HOPFIELD_SOURCE = "src.hopfield.Hopfield"
QAM_SOURCE = "src.qam.train_ry_linear_qam"
QAM_RECALL = "src.qam.qam_recall"
DEPTH_RULE = "min_8_and_n_bits"
OUTPUT_FILES = (
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
        "statistics",
        "seeds",
        "training_schedule",
        "declared_cell_count",
        "declared_qam_trainings",
        "execution",
        "output",
        "setting_origin",
    }
)


@dataclass(frozen=True)
class StorageCapacityConfig:
    """A schema-valid storage-capacity request.

    ``executable`` is true only for one module cell at the module training
    budget. The 33-cell grid stays false until
    ``to_runner_request(allow_historical_grid=True)``.
    """

    source_path: str
    experiment_id: str
    protocol: str
    procedure_ids: tuple[str, ...]
    grid: str
    widths: tuple
    seeds: tuple[int, ...]
    train_steps: int
    lr: float
    recall_noise: float
    bootstrap_samples: int
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
        request = StorageCapacityRequest(
            widths=self.widths,
            seeds=self.seeds,
            train_steps=self.train_steps,
            lr=self.lr,
            recall_noise=self.recall_noise,
            bootstrap_samples=self.bootstrap_samples,
            grid=self.grid,
        )
        if allow_historical_grid:
            if not is_historical_request(request):
                raise ConfigNotExecutable(
                    (
                        "allow_historical_grid accepts only the historical "
                        "storage-capacity grid.",
                    )
                )
            return request
        if not self.executable:
            raise ConfigNotExecutable(self.execution_blockers)
        return request


def load_storage_capacity_document(source: Path, payload):
    """Validate ``payload`` and return a config. Does not train."""
    parsed = _parse(payload)
    blockers = _execution_blockers(parsed)
    return StorageCapacityConfig(
        source_path=str(source),
        experiment_id=parsed["experiment_id"],
        protocol=PROTOCOL,
        procedure_ids=(HOPFIELD_ID, QAM_ID),
        grid=parsed["grid"],
        widths=tuple(parsed["widths"]),
        seeds=tuple(parsed["seeds"]),
        train_steps=parsed["train_steps"],
        lr=parsed["lr"],
        recall_noise=parsed["recall_noise"],
        bootstrap_samples=parsed["bootstrap_samples"],
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
    statistics = _object(payload, "statistics", errors)
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
    widths = _parse_dataset(dataset, errors) if isinstance(dataset, dict) else []
    recall_noise = _parse_perturbation(perturbation, errors) if isinstance(perturbation, dict) else None
    train_steps, lr = _parse_models(payload.get("models"), errors)
    bootstrap_samples = _parse_statistics(statistics, errors) if isinstance(statistics, dict) else None
    seed_values = _parse_seeds(seeds, errors) if isinstance(seeds, dict) else []
    if isinstance(execution, dict):
        _parse_execution(execution, errors)
    destination = _parse_output(output, errors) if isinstance(output, dict) else None
    if payload.get("metrics") != ["batch_recall"]:
        errors.append("metrics: expected ['batch_recall']")
    n_cells = sum(len(m_values) for _n_bits, m_values in widths)
    declared_cells = payload.get("declared_cell_count")
    declared_trainings = payload.get("declared_qam_trainings")
    if type(declared_cells) is not int or isinstance(declared_cells, bool) or declared_cells < 1:
        errors.append("declared_cell_count: expected an integer >= 1")
    elif declared_cells != n_cells:
        errors.append(
            "declared_cell_count: must equal the number of (n_bits, m) pairs. "
            "Seeds repeat inside each cell and are not extra cells."
        )
    if type(declared_trainings) is not int or isinstance(declared_trainings, bool) or declared_trainings < 1:
        errors.append("declared_qam_trainings: expected an integer >= 1")
    elif seed_values and declared_trainings != n_cells * len(seed_values):
        errors.append(
            "declared_qam_trainings: must equal declared cells times len(seeds). "
            "Hopfield trains once per seed in the same cell."
        )
    if isinstance(origins, dict) and bootstrap_samples is not None and recall_noise is not None:
        _check_origins(
            origins,
            widths=widths,
            grid=payload.get("grid"),
            errors=errors,
        )
    if errors:
        raise ConfigValidationError(errors)
    return {
        "experiment_id": experiment_id,
        "grid": payload["grid"],
        "widths": widths,
        "seeds": seed_values,
        "train_steps": train_steps,
        "lr": lr,
        "recall_noise": recall_noise,
        "bootstrap_samples": bootstrap_samples,
        "destination": destination,
    }


def _execution_blockers(parsed):
    blockers = []
    if parsed["grid"] != "smoke":
        blockers.append("grid 'full' is not executable without allow_historical_grid.")
    if parsed["train_steps"] != int(historical.train_steps):
        blockers.append(
            f"train_steps must stay {int(historical.train_steps)}. "
            "A shorter budget changes the RY-linear training."
        )
    if tuple(parsed["seeds"]) != tuple(range(int(historical.trials_per_m))):
        blockers.append(
            "seeds must stay 0..4. Wilcoxon and the bootstrap use that trial count."
        )
    if parsed["bootstrap_samples"] != int(historical.bootstrap_samples):
        blockers.append(
            f"bootstrap_samples must stay {int(historical.bootstrap_samples)} "
            "on the trial stream."
        )
    if len(parsed["widths"]) > MAX_SMOKE_WIDTHS:
        blockers.append(f"width count exceeds the smoke cap {MAX_SMOKE_WIDTHS}.")
    if any(len(m_values) > MAX_SMOKE_M_VALUES for _n, m_values in parsed["widths"]):
        blockers.append(f"m count exceeds the smoke cap {MAX_SMOKE_M_VALUES}.")
    if sum(len(m_values) for _n, m_values in parsed["widths"]) == historical_cell_count():
        blockers.append(
            "the historical 33-cell grid is not executable without allow_historical_grid."
        )
    return tuple(blockers)


def _parse_dataset(dataset, errors):
    _reject_unknown(
        dataset,
        {
            "pattern_generator",
            "alphabet",
            "m_rule_id",
            "reduced_scale_for_large_n",
            "hopfield_max_factor",
            "qam_max_factor",
            "n_at_least_16_m_values",
            "widths",
        },
        "dataset",
        errors,
    )
    if dataset.get("pattern_generator") != PATTERN_GENERATOR:
        errors.append("dataset.pattern_generator: expected 'numpy.random'")
    if dataset.get("alphabet") != [-1, 1]:
        errors.append("dataset.alphabet: expected [-1, 1]")
    if dataset.get("m_rule_id") != M_RULE:
        errors.append(f"dataset.m_rule_id: expected {M_RULE!r}")
    if dataset.get("reduced_scale_for_large_n") is not True:
        errors.append(
            "dataset.reduced_scale_for_large_n: expected true. "
            "That is the module flag for the n>=16 list, not a smoke switch."
        )
    if dataset.get("hopfield_max_factor") != float(historical.hopfield_max_factor):
        errors.append("dataset.hopfield_max_factor: expected 1.0")
    if dataset.get("qam_max_factor") != float(historical.qam_max_factor):
        errors.append("dataset.qam_max_factor: expected 1.0")
    if dataset.get("n_at_least_16_m_values") != list(N_AT_LEAST_16_M):
        errors.append(f"dataset.n_at_least_16_m_values: expected {list(N_AT_LEAST_16_M)}")
    widths = dataset.get("widths")
    parsed = []
    if not isinstance(widths, list) or not widths:
        errors.append("dataset.widths: expected a non-empty list")
        return parsed
    for index, width in enumerate(widths):
        label = f"dataset.widths[{index}]"
        if not isinstance(width, dict):
            errors.append(f"{label}: expected an object")
            continue
        _reject_unknown(width, {"n_bits", "m_values"}, label, errors)
        n_bits = width.get("n_bits")
        if type(n_bits) is not int or isinstance(n_bits, bool) or n_bits < 1:
            errors.append(f"{label}.n_bits: expected an integer >= 1")
            continue
        m_values = _int_list(width.get("m_values"), f"{label}.m_values", errors, minimum=1)
        full = m_values_for(n_bits)
        if m_values and not _is_subsequence(m_values, full):
            errors.append(
                f"{label}.m_values: expected a subsequence of the module list {list(full)}"
            )
        parsed.append((n_bits, tuple(m_values)))
    return parsed


def _parse_perturbation(perturbation, errors):
    _reject_unknown(
        perturbation,
        {"procedure_id", "implementation", "level", "rng", "flip_rule"},
        "perturbation",
        errors,
    )
    if perturbation.get("procedure_id") != "fixed_count":
        errors.append("perturbation.procedure_id: expected 'fixed_count'")
    if perturbation.get("implementation") != "src.data_generation.corrupt_fixed_count":
        errors.append(
            "perturbation.implementation: expected 'src.data_generation.corrupt_fixed_count'"
        )
    if perturbation.get("rng") != "numpy.random":
        errors.append("perturbation.rng: expected 'numpy.random'")
    if perturbation.get("flip_rule") != "max(1, int(n_bits * level))":
        errors.append("perturbation.flip_rule: expected 'max(1, int(n_bits * level))'")
    level = perturbation.get("level")
    if isinstance(level, bool) or not isinstance(level, (int, float)):
        errors.append("perturbation.level: expected 0.15")
        return None
    if float(level) != float(historical.recall_noise):
        errors.append(f"perturbation.level: expected {float(historical.recall_noise)}")
    return float(level)


def _parse_models(models, errors):
    train_steps = None
    lr = None
    if not isinstance(models, list) or len(models) != 2:
        errors.append("models: expected hopfield_synchronous then qam_ry_linear")
        return train_steps, lr
    hopfield, qam = models
    if not isinstance(hopfield, dict) or not isinstance(qam, dict):
        errors.append("models: each entry must be an object")
        return train_steps, lr
    _reject_unknown(
        hopfield,
        {"procedure_id", "source", "parameters", "training", "recall"},
        "models[0]",
        errors,
    )
    if hopfield.get("procedure_id") != HOPFIELD_ID:
        errors.append("models[0].procedure_id: expected 'hopfield_synchronous'")
    if hopfield.get("source") != HOPFIELD_SOURCE:
        errors.append(f"models[0].source: expected {HOPFIELD_SOURCE!r}")
    parameters = hopfield.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("steps") != 10 or set(parameters) != {"steps"}:
        errors.append('models[0].parameters: expected {"steps": 10}')
    if hopfield.get("training") != {"operation": "train"}:
        errors.append("models[0].training: expected {'operation': 'train'}")
    if hopfield.get("recall") != {"operation": "recall", "returns": ["pattern"]}:
        errors.append("models[0].recall: expected pattern recall")

    _reject_unknown(
        qam,
        {"procedure_id", "source", "recall", "parameters", "training"},
        "models[1]",
        errors,
    )
    if qam.get("procedure_id") != QAM_ID:
        errors.append(
            "models[1].procedure_id: expected 'qam_ry_linear'. "
            "qam_paulix, qam_rot_fullbatch, and qam_rot_minibatch are different circuits."
        )
    if qam.get("source") != QAM_SOURCE:
        errors.append(f"models[1].source: expected {QAM_SOURCE!r}")
    if qam.get("recall") != QAM_RECALL:
        errors.append(f"models[1].recall: expected {QAM_RECALL!r}")
    qam_parameters = qam.get("parameters")
    if not isinstance(qam_parameters, dict):
        errors.append("models[1].parameters: expected an object")
        return train_steps, lr
    expected = {
        "depth_rule_id": DEPTH_RULE,
        "train_steps": int(historical.train_steps),
        "lr": float(historical.lr),
        "return_tuple": True,
        "init_scale": 0.01,
        "encoding": "RY(x * pi)",
        "ansatz": "RY + linear CNOT",
    }
    if qam_parameters != expected:
        errors.append(
            "models[1].parameters: expected depth rule "
            f"{DEPTH_RULE}, train_steps {historical.train_steps}, lr {historical.lr}, "
            "return_tuple true, init_scale 0.01, RY(x * pi), and RY + linear CNOT."
        )
    else:
        train_steps = int(qam_parameters["train_steps"])
        lr = float(qam_parameters["lr"])
    if qam.get("training") != {"operation": "train", "return_tuple": True}:
        errors.append("models[1].training: expected train with return_tuple true")
    return train_steps, lr


def _parse_statistics(statistics, errors):
    _reject_unknown(
        statistics,
        {"bootstrap", "hypothesis_test", "critical_capacity", "not_used"},
        "statistics",
        errors,
    )
    bootstrap = statistics.get("bootstrap")
    samples = None
    if not isinstance(bootstrap, dict):
        errors.append("statistics.bootstrap: expected an object")
    else:
        _reject_unknown(
            bootstrap,
            {"implementation", "samples", "confidence", "separate_seed", "rng"},
            "statistics.bootstrap",
            errors,
        )
        if bootstrap.get("implementation") != "experiments.storage_capacity.bootstrap_ci":
            errors.append(
                "statistics.bootstrap.implementation: expected "
                "'experiments.storage_capacity.bootstrap_ci'"
            )
        samples = bootstrap.get("samples")
        if (
            type(samples) is not int
            or isinstance(samples, bool)
            or samples != int(historical.bootstrap_samples)
        ):
            errors.append(
                f"statistics.bootstrap.samples: expected {int(historical.bootstrap_samples)}"
            )
            samples = None
        if bootstrap.get("confidence") != 0.95:
            errors.append("statistics.bootstrap.confidence: expected 0.95")
        if bootstrap.get("separate_seed") is not False:
            errors.append(
                "statistics.bootstrap.separate_seed: expected false. "
                "The resamples continue numpy.random after the last trial."
            )
        if bootstrap.get("rng") != "numpy.random":
            errors.append("statistics.bootstrap.rng: expected 'numpy.random'")
    test = statistics.get("hypothesis_test")
    if not isinstance(test, dict):
        errors.append("statistics.hypothesis_test: expected an object")
    else:
        _reject_unknown(
            test,
            {"name", "argument_order", "on_failure"},
            "statistics.hypothesis_test",
            errors,
        )
        if test.get("name") != "wilcoxon":
            errors.append("statistics.hypothesis_test.name: expected 'wilcoxon'")
        if test.get("argument_order") != ["hopfield", "qam"]:
            errors.append(
                "statistics.hypothesis_test.argument_order: expected ['hopfield', 'qam']"
            )
        if "on_failure" not in test or test.get("on_failure") is not WILCOXON_SENTINEL:
            errors.append(
                "statistics.hypothesis_test.on_failure: expected null. "
                "Storage capacity stores None when Wilcoxon raises."
            )
    critical = statistics.get("critical_capacity")
    if not isinstance(critical, dict):
        errors.append("statistics.critical_capacity: expected an object")
    else:
        _reject_unknown(
            critical,
            {"definition", "threshold", "comparison", "scan"},
            "statistics.critical_capacity",
            errors,
        )
        if critical.get("definition") != "m / n_bits":
            errors.append("statistics.critical_capacity.definition: expected 'm / n_bits'")
        if critical.get("threshold") != CRITICAL_THRESHOLD:
            errors.append("statistics.critical_capacity.threshold: expected 0.75")
        if critical.get("comparison") != "strictly_below":
            errors.append("statistics.critical_capacity.comparison: expected 'strictly_below'")
        if critical.get("scan") != "capacity_ratio ascending":
            errors.append(
                "statistics.critical_capacity.scan: expected 'capacity_ratio ascending'"
            )
    not_used = statistics.get("not_used")
    if not_used != ["experiments.storage_capacity.confidence_interval"]:
        errors.append(
            "statistics.not_used: expected "
            "['experiments.storage_capacity.confidence_interval']"
        )
    return samples


def _parse_seeds(seeds, errors):
    _reject_unknown(
        seeds,
        {"values", "generator", "reapplied_for_every_cell", "formula"},
        "seeds",
        errors,
    )
    if seeds.get("generator") != "numpy.random.seed":
        errors.append("seeds.generator: expected 'numpy.random.seed'")
    if seeds.get("reapplied_for_every_cell") is not True:
        errors.append("seeds.reapplied_for_every_cell: expected true")
    if seeds.get("formula") != "the trial index in range(trials_per_m)":
        errors.append("seeds.formula: expected 'the trial index in range(trials_per_m)'")
    values = _int_list(seeds.get("values"), "seeds.values", errors, minimum=0)
    if values and values != list(range(int(historical.trials_per_m))):
        errors.append(
            f"seeds.values: expected {list(range(int(historical.trials_per_m)))}"
        )
    return values


def _parse_execution(execution, errors):
    _reject_unknown(
        execution,
        {
            "process_model",
            "historical_process_model",
            "max_widths",
            "max_m_values",
            "caps_are_authoritative_in",
        },
        "execution",
        errors,
    )
    if execution.get("process_model") != FRAMEWORK_PROCESS:
        errors.append("execution.process_model: expected 'single_process'")
    if execution.get("historical_process_model") != HISTORICAL_PROCESS:
        errors.append("execution.historical_process_model: expected 'single_process'")
    if execution.get("max_widths") != MAX_SMOKE_WIDTHS:
        errors.append(
            f"execution.max_widths: expected {MAX_SMOKE_WIDTHS}. "
            "The cap lives in benchmark.protocols.storage_capacity."
        )
    if execution.get("max_m_values") != MAX_SMOKE_M_VALUES:
        errors.append(
            f"execution.max_m_values: expected {MAX_SMOKE_M_VALUES}. "
            "The cap lives in benchmark.protocols.storage_capacity."
        )
    if execution.get("caps_are_authoritative_in") != "benchmark.protocols.storage_capacity":
        errors.append(
            "execution.caps_are_authoritative_in: expected "
            "'benchmark.protocols.storage_capacity'"
        )


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
    if not isinstance(destination, str) or not destination:
        errors.append("output.destination: expected a path")
        return None
    path = Path(destination)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    assert_output_allowed(path)
    return destination


def _check_origins(origins, *, widths, grid, errors):
    full_widths = tuple((n_bits, m_values) for n_bits, m_values in historical_request().widths)
    widths_origin = (
        ORIGIN_HISTORICAL if tuple(widths) == full_widths else ORIGIN_DEMONSTRATION
    )
    if widths and any(
        not _is_subsequence(list(m_values), m_values_for(n_bits)) for n_bits, m_values in widths
    ):
        widths_origin = None
    expected = {
        "dataset.widths": widths_origin,
        "dataset.m_rule_id": ORIGIN_HISTORICAL,
        "dataset.reduced_scale_for_large_n": ORIGIN_HISTORICAL,
        "dataset.pattern_generator": ORIGIN_HISTORICAL,
        "models[0].procedure_id": ORIGIN_HISTORICAL,
        "models[0].parameters.steps": ORIGIN_HISTORICAL,
        "models[1].procedure_id": ORIGIN_HISTORICAL,
        "models[1].parameters.depth_rule_id": ORIGIN_HISTORICAL,
        "models[1].parameters.train_steps": ORIGIN_HISTORICAL,
        "models[1].parameters.lr": ORIGIN_HISTORICAL,
        "models[1].parameters.return_tuple": ORIGIN_HISTORICAL,
        "perturbation.level": ORIGIN_HISTORICAL,
        "seeds.values": ORIGIN_HISTORICAL,
        "statistics.bootstrap.samples": ORIGIN_HISTORICAL,
        "statistics.bootstrap.separate_seed": ORIGIN_HISTORICAL,
        "statistics.hypothesis_test.on_failure": ORIGIN_HISTORICAL,
        "statistics.critical_capacity.threshold": ORIGIN_HISTORICAL,
        "grid": ORIGIN_HISTORICAL if grid == "full" else ORIGIN_DEMONSTRATION,
        "execution.process_model": ORIGIN_HISTORICAL,
        "output.destination": ORIGIN_DEMONSTRATION,
        "comparable_to_archive": ORIGIN_DEMONSTRATION,
    }
    if set(origins) != set(expected):
        errors.append("setting_origin: expected exactly " + ", ".join(sorted(expected)))
        return
    for key, label in expected.items():
        if label is None:
            errors.append(f"setting_origin.{key}: the widths are not the module grid or a subset")
        elif origins.get(key) != label:
            errors.append(f"setting_origin.{key}: expected {label!r}")


def _read_audit(errors):
    path = PROJECT_ROOT / AUDIT_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"historical_reference: cannot read audit ({exc})")
        return
    record = payload.get("experiments", {}).get(AUDIT_KEY)
    if not isinstance(record, dict):
        errors.append("historical_reference: audit is missing experiments.storage_capacity")
        return
    if record.get("module") != "experiments.storage_capacity":
        errors.append("audit module is not experiments.storage_capacity")
    if record.get("n_bits_list") != list(historical.n_bits_list):
        errors.append("audit n_bits_list does not match the module")
    if record.get("trials_per_m") != int(historical.trials_per_m):
        errors.append("audit trials_per_m does not match the module")
    if float(record.get("recall_noise", -1)) != float(historical.recall_noise):
        errors.append("audit recall_noise does not match the module")
    if record.get("bootstrap_samples") != int(historical.bootstrap_samples):
        errors.append("audit bootstrap_samples does not match the module")
    qam = record.get("qam") if isinstance(record.get("qam"), dict) else {}
    if qam.get("train_steps") != int(historical.train_steps):
        errors.append("audit qam train_steps does not match the module")
    if float(qam.get("lr", -1)) != float(historical.lr):
        errors.append("audit qam lr does not match the module")
    m_values = record.get("m_values") if isinstance(record.get("m_values"), dict) else {}
    if m_values.get("n_at_least_16") != list(N_AT_LEAST_16_M):
        errors.append("audit n>=16 m list does not match the module branch")


def _is_subsequence(values, full):
    if not values:
        return False
    cursor = 0
    full_list = list(full)
    for value in values:
        if value not in full_list[cursor:]:
            return False
        cursor = full_list.index(value, cursor) + 1
    return True


def _reject_unknown(obj, allowed, label, errors):
    if not isinstance(obj, dict):
        return
    for key in sorted(set(obj) - set(allowed)):
        errors.append(f"{label}: unknown field {key}")


def _object(payload, key, errors):
    value = payload.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key}: expected an object")
        return None
    return value


def _exact(obj, key, expected, label, errors, *, integer=False):
    value = obj.get(key)
    if integer and (type(value) is not int or isinstance(value, bool)):
        errors.append(f"{label}: expected integer {expected}")
        return
    if value != expected:
        errors.append(f"{label}: expected {expected!r}")


def _exact_str(obj, key, expected, label, errors):
    if obj.get(key) != expected:
        errors.append(f"{label}: expected {expected!r}")


def _exact_bool(obj, key, expected, label, errors):
    if obj.get(key) is not expected:
        errors.append(f"{label}: expected {str(expected).lower()}")


def _one_of(obj, key, allowed, label, errors):
    if obj.get(key) not in allowed:
        errors.append(f"{label}: expected one of {sorted(allowed)}")


def _non_empty_str(obj, key, label, errors):
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        errors.append(f"{label}: expected a non-empty string")
        return ""
    return value


def _int_list(value, label, errors, *, minimum):
    if not isinstance(value, list) or not value:
        errors.append(f"{label}: expected a non-empty list of integers")
        return []
    parsed = []
    for item in value:
        if type(item) is not int or isinstance(item, bool) or item < minimum:
            errors.append(f"{label}: expected integers >= {minimum}")
            return []
        parsed.append(item)
    if len(set(parsed)) != len(parsed):
        errors.append(f"{label}: values must be distinct")
    return parsed
