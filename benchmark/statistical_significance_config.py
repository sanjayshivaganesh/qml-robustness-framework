"""Load a statistical-significance request. Does not construct a model or train.

The noise-sweep schema stays in ``benchmark.config_loader``. This parser
accepts ``protocol`` ``statistical_significance`` only. A historical
document is valid and not executable until ``allow_historical_grid`` is passed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import experiments.statistical_significance as historical
from benchmark.config_loader import ConfigNotExecutable, ConfigValidationError
from benchmark.output import assert_output_allowed
from benchmark.protocols.statistical_significance import (
    BOOTSTRAP_SEEDED,
    CI_PERCENTILES,
    COMPARISON_GROUPS,
    FRAMEWORK_PROCESS,
    HISTORICAL_PROCESS,
    MAX_SMOKE_CAPACITIES,
    MAX_SMOKE_NOISE_LEVELS,
    MAX_SMOKE_SEEDS,
    MAX_SMOKE_WIDTHS,
    PAIRED_DIFFERENCE,
    PATTERN_GENERATOR,
    PATTERN_RULE,
    PROTOCOL,
    TRAINING_SCHEDULE,
    WILCOXON_SENTINEL,
    StatisticalRequest,
    historical_request,
    is_historical_request,
)
from src.paths import PROJECT_ROOT

SCHEMA_VERSION = 1
DOCUMENT_ROLE = "benchmark_request"
LOADER_STATUS = "not_implemented"
AUDIT_PATH = "baseline/configs/original_experiments.json"
AUDIT_KEY = "statistical_significance"
AUDIT_ROLE = "audit_record_only"
HOPFIELD_ID = "hopfield_synchronous"
QAM_ID = "qam_paulix"
HOPFIELD_SOURCE = "src.hopfield.Hopfield"
QAM_SOURCE = "experiments.statistical_significance.qam_recall"
QAM_BUILD = "experiments.statistical_significance.build_qam"
QAM_TRAIN = "experiments.statistical_significance.train_qam"
OUTPUT_FILES = (
    "manifest.json",
    "configuration.json",
    "trials.csv",
    "trials.json",
    "summary.csv",
    "summary.json",
    "effect_sizes.csv",
    "effect_sizes.json",
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
        "declared_task_count",
        "execution",
        "output",
        "setting_origin",
    }
)


@dataclass(frozen=True)
class StatisticalSignificanceConfig:
    """A schema-valid statistical-significance request.

    ``executable`` is true only for one module cell. The 720-task grid stays
    false until ``to_runner_request(allow_historical_grid=True)``.
    """

    source_path: str
    experiment_id: str
    protocol: str
    procedure_ids: tuple[str, ...]
    grid: str
    n_bits_list: tuple[int, ...]
    capacity_ratios: tuple[float, ...]
    corruption_levels: tuple[float, ...]
    seeds: tuple[int, ...]
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
        request = StatisticalRequest(
            n_bits_list=self.n_bits_list,
            capacity_ratios=self.capacity_ratios,
            corruption_levels=self.corruption_levels,
            seeds=self.seeds,
            bootstrap_samples=self.bootstrap_samples,
            grid=self.grid,
        )
        if allow_historical_grid:
            if not is_historical_request(request):
                raise ConfigNotExecutable(
                    (
                        "allow_historical_grid accepts only the historical "
                        "statistical-significance grid.",
                    )
                )
            return request
        if not self.executable:
            raise ConfigNotExecutable(self.execution_blockers)
        return request


def load_statistical_significance_document(source: Path, payload):
    """Validate ``payload`` and return a config. Does not train."""
    parsed = _parse(payload)
    blockers = _execution_blockers(parsed)
    return StatisticalSignificanceConfig(
        source_path=str(source),
        experiment_id=parsed["experiment_id"],
        protocol=PROTOCOL,
        procedure_ids=(HOPFIELD_ID, QAM_ID),
        grid=parsed["grid"],
        n_bits_list=tuple(parsed["n_bits_list"]),
        capacity_ratios=tuple(parsed["capacity_ratios"]),
        corruption_levels=tuple(parsed["corruption_levels"]),
        seeds=tuple(parsed["seeds"]),
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

    audit = _read_audit(errors)
    n_bits_list = []
    capacity_ratios = []
    if isinstance(dataset, dict):
        n_bits_list, capacity_ratios = _parse_dataset(dataset, errors)
    corruption_levels = []
    if isinstance(perturbation, dict):
        corruption_levels = _parse_perturbation(perturbation, errors)
    _parse_models(payload.get("models"), errors)
    bootstrap_samples = None
    if isinstance(statistics, dict):
        bootstrap_samples = _parse_statistics(statistics, errors)
    seed_values = []
    if isinstance(seeds, dict):
        seed_values = _parse_seeds(seeds, errors)
    if isinstance(execution, dict):
        _parse_execution(execution, errors)
    destination = _parse_output(output, errors) if isinstance(output, dict) else None
    if payload.get("metrics") != ["pattern_accuracy_pair"]:
        errors.append("metrics: expected ['pattern_accuracy_pair']")
    declared = payload.get("declared_task_count")
    if type(declared) is not int or isinstance(declared, bool) or declared < 1:
        errors.append("declared_task_count: expected an integer >= 1")
    elif declared != len(n_bits_list) * len(capacity_ratios) * len(corruption_levels) * len(seed_values):
        errors.append(
            "declared_task_count: must equal "
            "len(n_bits_list) * len(capacity_ratios) * len(corruption_levels) * len(seeds). "
            "Each task trains both models. Noise is not scored from a shared training."
        )
    if audit is not None and isinstance(origins, dict) and bootstrap_samples is not None:
        _check_origins(
            origins,
            n_bits_list=n_bits_list,
            capacity_ratios=capacity_ratios,
            corruption_levels=corruption_levels,
            seeds=seed_values,
            grid=payload.get("grid"),
            errors=errors,
        )
    if errors:
        raise ConfigValidationError(errors)
    return {
        "experiment_id": experiment_id,
        "grid": payload["grid"],
        "n_bits_list": n_bits_list,
        "capacity_ratios": capacity_ratios,
        "corruption_levels": corruption_levels,
        "seeds": seed_values,
        "bootstrap_samples": bootstrap_samples,
        "destination": destination,
    }


def _execution_blockers(parsed):
    blockers = []
    if parsed["grid"] != "smoke":
        blockers.append("grid 'full' is not executable without allow_historical_grid.")
    if parsed["bootstrap_samples"] != int(historical.BOOTSTRAP_SAMPLES):
        blockers.append(
            f"bootstrap_samples must stay {int(historical.BOOTSTRAP_SAMPLES)} and unseeded."
        )
    if len(parsed["n_bits_list"]) > MAX_SMOKE_WIDTHS:
        blockers.append(f"width count exceeds the smoke cap {MAX_SMOKE_WIDTHS}.")
    if len(parsed["capacity_ratios"]) > MAX_SMOKE_CAPACITIES:
        blockers.append(f"capacity count exceeds the smoke cap {MAX_SMOKE_CAPACITIES}.")
    if len(parsed["corruption_levels"]) > MAX_SMOKE_NOISE_LEVELS:
        blockers.append(f"noise count exceeds the smoke cap {MAX_SMOKE_NOISE_LEVELS}.")
    if len(parsed["seeds"]) > MAX_SMOKE_SEEDS:
        blockers.append(f"seed count exceeds the smoke cap {MAX_SMOKE_SEEDS}.")
    full = historical_request()
    task_count = (
        len(parsed["n_bits_list"])
        * len(parsed["capacity_ratios"])
        * len(parsed["corruption_levels"])
        * len(parsed["seeds"])
    )
    if task_count == len(tuple(full.tasks())):
        blockers.append("the historical 720-task grid is not executable without allow_historical_grid.")
    return tuple(blockers)


def _parse_dataset(dataset, errors):
    _reject_unknown(
        dataset,
        {"pattern_generator", "alphabet", "n_bits_list", "capacity_ratios", "n_patterns"},
        "dataset",
        errors,
    )
    if dataset.get("pattern_generator") != PATTERN_GENERATOR:
        errors.append(
            "dataset.pattern_generator: expected 'numpy.random'. "
            "run_trial draws numpy.random.choice after seeding both global generators."
        )
    if dataset.get("alphabet") != [-1, 1]:
        errors.append("dataset.alphabet: expected [-1, 1]")
    rule = dataset.get("n_patterns")
    if not isinstance(rule, dict) or rule.get("rule_id") != PATTERN_RULE:
        errors.append(f"dataset.n_patterns.rule_id: expected {PATTERN_RULE!r}")
    elif set(rule) - {"rule_id"}:
        errors.append("dataset.n_patterns: the count is the rule, not a stored integer")
    n_bits_list = _int_list(dataset.get("n_bits_list"), "dataset.n_bits_list", errors, minimum=1)
    capacity_ratios = _float_list(
        dataset.get("capacity_ratios"), "dataset.capacity_ratios", errors, low=0.0, high=1.0
    )
    return n_bits_list, capacity_ratios


def _parse_perturbation(perturbation, errors):
    _reject_unknown(
        perturbation,
        {"procedure_id", "implementation", "when", "levels"},
        "perturbation",
        errors,
    )
    if perturbation.get("procedure_id") != "bernoulli":
        errors.append("perturbation.procedure_id: expected 'bernoulli'")
    if perturbation.get("implementation") != "src.data_generation.corrupt_bernoulli":
        errors.append(
            "perturbation.implementation: expected 'src.data_generation.corrupt_bernoulli'"
        )
    if perturbation.get("when") != "each_stored_pattern_inside_hopfield_then_inside_qam":
        errors.append(
            "perturbation.when: expected 'each_stored_pattern_inside_hopfield_then_inside_qam'"
        )
    return _float_list(perturbation.get("levels"), "perturbation.levels", errors, low=0.0, high=1.0)


def _parse_models(models, errors):
    if not isinstance(models, list) or len(models) != 2:
        errors.append("models: expected hopfield_synchronous then qam_paulix")
        return
    hopfield, qam = models
    if not isinstance(hopfield, dict) or not isinstance(qam, dict):
        errors.append("models: each entry must be an object")
        return
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
        errors.append("models[0].parameters: expected {\"steps\": 10}")
    _expect_operation(hopfield.get("training"), "models[0].training", errors)
    _expect_recall(hopfield.get("recall"), "models[0].recall", errors, distinct_from=None)

    _reject_unknown(
        qam,
        {
            "procedure_id",
            "source",
            "build",
            "train",
            "parameters",
            "closed_over",
            "constructor_parameters",
            "training",
            "recall",
        },
        "models[1]",
        errors,
    )
    if qam.get("procedure_id") != QAM_ID:
        errors.append(
            "models[1].procedure_id: expected 'qam_paulix'. "
            "qam_ry_linear, qam_rot_fullbatch, and qam_rot_minibatch are different circuits."
        )
    if qam.get("source") != QAM_SOURCE:
        errors.append(f"models[1].source: expected {QAM_SOURCE!r}")
    if qam.get("build") != QAM_BUILD:
        errors.append(f"models[1].build: expected {QAM_BUILD!r}")
    if qam.get("train") != QAM_TRAIN:
        errors.append(f"models[1].train: expected {QAM_TRAIN!r}")
    if qam.get("closed_over") is not True:
        errors.append("models[1].closed_over: depth, epochs, and lr stay the module constants")
    if qam.get("constructor_parameters") != ["n_bits"]:
        errors.append("models[1].constructor_parameters: expected ['n_bits']")
    qam_parameters = qam.get("parameters")
    expected_parameters = {
        "depth": int(historical.DEPTH),
        "epochs": int(historical.EPOCHS),
        "lr": float(historical.LR),
        "init": "normal(0, 0.1)",
        "encoding": "PauliX where the bit is -1",
        "ansatz": "RY + linear CNOT",
    }
    if qam_parameters != expected_parameters:
        errors.append(
            "models[1].parameters: expected depth "
            f"{historical.DEPTH}, epochs {historical.EPOCHS}, lr {historical.LR}, "
            "init 'normal(0, 0.1)', PauliX encoding, and the RY + linear CNOT ansatz. "
            "Those values are closed over by build_qam and train_qam."
        )
    _expect_operation(qam.get("training"), "models[1].training", errors)
    _expect_recall(
        qam.get("recall"),
        "models[1].recall",
        errors,
        distinct_from="src.qam.qam_recall",
    )


def _parse_statistics(statistics, errors):
    _reject_unknown(
        statistics,
        {"comparison_groups", "paired_difference", "bootstrap", "hypothesis_test", "effect_size"},
        "statistics",
        errors,
    )
    if statistics.get("comparison_groups") != list(COMPARISON_GROUPS):
        errors.append(
            "statistics.comparison_groups: expected ['n_bits', 'capacity_ratio', 'noise']"
        )
    if statistics.get("paired_difference") != PAIRED_DIFFERENCE:
        errors.append("statistics.paired_difference: expected 'qam_minus_hopfield'")
    bootstrap = statistics.get("bootstrap")
    samples = None
    if not isinstance(bootstrap, dict):
        errors.append("statistics.bootstrap: expected an object")
    else:
        _reject_unknown(
            bootstrap,
            {"implementation", "samples", "seeded", "percentiles"},
            "statistics.bootstrap",
            errors,
        )
        if bootstrap.get("implementation") != "experiments.statistical_significance.bootstrap_diff":
            errors.append(
                "statistics.bootstrap.implementation: expected "
                "'experiments.statistical_significance.bootstrap_diff'"
            )
        samples = bootstrap.get("samples")
        if type(samples) is not int or isinstance(samples, bool) or samples != int(historical.BOOTSTRAP_SAMPLES):
            errors.append(
                f"statistics.bootstrap.samples: expected {int(historical.BOOTSTRAP_SAMPLES)}"
            )
            samples = None
        if bootstrap.get("seeded") is not BOOTSTRAP_SEEDED:
            errors.append("statistics.bootstrap.seeded: expected false. This protocol does not seed the resample.")
        if bootstrap.get("percentiles") != list(CI_PERCENTILES):
            errors.append("statistics.bootstrap.percentiles: expected [2.5, 97.5]")
    test = statistics.get("hypothesis_test")
    if not isinstance(test, dict):
        errors.append("statistics.hypothesis_test: expected an object")
    else:
        _reject_unknown(
            test,
            {"name", "argument_order", "on_failure", "alpha_decision"},
            "statistics.hypothesis_test",
            errors,
        )
        if test.get("name") != "wilcoxon":
            errors.append("statistics.hypothesis_test.name: expected 'wilcoxon'")
        if test.get("argument_order") != ["qam", "hopfield"]:
            errors.append("statistics.hypothesis_test.argument_order: expected ['qam', 'hopfield']")
        if test.get("on_failure") != WILCOXON_SENTINEL:
            errors.append("statistics.hypothesis_test.on_failure: expected 1.0")
        if "alpha_decision" not in test or test.get("alpha_decision") is not None:
            errors.append(
                "statistics.hypothesis_test.alpha_decision: expected null. "
                "main stores the p-value and does not apply a cutoff."
            )
    effect = statistics.get("effect_size")
    if not isinstance(effect, dict):
        errors.append("statistics.effect_size: expected an object")
    else:
        _reject_unknown(
            effect,
            {"name", "implementation", "x", "y", "magnitude", "in_experiment_main"},
            "statistics.effect_size",
            errors,
        )
        if effect.get("name") != "cliffs_delta":
            errors.append("statistics.effect_size.name: expected 'cliffs_delta'")
        if effect.get("implementation") != "src.metrics.cliffs_delta":
            errors.append("statistics.effect_size.implementation: expected 'src.metrics.cliffs_delta'")
        if effect.get("x") != "hopfield" or effect.get("y") != "qam_paulix":
            errors.append("statistics.effect_size: x is hopfield and y is qam_paulix")
        if effect.get("magnitude") != "src.metrics.cliffs_delta_magnitude":
            errors.append(
                "statistics.effect_size.magnitude: expected 'src.metrics.cliffs_delta_magnitude'"
            )
        if effect.get("in_experiment_main") is not False:
            errors.append(
                "statistics.effect_size.in_experiment_main: expected false. "
                "main does not compute Cliff's delta."
            )
    return samples


def _parse_seeds(seeds, errors):
    _reject_unknown(
        seeds,
        {"values", "generators", "formula", "reused_across_cells"},
        "seeds",
        errors,
    )
    if seeds.get("generators") != ["numpy.random.seed", "pennylane.numpy.random.seed"]:
        errors.append(
            "seeds.generators: expected ['numpy.random.seed', 'pennylane.numpy.random.seed']"
        )
    if seeds.get("formula") != "the task seed itself":
        errors.append("seeds.formula: expected 'the task seed itself'")
    if seeds.get("reused_across_cells") is not True:
        errors.append("seeds.reused_across_cells: expected true")
    return _int_list(seeds.get("values"), "seeds.values", errors, minimum=0)


def _parse_execution(execution, errors):
    _reject_unknown(
        execution,
        {
            "process_model",
            "historical_process_model",
            "max_widths",
            "max_capacities",
            "max_noise_levels",
            "max_seeds",
            "caps_are_authoritative_in",
        },
        "execution",
        errors,
    )
    if execution.get("process_model") != FRAMEWORK_PROCESS:
        errors.append("execution.process_model: expected 'single_process'")
    if execution.get("historical_process_model") != HISTORICAL_PROCESS:
        errors.append("execution.historical_process_model: expected 'multiprocessing_pool'")
    caps = {
        "max_widths": MAX_SMOKE_WIDTHS,
        "max_capacities": MAX_SMOKE_CAPACITIES,
        "max_noise_levels": MAX_SMOKE_NOISE_LEVELS,
        "max_seeds": MAX_SMOKE_SEEDS,
    }
    for key, expected in caps.items():
        if execution.get(key) != expected:
            errors.append(
                f"execution.{key}: expected {expected}. "
                "The cap lives in benchmark.protocols.statistical_significance."
            )
    if execution.get("caps_are_authoritative_in") != "benchmark.protocols.statistical_significance":
        errors.append(
            "execution.caps_are_authoritative_in: expected "
            "'benchmark.protocols.statistical_significance'"
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


def _check_origins(
    origins,
    *,
    n_bits_list,
    capacity_ratios,
    corruption_levels,
    seeds,
    grid,
    errors,
):
    full = historical_request()
    expected = {
        "dataset.n_bits_list": _axis_origin(n_bits_list, full.n_bits_list),
        "dataset.capacity_ratios": _axis_origin(capacity_ratios, full.capacity_ratios),
        "dataset.n_patterns.rule_id": ORIGIN_HISTORICAL,
        "dataset.pattern_generator": ORIGIN_HISTORICAL,
        "models[0].procedure_id": ORIGIN_HISTORICAL,
        "models[0].parameters.steps": ORIGIN_HISTORICAL,
        "models[1].procedure_id": ORIGIN_HISTORICAL,
        "models[1].parameters.depth": ORIGIN_HISTORICAL,
        "models[1].parameters.epochs": ORIGIN_HISTORICAL,
        "models[1].parameters.lr": ORIGIN_HISTORICAL,
        "models[1].parameters.init": ORIGIN_HISTORICAL,
        "perturbation.levels": _axis_origin(corruption_levels, full.corruption_levels),
        "seeds.values": _axis_origin(seeds, full.seeds),
        "statistics.bootstrap.samples": ORIGIN_HISTORICAL,
        "statistics.bootstrap.seeded": ORIGIN_HISTORICAL,
        "statistics.hypothesis_test.on_failure": ORIGIN_HISTORICAL,
        "grid": ORIGIN_HISTORICAL if grid == "full" else ORIGIN_DEMONSTRATION,
        "execution.process_model": ORIGIN_DEMONSTRATION,
        "execution.historical_process_model": ORIGIN_HISTORICAL,
        "output.destination": ORIGIN_DEMONSTRATION,
        "comparable_to_archive": ORIGIN_DEMONSTRATION,
    }
    if set(origins) != set(expected):
        errors.append(
            "setting_origin: expected exactly "
            + ", ".join(sorted(expected))
        )
        return
    for key, label in expected.items():
        if label is None:
            errors.append(
                f"setting_origin.{key}: the value is not the module axis or a subset of it"
            )
        elif origins.get(key) != label:
            errors.append(f"setting_origin.{key}: expected {label!r}")


def _axis_origin(value, full):
    got = tuple(value)
    if got == tuple(full):
        return ORIGIN_HISTORICAL
    if got and all(item in tuple(full) for item in got):
        return ORIGIN_DEMONSTRATION
    return None


def _read_audit(errors):
    path = PROJECT_ROOT / AUDIT_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"historical_reference: cannot read audit ({exc})")
        return None
    record = payload.get("experiments", {}).get(AUDIT_KEY)
    if not isinstance(record, dict):
        errors.append("historical_reference: audit is missing experiments.statistical_significance")
        return None
    if record.get("n_bits_list") != list(historical.N_BITS_LIST):
        errors.append("audit n_bits_list does not match experiments.statistical_significance")
    if [float(value) for value in record.get("capacity_ratios", [])] != [
        float(value) for value in historical.CAPACITY_RATIOS
    ]:
        errors.append("audit capacity_ratios do not match the module")
    if [float(value) for value in record.get("corruption_levels", [])] != [
        float(value) for value in historical.CORRUPTION_LEVELS
    ]:
        errors.append("audit corruption_levels do not match the module")
    if record.get("seeds") != "range(30)" or list(historical.SEEDS) != list(range(30)):
        errors.append("audit seeds and the module seeds are not range(30)")
    qam = record.get("qam") if isinstance(record.get("qam"), dict) else {}
    if qam.get("depth") != int(historical.DEPTH) or qam.get("epochs") != int(historical.EPOCHS):
        errors.append("audit qam depth or epochs do not match the module")
    if float(qam.get("lr", -1)) != float(historical.LR):
        errors.append("audit qam lr does not match the module")
    if record.get("bootstrap_samples") != int(historical.BOOTSTRAP_SAMPLES):
        errors.append("audit bootstrap_samples does not match the module")
    return record


def _expect_operation(value, label, errors):
    if value != {"operation": "train"}:
        errors.append(f"{label}: expected {{'operation': 'train'}}")


def _expect_recall(value, label, errors, *, distinct_from):
    expected = {"operation": "recall", "returns": ["pattern"]}
    if distinct_from is not None:
        expected["distinct_from"] = distinct_from
    if value != expected:
        errors.append(f"{label}: expected {expected}")


def _reject_unknown(obj, allowed, label, errors):
    if not isinstance(obj, dict):
        return
    unknown = sorted(set(obj) - set(allowed))
    for key in unknown:
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


def _float_list(value, label, errors, *, low, high):
    if not isinstance(value, list) or not value:
        errors.append(f"{label}: expected a non-empty list of numbers")
        return []
    parsed = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            errors.append(f"{label}: expected numbers in [{low}, {high}]")
            return []
        number = float(item)
        if number < low or number > high:
            errors.append(f"{label}: expected numbers in [{low}, {high}]")
            return []
        parsed.append(number)
    return parsed
