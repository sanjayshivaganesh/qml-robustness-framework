"""Load the noise-sweep request file. Does not train or write results/."""

import json
from pathlib import Path

import pytest

from benchmark.config_loader import (
    ConfigNotExecutable,
    ConfigValidationError,
    load_benchmark_config,
)
from benchmark.protocols.noise_robustness_sweep import NoiseSweepRequest
from src.paths import PROJECT_ROOT

EXAMPLE = PROJECT_ROOT / "configs" / "noise_robustness_sweep.smoke.json"


def _write(tmp_path, payload):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _example(tmp_path, mutate):
    document = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    mutate(document)
    return _write(tmp_path, document)


def test_example_configuration_is_executable_and_builds_a_request():
    loaded = load_benchmark_config(EXAMPLE)
    assert loaded.executable is True
    assert loaded.execution_blockers == ()
    assert loaded.protocol == "noise_robustness_sweep"
    assert loaded.n_bits_list == (2,)
    assert loaded.corruption_levels == (0.0,)
    assert loaded.trial_seeds == (0, 1)
    assert loaded.train_steps == 1
    assert loaded.capacity_ratio == 0.3
    request = loaded.to_runner_request()
    assert isinstance(request, NoiseSweepRequest)
    assert request.grid == "smoke"
    assert loaded.resolved_output_dir() == (
        PROJECT_ROOT / "runs" / "noise_robustness_sweep_smoke"
    )
    assert not (PROJECT_ROOT / "runs" / "noise_robustness_sweep_smoke").exists()


def test_loading_does_not_construct_a_circuit(monkeypatch):
    def boom(self, *args, **kwargs):
        raise AssertionError("QAM was constructed")

    monkeypatch.setattr(
        "experiments.noise_robustness_sweep.QAM.__init__",
        boom,
    )
    loaded = load_benchmark_config(EXAMPLE)
    assert loaded.executable is True


def test_missing_required_field(tmp_path):
    path = _example(tmp_path, lambda document: document.pop("experiment"))
    with pytest.raises(ConfigValidationError, match="experiment: required field is missing"):
        load_benchmark_config(path)


def test_invalid_probability_is_not_coerced(tmp_path):
    def mutate(document):
        document["perturbation"]["levels"] = [1.5, -0.1]

    path = _example(tmp_path, mutate)
    with pytest.raises(ConfigValidationError, match="outside the Bernoulli range"):
        load_benchmark_config(path)


def test_integral_float_is_not_accepted_as_an_integer(tmp_path):
    def mutate(document):
        document["models"][1]["parameters"]["train_steps"] = 1.0

    path = _example(tmp_path, mutate)
    with pytest.raises(ConfigValidationError, match="Values are not coerced"):
        load_benchmark_config(path)


def test_unknown_model(tmp_path):
    def mutate(document):
        document["models"][1]["procedure_id"] = "qam"

    path = _example(tmp_path, mutate)
    with pytest.raises(ConfigValidationError, match="not a registered model"):
        load_benchmark_config(path)


def test_unsupported_perturbation(tmp_path):
    def mutate(document):
        document["perturbation"]["procedure_id"] = "fixed_count"

    path = _example(tmp_path, mutate)
    with pytest.raises(ConfigValidationError, match="fixed_count"):
        load_benchmark_config(path)


def test_invalid_metric_name(tmp_path):
    def mutate(document):
        document["metrics"] = ["recovery_gain", "infer_time"]

    path = _example(tmp_path, mutate)
    with pytest.raises(ConfigValidationError, match="recovery_gain"):
        load_benchmark_config(path)


def test_invalid_trial_count(tmp_path):
    def mutate(document):
        document["seeds"]["trial_count"] = 3

    path = _example(tmp_path, mutate)
    with pytest.raises(ConfigValidationError, match="trial_count"):
        load_benchmark_config(path)


def test_malformed_configuration_file(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="malformed JSON"):
        load_benchmark_config(path)


def test_output_safety_settings(tmp_path):
    archived = _example(
        tmp_path,
        lambda document: document["output"].__setitem__(
            "destination", "results/noise_robustness_sweep"
        ),
    )
    with pytest.raises(ConfigValidationError, match="archived results"):
        load_benchmark_config(archived)

    overwrite = _example(
        tmp_path,
        lambda document: document["output"].__setitem__("overwrite_archived_results", True),
    )
    with pytest.raises(ConfigValidationError, match="overwrite_archived_results"):
        load_benchmark_config(overwrite)

    link = tmp_path / "archive_alias"
    link.symlink_to(PROJECT_ROOT / "results", target_is_directory=True)

    def point_at_link(document):
        document["output"]["destination"] = str(link / "noise_robustness_sweep")

    linked = _example(tmp_path, point_at_link)
    with pytest.raises(ConfigValidationError, match="archived results"):
        load_benchmark_config(linked)


def test_historical_budget_is_valid_and_not_executable(tmp_path):
    def mutate(document):
        document["grid"] = "full"
        document["models"][1]["parameters"]["train_steps"] = 300
        document["setting_origin"]["grid"] = "historical"
        document["setting_origin"]["models[1].parameters.train_steps"] = "historical"

    path = _example(tmp_path, mutate)
    loaded = load_benchmark_config(path)
    assert loaded.executable is False
    assert any("train_steps" in blocker for blocker in loaded.execution_blockers)
    assert any("grid" in blocker for blocker in loaded.execution_blockers)
    with pytest.raises(ConfigNotExecutable):
        loaded.to_runner_request()
    assert not (PROJECT_ROOT / "results" / "noise_robustness_sweep" / "summary").joinpath(
        "from_config.csv"
    ).exists()
