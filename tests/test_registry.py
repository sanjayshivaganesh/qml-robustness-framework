"""Registry resolution and one configured smoke run. Does not call experiment mains."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from pennylane import numpy as pnp

import experiments.circuit_depth as circuit_depth
import experiments.convergence as convergence
import experiments.noise_robustness_inference as noise_robustness_inference
import experiments.noise_robustness_sweep as noise_robustness_sweep
import experiments.scaling as scaling
import experiments.statistical_significance as statistical_significance
import experiments.storage_capacity as storage_capacity
from benchmark.adapters import ALL
from benchmark.adapters import hamming_nearest
from benchmark.adapters import hopfield_async_random
from benchmark.adapters import hopfield_async_sequential
from benchmark.adapters import hopfield_synchronous
from benchmark.adapters import qam_paulix
from benchmark.adapters import qam_rot_fullbatch
from benchmark.adapters import qam_rot_minibatch
from benchmark.adapters import qam_ry_linear
from benchmark.config_loader import ConfigNotExecutable, load_benchmark_config
from benchmark.registry import (
    InvalidModelParameters,
    ModelHasNoConstructor,
    UnknownModel,
    get,
    prepare,
    registered_ids,
)
from benchmark.runner import run_benchmark
from src.paths import PROJECT_ROOT

EXAMPLE = PROJECT_ROOT / "configs" / "noise_robustness_sweep.smoke.json"

EXPERIMENT_MAINS = (
    circuit_depth,
    convergence,
    noise_robustness_inference,
    noise_robustness_sweep,
    scaling,
    statistical_significance,
    storage_capacity,
)


def _snapshot(root: Path):
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _forbid_mains(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("an experiment main was called")

    for module in EXPERIMENT_MAINS:
        monkeypatch.setattr(module, "main", boom)


def test_every_registered_identifier_resolves_to_its_adapter():
    assert registered_ids() == tuple(adapter.identity.procedure_id for adapter in ALL)
    for adapter in ALL:
        registration = get(adapter.identity.procedure_id)
        assert registration.adapter is adapter
        assert registration.procedure_id == adapter.identity.procedure_id


def test_registered_implementations_are_distinct():
    implementations = [get(procedure_id).implementation for procedure_id in registered_ids()]
    assert len(implementations) == len(set(map(id, implementations)))
    assert get("hopfield_synchronous").implementation is hopfield_synchronous.implementation
    assert get("hopfield_async_sequential").implementation is (
        hopfield_async_sequential.recall_implementation
    )
    assert get("hopfield_async_random").implementation is hopfield_async_random.implementation
    assert get("qam_ry_linear").implementation is qam_ry_linear.train_implementation
    assert get("qam_rot_fullbatch").implementation is qam_rot_fullbatch.implementation
    assert get("qam_rot_minibatch").implementation is qam_rot_minibatch.implementation
    assert get("qam_paulix").implementation is qam_paulix.build_implementation
    assert get("hamming_nearest").implementation is hamming_nearest.implementation
    assert qam_rot_fullbatch.implementation is not qam_rot_minibatch.implementation
    assert hopfield_synchronous.implementation is not hopfield_async_random.implementation


def test_hopfield_parameters_reach_the_original_constructors():
    factory, args, kwargs = prepare(
        "hopfield_synchronous", {"n_bits": 2, "steps": 10}
    )
    assert factory is hopfield_synchronous.build
    assert args == (2,)
    assert kwargs == {}
    assert factory(*args, **kwargs).steps == 10

    factory, args, kwargs = prepare(
        "hopfield_synchronous", {"n_bits": 2, "steps": 4}
    )
    assert kwargs == {"steps": 4}
    assert factory(*args, **kwargs).steps == 4

    factory, args, kwargs = prepare("hopfield_async_random", {"n_bits": 3, "steps": 10})
    assert factory is hopfield_async_random.build
    assert args == (3,)
    assert kwargs == {}
    assert factory(*args, **kwargs).steps == 10

    factory, args, kwargs = prepare("hopfield_async_random", {"n_bits": 3, "steps": 7})
    assert kwargs == {"steps": 7}
    model = factory(*args, **kwargs)
    assert model.steps == 7
    assert type(model) is hopfield_async_random.implementation

    patterns = np.array([[1, -1], [-1, 1]])
    factory, args, kwargs = prepare(
        "hopfield_async_sequential", {"patterns": patterns}
    )
    assert factory is hopfield_async_sequential.store
    assert args == (patterns,)
    assert kwargs == {}
    stored = factory(*args)
    direct = hopfield_async_sequential.store_implementation(patterns)
    assert np.array_equal(stored, direct)


def test_variational_parameters_are_checked_without_construction(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("a variational constructor was called")

    monkeypatch.setattr(qam_rot_fullbatch, "build", boom)
    monkeypatch.setattr(qam_rot_minibatch, "build", boom)
    monkeypatch.setattr(qam_ry_linear, "build_circuit", boom)
    monkeypatch.setattr(qam_paulix, "build", boom)
    monkeypatch.setattr(noise_robustness_sweep.QAM, "__init__", boom)
    monkeypatch.setattr(scaling.QAM, "__init__", boom)

    factory, args, kwargs = prepare(
        "qam_rot_fullbatch",
        {"n_bits": 2, "train_steps": 1, "depth_argument": 6, "lr": 0.05},
    )
    assert factory is qam_rot_fullbatch.build
    assert args == (2,)
    assert kwargs == {"train_steps": 1}
    assert "depth" not in kwargs
    assert "lr" not in kwargs

    factory, args, kwargs = prepare(
        "qam_rot_minibatch",
        {"n_bits": 4, "depth": 4, "steps": 200, "lr": 0.05},
    )
    assert factory is qam_rot_minibatch.build
    assert args == (4,)
    assert kwargs == {"depth": 4, "steps": 200, "lr": 0.05}

    factory, args, kwargs = prepare("qam_ry_linear", {"n_bits": 2, "depth": 1})
    assert factory is qam_ry_linear.build_circuit
    assert args == (2, 1)
    assert kwargs == {}

    factory, args, kwargs = prepare("qam_paulix", {"n_bits": 2})
    assert factory is qam_paulix.build
    assert args == (2,)
    assert kwargs == {}

    with pytest.raises(InvalidModelParameters, match="depth_argument"):
        prepare(
            "qam_rot_fullbatch",
            {"n_bits": 2, "train_steps": 1, "depth_argument": 4, "lr": 0.05},
        )
    with pytest.raises(InvalidModelParameters, match="not coerced"):
        prepare("qam_rot_fullbatch", {"n_bits": 2, "train_steps": 1.0})
    with pytest.raises(InvalidModelParameters, match="unknown parameter"):
        prepare("qam_paulix", {"n_bits": 2, "epochs": 100, "lr": 0.02})
    with pytest.raises(InvalidModelParameters, match="depth"):
        prepare("qam_rot_minibatch", {"n_bits": 4, "steps": 200, "lr": 0.05})


def test_unsupported_identifier_fails_immediately():
    with pytest.raises(UnknownModel, match="unknown model 'qam'") as caught:
        get("qam")
    message = str(caught.value)
    assert "hopfield_synchronous" in message
    assert "qam_rot_fullbatch" in message
    with pytest.raises(UnknownModel, match="unknown model 'qam'"):
        prepare("qam", {"n_bits": 2, "train_steps": 1})
    with pytest.raises(ModelHasNoConstructor, match="recall"):
        prepare("hamming_nearest", {})


def test_non_executable_configuration_does_not_construct(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("QAM was constructed")

    monkeypatch.setattr(noise_robustness_sweep.QAM, "__init__", boom)
    document = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    document["grid"] = "full"
    document["models"][1]["parameters"]["train_steps"] = 300
    document["setting_origin"]["grid"] = "historical"
    document["setting_origin"]["models[1].parameters.train_steps"] = "historical"
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    loaded = load_benchmark_config(path)
    assert loaded.procedure_ids == ("hopfield_synchronous", "qam_rot_fullbatch")
    assert loaded.executable is False
    with pytest.raises(ConfigNotExecutable):
        run_benchmark(loaded, output_dir=tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


def test_configured_smoke_runs_end_to_end(tmp_path, monkeypatch):
    _forbid_mains(monkeypatch)
    archive = _snapshot(PROJECT_ROOT / "results")
    numpy_state = np.random.get_state()
    pennylane_state = pnp.random.get_state()
    output_dir = tmp_path / "smoke"
    try:
        loaded = load_benchmark_config(EXAMPLE)
        assert loaded.procedure_ids == ("hopfield_synchronous", "qam_rot_fullbatch")
        assert loaded.executable is True
        result = run_benchmark(loaded, output_dir=output_dir)
    finally:
        np.random.set_state(numpy_state)
        pnp.random.set_state(pennylane_state)

    assert result["main_invoked"] is False
    assert result["comparable_to_archive"] is False
    assert result["procedure_ids"] == ["hopfield_synchronous", "qam_rot_fullbatch"]
    assert "qam" not in result["procedure_ids"]
    run_dir = Path(result["output_dir"])
    assert run_dir.resolve().is_relative_to(tmp_path.resolve())
    assert run_dir.parent == output_dir
    assert not str(run_dir).startswith(str(PROJECT_ROOT / "runs"))
    assert [arm["procedure_id"] for cell in result["cells"] for arm in cell["arms"]] == [
        "hopfield_synchronous",
        "qam_rot_fullbatch",
        "hopfield_synchronous",
        "qam_rot_fullbatch",
    ]
    with (run_dir / "trials.csv").open(newline="", encoding="utf-8") as handle:
        recorded = [row["procedure_id"] for row in csv.DictReader(handle)]
    assert recorded == [
        "hopfield_synchronous",
        "qam_rot_fullbatch",
        "hopfield_synchronous",
        "qam_rot_fullbatch",
    ]
    with (run_dir / "aggregates.csv").open(newline="", encoding="utf-8") as handle:
        aggregates = {row["procedure_id"] for row in csv.DictReader(handle)}
    assert aggregates == {"hopfield_synchronous", "qam_rot_fullbatch"}
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["procedure_ids"] == ["hopfield_synchronous", "qam_rot_fullbatch"]
    assert manifest["main_invoked"] is False
    assert _snapshot(PROJECT_ROOT / "results") == archive
