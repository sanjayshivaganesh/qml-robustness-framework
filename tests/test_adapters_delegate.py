"""Adapter identity, delegation, return shape, and unsupported operations.

Variational training is mocked. These tests do not call an experiment main
and do not write under results/.
"""

import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmark.adapters import ALL
from benchmark.adapters import hamming_nearest
from benchmark.adapters import hopfield_async_random
from benchmark.adapters import hopfield_async_sequential
from benchmark.adapters import hopfield_synchronous
from benchmark.adapters import qam_paulix
from benchmark.adapters import qam_rot_fullbatch
from benchmark.adapters import qam_rot_minibatch
from benchmark.adapters import qam_ry_linear
from benchmark.identity import UnsupportedOperation, call
from src.hopfield import Hopfield, HopfieldAsyncRandom
from src.hopfield import hopfield_async_sequential_recall, hopfield_hebbian_weights
from src.paths import PROJECT_ROOT
from src.qam import (
    build_ry_linear_circuit,
    nearest_neighbor_recall,
    qam_recall,
    train_ry_linear_qam,
    train_ry_linear_qam_record_loss,
)


def test_eight_procedures_have_distinct_identities():
    ids = [adapter.identity.procedure_id for adapter in ALL]
    assert ids == [
        "hopfield_synchronous",
        "hopfield_async_sequential",
        "hopfield_async_random",
        "qam_ry_linear",
        "qam_rot_fullbatch",
        "qam_rot_minibatch",
        "qam_paulix",
        "hamming_nearest",
    ]
    assert len(set(ids)) == 8

    families = {adapter.identity.procedure_id: adapter.identity.family for adapter in ALL}
    for procedure_id in (
        "hopfield_synchronous",
        "hopfield_async_sequential",
        "hopfield_async_random",
    ):
        assert families[procedure_id] == "hopfield"
    for procedure_id in (
        "qam_ry_linear",
        "qam_rot_fullbatch",
        "qam_rot_minibatch",
        "qam_paulix",
    ):
        assert families[procedure_id] == "variational_qam"
    assert families["hamming_nearest"] == "hamming_nearest"


def test_describe_is_json_and_names_the_source():
    expected_sources = {
        "hopfield_synchronous": "src.hopfield.Hopfield",
        "hopfield_async_sequential": "src.hopfield.hopfield_async_sequential_recall",
        "hopfield_async_random": "src.hopfield.HopfieldAsyncRandom",
        "qam_ry_linear": "src.qam.train_ry_linear_qam",
        "qam_rot_fullbatch": "experiments.noise_robustness_sweep.QAM",
        "qam_rot_minibatch": "experiments.scaling.QAM",
        "qam_paulix": "experiments.statistical_significance.qam_recall",
        "hamming_nearest": "src.qam.nearest_neighbor_recall",
    }
    for adapter in ALL:
        payload = adapter.describe()
        json.dumps(payload)
        procedure_id = adapter.identity.procedure_id
        assert payload["procedure_id"] == procedure_id
        assert payload["source"] == expected_sources[procedure_id]
        assert payload["capabilities"] == list(adapter.identity.capabilities)
        assert payload["family"] == adapter.identity.family


def test_describe_rejects_non_serializable_hyperparameters():
    with pytest.raises(TypeError):
        hopfield_synchronous.describe(bad=np.array([1, -1]))


def test_only_rot_fullbatch_reseeds_on_construction():
    reseeding = [
        adapter.identity.procedure_id
        for adapter in ALL
        if adapter.rng_contract.construction_reseeds
    ]
    drawing = sorted(
        adapter.identity.procedure_id
        for adapter in ALL
        if adapter.rng_contract.construction_draws
    )
    assert reseeding == ["qam_rot_fullbatch"]
    assert qam_rot_fullbatch.rng_contract.reseed_value == 42
    assert drawing == ["qam_rot_fullbatch", "qam_rot_minibatch"]


def test_adapter_package_does_not_copy_circuits_or_write_results():
    root = PROJECT_ROOT / "benchmark"
    for path in root.rglob("*.py"):
        text = path.read_text()
        assert "AdamOptimizer" not in text
        assert "qml." not in text
        assert "def loss(" not in text
        assert "results_subdir" not in text
        assert "to_csv" not in text


def test_importing_adapters_does_not_touch_archived_results():
    script = r"""
from pathlib import Path
root = Path("results")

def snapshot():
    if not root.exists():
        return {}
    return {
        p.relative_to(root).as_posix(): p.stat().st_mtime_ns
        for p in root.rglob("*")
        if p.is_file()
    }

before = snapshot()
import benchmark.adapters
after = snapshot()
assert before == after
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_unsupported_operations_name_the_procedure():
    with pytest.raises(UnsupportedOperation, match="hamming_nearest does not support train"):
        call(hamming_nearest, "train")
    with pytest.raises(
        UnsupportedOperation, match="hopfield_async_sequential does not support train"
    ):
        call(hopfield_async_sequential, "train")
    with pytest.raises(
        UnsupportedOperation, match="hopfield_synchronous does not support store"
    ):
        call(hopfield_synchronous, "store")
    with pytest.raises(UnsupportedOperation, match="qam_ry_linear does not support store"):
        call(qam_ry_linear, "store")
    assert not hasattr(hamming_nearest, "train")
    assert not hasattr(hopfield_async_sequential, "train")


def test_supported_call_with_missing_arguments_raises_type_error():
    with pytest.raises(TypeError):
        call(hopfield_synchronous, "build")


def test_synchronous_hopfield_delegates_and_returns_timed_pair():
    patterns = np.array([[1.0, -1.0]])
    direct = Hopfield(2)
    direct.train(patterns)
    via = call(hopfield_synchronous, "build", 2)
    assert isinstance(via, Hopfield)
    assert via.steps == 10
    assert hopfield_synchronous.build(2, steps=3).steps == 3
    elapsed = hopfield_synchronous.train(via, patterns)
    assert isinstance(elapsed, float)
    assert np.array_equal(via.W, direct.W)
    recalled = hopfield_synchronous.recall(via, patterns[0])
    assert np.array_equal(recalled, direct.recall(patterns[0]))
    timed = hopfield_synchronous.recall_timed(via, patterns[0])
    assert len(timed) == 2
    assert np.array_equal(timed[0], recalled)
    assert isinstance(timed[1], float)
    info = hopfield_synchronous.describe()["hyperparameters"]
    assert info["default_steps"] == 10
    assert info["recall_timed_divisor"] == "n_bits"


def test_sequential_hopfield_returns_energy_triple_and_is_not_synchronous():
    patterns = np.array([[1.0, -1.0], [-1.0, -1.0]])
    query = np.array([1.0, 1.0])
    weights = hopfield_async_sequential.store(patterns)
    assert np.array_equal(weights, hopfield_hebbian_weights(patterns))
    direct = hopfield_async_sequential_recall(weights, query, steps=2)
    via = hopfield_async_sequential.recall(weights, query, steps=2)
    assert len(via) == 3
    assert np.array_equal(via[0], direct[0])
    assert float(via[1]) == float(direct[1])
    assert float(via[2]) == float(direct[2])
    assert hopfield_async_sequential.describe()["hyperparameters"]["default_steps"] == 20
    assert hopfield_async_sequential.store_implementation is hopfield_hebbian_weights
    assert hopfield_async_sequential.recall_implementation is hopfield_async_sequential_recall


def test_random_order_hopfield_is_a_separate_class_without_an_rng_argument():
    state = np.random.get_state()
    try:
        patterns = np.array([[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]])
        model = hopfield_async_random.build(4)
        assert isinstance(model, HopfieldAsyncRandom)
        assert not isinstance(model, Hopfield)
        assert model.steps == 10
        assert hopfield_async_random.train(model, patterns) is None
        recalled = hopfield_async_random.recall(model, patterns[0].copy(), steps=1)
        values = np.asarray(recalled, dtype=float)
        assert values.shape == (4,)
        assert set(np.unique(values)).issubset({-1.0, 1.0})
    finally:
        np.random.set_state(state)

    for operation in ("build", "train", "recall"):
        assert "rng" not in inspect.signature(getattr(hopfield_async_random, operation)).parameters
    info = hopfield_async_random.describe()["hyperparameters"]
    assert info["default_steps"] == 10
    assert info["accepts_rng_argument"] is False
    assert hopfield_async_random.implementation is HopfieldAsyncRandom


def test_hamming_nearest_returns_the_stored_row_and_is_not_variational():
    patterns = np.array([[1, -1, 1], [-1, -1, -1]])
    exact = np.array([-1, -1, -1])
    exact_hit = hamming_nearest.recall(patterns, exact)
    assert np.array_equal(exact_hit, patterns[1])
    assert np.shares_memory(exact_hit, patterns)
    tie = np.array([1, -1, -1])
    # Distance 1 to both rows. argmin keeps the first row.
    tie_hit = hamming_nearest.recall(patterns, tie)
    assert np.array_equal(tie_hit, patterns[0])
    assert np.shares_memory(tie_hit, patterns)
    assert hamming_nearest.implementation is nearest_neighbor_recall
    info = hamming_nearest.describe()["hyperparameters"]
    assert info["variational"] is False
    assert info["csv_model_label_in_inference_archive"] == "qam"
    assert hamming_nearest.identity.family != "variational_qam"
    with pytest.raises(np.exceptions.AxisError):
        hamming_nearest.recall(np.array([1, -1]), np.array([1, -1]))


def test_ry_linear_signatures_match_src_and_depth_recall_is_not_wrapped():
    source = inspect.signature(train_ry_linear_qam)
    wrapped = inspect.signature(qam_ry_linear.train)
    assert list(source.parameters) == list(wrapped.parameters)
    for name, parameter in source.parameters.items():
        assert wrapped.parameters[name].default == parameter.default
    assert inspect.signature(qam_ry_linear.recall).parameters.keys() == inspect.signature(
        qam_recall
    ).parameters.keys()
    assert inspect.signature(qam_ry_linear.build_circuit).parameters[
        "return_tuple"
    ].default == inspect.signature(build_ry_linear_circuit).parameters["return_tuple"].default
    assert inspect.signature(qam_ry_linear.train_recording_loss).parameters[
        "steps"
    ].default == inspect.signature(train_ry_linear_qam_record_loss).parameters["steps"].default
    assert qam_ry_linear.train_implementation is train_ry_linear_qam
    assert qam_ry_linear.recall_implementation is qam_recall
    assert qam_ry_linear.describe()["hyperparameters"]["circuit_depth_recall"] == "not_wrapped"
    depth_text = (PROJECT_ROOT / "experiments" / "circuit_depth.py").read_text()
    adapter_text = Path(qam_ry_linear.__file__).read_text()
    assert "out[out == 0] = 1" in depth_text
    assert "out[out == 0] = 1" not in adapter_text


def test_ry_linear_forwards_return_tuple_and_recall_result(monkeypatch):
    sentinel = ("params", "circuit", 0.2)

    def fake_train(patterns, n_bits, depth, steps=100, lr=0.05, return_tuple=True):
        fake_train.return_tuple = return_tuple
        fake_train.steps = steps
        return sentinel

    monkeypatch.setattr(qam_ry_linear, "train_ry_linear_qam", fake_train)
    assert qam_ry_linear.train([[1, -1]], 2, 1, return_tuple=False) is sentinel
    assert fake_train.return_tuple is False

    history = ("params", "circuit", [1.0, 0.5])

    def fake_record(patterns, n_bits, depth, steps=300, lr=0.05):
        return history

    monkeypatch.setattr(qam_ry_linear, "train_ry_linear_qam_record_loss", fake_record)
    assert qam_ry_linear.train_recording_loss([[1, -1]], 2, 1) is history

    recalled = object()
    monkeypatch.setattr(qam_ry_linear, "qam_recall", lambda x, circuit, params: recalled)
    assert qam_ry_linear.recall([1], "circuit", "params") is recalled


def test_rot_fullbatch_constructor_is_the_noise_sweep_class(monkeypatch):
    from experiments.noise_robustness_sweep import QAM

    assert qam_rot_fullbatch.implementation is QAM
    constructor = inspect.getsource(QAM.__init__)
    assert "seed(42)" in constructor
    assert "seed(" not in inspect.getsource(qam_rot_fullbatch.build)
    info = qam_rot_fullbatch.describe()["hyperparameters"]
    assert info["class_default_depth"] == 6
    assert info["class_default_train_steps"] == 300
    assert info["class_default_lr"] == 0.05
    assert info["protocol_call"] == "QAM(n_bits)"
    assert info["recall_returns"] == "(pattern, wall_time / n_bits)"
    assert "/len(x)" in inspect.getsource(QAM.recall).replace(" ", "")

    seen = {}
    pair = ("pattern", 0.125)

    class FakeQAM:
        def __init__(self, n_bits, depth=6, train_steps=300, lr=0.05):
            seen.update(
                n_bits=n_bits,
                depth=depth,
                train_steps=train_steps,
                lr=lr,
            )

        def train(self, patterns):
            return 0.25

        def recall(self, x):
            return pair

    seeds = []
    import pennylane

    monkeypatch.setattr(qam_rot_fullbatch, "QAM", FakeQAM)
    monkeypatch.setattr(
        pennylane.numpy.random,
        "seed",
        lambda *args, **kwargs: seeds.append(args),
    )
    model = qam_rot_fullbatch.build(5)
    assert seen == {"n_bits": 5, "depth": 6, "train_steps": 300, "lr": 0.05}
    assert seeds == []
    assert qam_rot_fullbatch.train(model, [[1, -1]]) == 0.25
    assert qam_rot_fullbatch.recall(model, [1, -1]) is pair
    assert qam_rot_fullbatch.implementation is QAM


def test_rot_fullbatch_rejects_unknown_constructor_argument(monkeypatch):
    def fake(n_bits, depth=6, train_steps=300, lr=0.05):
        raise AssertionError("constructor should reject the keyword first")

    monkeypatch.setattr(qam_rot_fullbatch, "QAM", fake)
    with pytest.raises(TypeError):
        qam_rot_fullbatch.build(2, depth_policy=1)


def test_rot_minibatch_uses_scaling_depth_not_the_class_default(monkeypatch):
    from experiments.scaling import QAM

    assert qam_rot_minibatch.implementation is QAM
    info = qam_rot_minibatch.describe()["hyperparameters"]
    assert info["protocol_depth"] == "min(6, n_bits)"
    assert info["class_default_depth"] == 3
    assert info["steps"] == 200
    assert info["lr"] == 0.05
    assert info["recall_returns"] == "signed pattern"
    assert "/len(" not in inspect.getsource(QAM.recall).replace(" ", "")
    main_text = inspect.getsource(
        __import__("experiments.scaling", fromlist=["main"]).main
    )
    assert "depth=min(6, n_bits)" in main_text
    assert "steps=200" in main_text
    assert "lr=0.05" in main_text
    train_text = inspect.getsource(QAM.train)
    assert "1e-4" in train_text
    assert ">= 15" in train_text

    seen = {}

    class FakeQAM:
        def __init__(self, n_bits, depth=3, steps=200, lr=0.05):
            seen["depth"] = depth
            seen["steps"] = steps
            seen["lr"] = lr
            seen["n_bits"] = n_bits

        def train(self, patterns):
            return 1.5

        def recall(self, x):
            return x

    monkeypatch.setattr(qam_rot_minibatch, "QAM", FakeQAM)
    model = qam_rot_minibatch.build(4)
    assert seen["n_bits"] == 4
    assert seen["depth"] == 4
    assert seen["depth"] != 3
    assert seen["steps"] == 200
    assert seen["lr"] == 0.05
    assert qam_rot_minibatch.train(model, [[1, -1, 1, -1]]) == 1.5
    assert np.array_equal(qam_rot_minibatch.recall(model, [1, -1, 1, -1]), [1, -1, 1, -1])
    qam_rot_minibatch.build(8)
    assert seen["depth"] == 6
    qam_rot_minibatch.build(4, depth=2, steps=5, lr=0.01)
    assert seen == {"n_bits": 4, "depth": 2, "steps": 5, "lr": 0.01}
    for operation in ("build", "train", "recall"):
        assert "rng" not in inspect.signature(getattr(qam_rot_minibatch, operation)).parameters


def test_paulix_recall_is_not_the_shared_qam_recall(monkeypatch):
    from experiments import statistical_significance as stats

    assert qam_paulix.recall_implementation is stats.qam_recall
    assert qam_paulix.recall_implementation is not qam_recall
    assert qam_paulix.build_implementation is stats.build_qam
    assert qam_paulix.train_implementation is stats.train_qam
    assert list(inspect.signature(qam_paulix.recall).parameters) == ["params", "circuit", "x"]
    assert list(inspect.signature(qam_recall).parameters) == ["x", "circuit", "params"]
    info = qam_paulix.describe()["hyperparameters"]
    assert info["depth"] == stats.DEPTH == 10
    assert info["epochs"] == stats.EPOCHS == 100
    assert info["lr"] == stats.LR == 0.02
    assert info["distinct_from"] == "src.qam.qam_recall"

    sentinel_circuit = object()
    sentinel_params = object()
    sentinel_state = object()

    monkeypatch.setattr(qam_paulix, "build_qam", lambda n_bits: (sentinel_circuit, n_bits))
    monkeypatch.setattr(
        qam_paulix,
        "train_qam",
        lambda patterns, circuit: (sentinel_params, circuit),
    )
    monkeypatch.setattr(
        qam_paulix,
        "paulix_qam_recall",
        lambda params, circuit, x: sentinel_state,
    )
    built = qam_paulix.build(8)
    assert built == (sentinel_circuit, 8)
    trained = qam_paulix.train(np.ones((1, 8)), built)
    assert trained[0] is sentinel_params
    assert qam_paulix.recall(sentinel_params, sentinel_circuit, [-1, 1]) is sentinel_state
