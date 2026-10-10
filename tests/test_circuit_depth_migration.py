"""Circuit-depth migration. Does not start the 21-cell sweep.

The smoke cell is n=8, m=4, depth 1, seeds 0..4, and 100 RY-linear steps
with return_tuple false. Depth 1 is the first module depth. It is not the
seven-depth table, and it is not the storage-capacity rule min(8, n).
"""

import csv
import inspect
import json
import logging
from pathlib import Path

import numpy as np
import pytest
from pennylane import numpy as pnp

import experiments.circuit_depth as historical
from benchmark.adapters import qam_ry_linear
from benchmark.cli import EXIT_NOT_EXECUTABLE, EXIT_OK, main
from benchmark.config_loader import ConfigNotExecutable, ConfigValidationError, load_benchmark_config
from benchmark.log import LOGGER_NAME
from benchmark.output import ArchivedResultsRefused
from benchmark.protocols.circuit_depth import (
    RECALL_IMPLEMENTATION,
    HistoricalGridRefused,
    historical_cases,
    historical_cell_count,
    historical_depths,
    historical_request,
    is_historical_circuit_depth_request,
    lightweight_request,
    parameter_count,
    run_circuit_depth,
    sign_circuit_output,
)
from benchmark.registry import get, prepare
from benchmark.runner import run_benchmark
from src.data_generation import corrupt_fixed_count
from src.metrics import full_and_bitwise_accuracy
from src.paths import PROJECT_ROOT
from src.qam import build_ry_linear_circuit, qam_recall, train_ry_linear_qam

SMOKE = PROJECT_ROOT / "configs" / "circuit_depth.smoke.json"
HISTORICAL = PROJECT_ROOT / "configs" / "circuit_depth.json"
ARCHIVE = PROJECT_ROOT / "results" / "circuit_depth" / "summary" / "depth_scaling.csv"
ARCHIVE_LOG = PROJECT_ROOT / "results" / "circuit_depth" / "depth_scaling.log"


def _states():
    return np.random.get_state(), pnp.random.get_state()


def _restore(states):
    numpy_state, pennylane_state = states
    np.random.set_state(numpy_state)
    pnp.random.set_state(pennylane_state)


def _same_state(left, right):
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def _archive_bytes():
    return ARCHIVE.read_bytes(), ARCHIVE_LOG.read_bytes()


def test_historical_configuration_selects_the_depth_script_circuit():
    config = load_benchmark_config(HISTORICAL)
    audit = json.loads(
        (PROJECT_ROOT / "baseline" / "configs" / "original_experiments.json").read_text(
            encoding="utf-8"
        )
    )["experiments"]["circuit_depth"]
    assert config.protocol == "circuit_depth"
    assert config.procedure_ids == ("qam_ry_linear",)
    assert config.executable is False
    assert config.return_tuple is False
    assert config.cases == historical_cases() == ((8, 4), (12, 8), (16, 12))
    assert config.depths == historical_depths() == tuple(audit["depths"])
    assert 10 in config.depths and 12 in config.depths
    assert config.train_steps == historical.train_steps == audit["train_steps"] == 100
    assert config.lr == historical.lr == audit["lr"] == 0.05
    assert config.recall_noise == historical.recall_noise == 0.15
    assert config.seeds == tuple(range(historical.trials)) == (0, 1, 2, 3, 4)
    assert historical_cell_count() == 3 * 7 == 21
    assert len(config.cases) * len(config.depths) == 21
    assert audit["qam"]["return_tuple"] is False
    assert audit["qam"]["function"] == "src.qam.train_ry_linear_qam"
    assert "qam_recall" not in audit["qam"]["recall"]
    request = historical_request()
    assert is_historical_circuit_depth_request(request)
    assert parameter_count(12, 8) == 96
    assert parameter_count(1, 8) == 8
    adapter = get("qam_ry_linear")
    assert adapter.adapter.train is qam_ry_linear.train
    assert qam_ry_linear.train_implementation is train_ry_linear_qam
    _factory, _args, kwargs = prepare(
        "qam_ry_linear",
        {"n_bits": 8, "depth": 12, "return_tuple": False},
    )
    assert kwargs["return_tuple"] is False
    assert "qam_recall(" not in inspect.getsource(sign_circuit_output)
    assert "return_tuple=False" in inspect.getsource(run_circuit_depth.__globals__["_run_cell"])
    assert ".recall(" not in inspect.getsource(run_circuit_depth.__globals__["_run_cell"])


def test_depth_is_the_layer_count_and_the_sign_map_is_the_inline_one():
    list_circuit = build_ry_linear_circuit(2, 1, return_tuple=False)
    tuple_circuit = build_ry_linear_circuit(2, 1, return_tuple=True)
    params = pnp.zeros((1, 2))
    query = pnp.array([1, -1])
    assert isinstance(list_circuit(params, query), list)
    assert isinstance(tuple_circuit(params, query), tuple)

    def stand_in(_params, _query):
        return [0, 0.2, -0.3]

    signed = np.array(sign_circuit_output(stand_in, None, None))
    via_recall = np.array(qam_recall(None, stand_in, None))
    assert np.array_equal(signed, [1, 1, -1])
    assert np.array_equal(signed, via_recall)
    smoke = load_benchmark_config(SMOKE)
    assert smoke.depths == (1,)
    assert smoke.cases == ((8, 4),)
    assert smoke.grid == "smoke"
    assert smoke.executable is True
    assert lightweight_request().depths == (1,)
    assert lightweight_request().return_tuple is False


def test_smoke_configuration_keeps_the_module_training_budget():
    config = load_benchmark_config(SMOKE)
    assert config.train_steps == 100
    assert config.seeds == (0, 1, 2, 3, 4)
    assert config.lr == 0.05
    assert config.recall_noise == 0.15
    assert config.return_tuple is False
    assert RECALL_IMPLEMENTATION == "inline_sign_of_circuit_expectations"


def test_historical_grid_and_validation_do_not_train(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("training or main started")

    monkeypatch.setattr(historical, "main", boom)
    monkeypatch.setattr("src.qam.train_ry_linear_qam", boom)
    before = _archive_bytes()
    historical_config = load_benchmark_config(HISTORICAL)
    assert historical_config.executable is False
    with pytest.raises(ConfigNotExecutable):
        run_benchmark(historical_config, output_dir=tmp_path)
    with pytest.raises(HistoricalGridRefused, match="only grid='smoke'"):
        run_circuit_depth(historical_request(), output_dir=tmp_path)
    load_benchmark_config(SMOKE)
    load_benchmark_config(HISTORICAL)
    assert list(tmp_path.iterdir()) == []
    assert _archive_bytes() == before


def test_one_cell_matches_the_direct_schedule(tmp_path, monkeypatch, caplog):
    def boom(*_args, **_kwargs):
        raise AssertionError("experiment main started")

    monkeypatch.setattr(historical, "main", boom)
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    states = _states()
    before = _archive_bytes()
    try:
        _restore(states)
        payload = run_benchmark(
            load_benchmark_config(SMOKE),
            output_dir=tmp_path,
        )
        framework_state = np.random.get_state()
        _restore(states)
        direct = _direct_cell()
        direct_state = np.random.get_state()
    finally:
        _restore(states)

    assert _same_state(framework_state, direct_state)
    assert payload["comparable_to_archive"] is False
    assert payload["main_invoked"] is False
    assert payload["procedure_ids"] == ["qam_ry_linear"]
    row = payload["rows"][0]
    assert row["n_bits"] == 8
    assert row["m"] == 4
    assert row["depth"] == 1
    assert row["return_tuple"] is False
    assert row["procedure_id"] == "qam_ry_linear"
    assert row["params"] == 8
    assert row["sample_size"] == 5
    assert row["full_acc_mean"] == direct["full_acc_mean"]
    assert row["bit_acc_mean"] == direct["bit_acc_mean"]
    assert len(payload["trials"]) == 5
    for trial, expected in zip(payload["trials"], direct["trials"]):
        assert trial["patterns"] == expected["patterns"]
        assert trial["full_acc"] == expected["full_acc"]
        assert trial["bit_acc"] == expected["bit_acc"]
        assert np.allclose(trial["params"], expected["params"])
        assert trial["return_tuple"] is False
        assert trial["depth"] == 1
    archived = _first_archive_row()
    assert row["full_acc_mean"] == pytest.approx(float(archived["full_acc_mean"]))
    assert row["bit_acc_mean"] == pytest.approx(float(archived["bit_acc_mean"]))
    assert row["params"] == int(archived["params"])
    run_dir = Path(payload["output_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["comparable_to_archive"] is False
    assert manifest["resumable"] is False
    assert manifest["return_tuple"] is False
    assert manifest["recall_implementation"] == RECALL_IMPLEMENTATION
    assert manifest["depth_rule_not_applied"] == "min(8, n_bits)"
    assert manifest["bootstrap_used"] is False
    assert manifest["rng"]["exact_reproducibility_claimed"] is False
    assert manifest["rng"]["pennylane_seed_call"] is False
    assert manifest["historical_cells"] == 21
    assert manifest["smoke_cells"] == 1
    text = caplog.text
    assert "trial_start" in text
    assert "model_train" in text
    assert "model_evaluate" in text
    assert "aggregate_start" in text
    assert "persist_complete" in text
    assert "progress=1/5" in text
    assert "progress=5/5" in text
    assert _archive_bytes() == before


def test_archived_table_keeps_its_schema():
    with ARCHIVE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 21
    assert list(rows[0]) == [
        "n_bits",
        "m",
        "depth",
        "full_acc_mean",
        "bit_acc_mean",
        "train_time_mean",
        "params",
    ]
    assert rows[0]["n_bits"] == "8"
    assert rows[0]["m"] == "4"
    assert rows[0]["depth"] == "1"
    assert rows[-1]["n_bits"] == "16"
    assert rows[-1]["depth"] == "12"


def test_command_line_distinguishes_the_smoke_from_the_historical_grid(capsys, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("training or main started")

    monkeypatch.setattr(historical, "main", boom)
    monkeypatch.setattr("src.qam.train_ry_linear_qam", boom)
    assert main(["validate", str(SMOKE)]) == EXIT_OK
    smoke_plan = capsys.readouterr().out
    assert "executable: yes" in smoke_plan
    assert "qam_ry_linear" in smoke_plan
    assert "return_tuple: false" in smoke_plan
    assert "train_steps: 100" in smoke_plan
    assert "dataset.depths: 1" in smoke_plan
    assert "cells: 1" in smoke_plan
    assert main(["validate", str(HISTORICAL)]) == EXIT_OK
    historical_plan = capsys.readouterr().out
    assert "executable: no" in historical_plan
    assert "cells: 21" in historical_plan
    assert "dataset.depths: 1 2 4 6 8 10 12" in historical_plan
    assert main(["run", str(HISTORICAL), "--dry-run"]) == EXIT_OK
    assert "full_grid: not started" in capsys.readouterr().out
    assert main(["run", str(HISTORICAL)]) == EXIT_NOT_EXECUTABLE
    assert main(["run", str(SMOKE), "--full-grid", "--dry-run"]) == EXIT_NOT_EXECUTABLE
    assert "--full-grid applies only" in capsys.readouterr().err


def test_output_under_the_archive_is_refused(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("training started")

    monkeypatch.setattr("src.qam.train_ry_linear_qam", boom)
    before = _archive_bytes()
    with pytest.raises(ArchivedResultsRefused):
        run_benchmark(load_benchmark_config(SMOKE), output_dir=ARCHIVE.parent)
    assert _archive_bytes() == before


def test_a_different_circuit_setting_is_rejected(tmp_path):
    document = json.loads(SMOKE.read_text(encoding="utf-8"))
    document["models"][0]["parameters"]["return_tuple"] = True
    path = tmp_path / "tuple.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="return_tuple"):
        load_benchmark_config(path)

    document = json.loads(SMOKE.read_text(encoding="utf-8"))
    document["models"][0]["recall"] = "src.qam.qam_recall"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="recall"):
        load_benchmark_config(path)

    document = json.loads(SMOKE.read_text(encoding="utf-8"))
    document["models"][0]["parameters"]["train_steps"] = 1
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="train_steps"):
        load_benchmark_config(path)

    document = json.loads(SMOKE.read_text(encoding="utf-8"))
    document["dataset"]["cases"] = [{"n_bits": 8, "m": 1}]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="not a module"):
        load_benchmark_config(path)

    document = json.loads(SMOKE.read_text(encoding="utf-8"))
    document["dataset"]["depths"] = [3]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="not a module depth"):
        load_benchmark_config(path)


def _direct_cell():
    n_bits, m, depth = 8, 4, 1
    trials = []
    full_scores = []
    bit_scores = []
    for seed in range(5):
        np.random.seed(seed)
        patterns = np.random.choice([-1, 1], size=(m, n_bits))
        params, circuit, _train_time = train_ry_linear_qam(
            patterns,
            n_bits,
            depth,
            steps=100,
            lr=0.05,
            return_tuple=False,
        )
        recalled = []
        for pattern in patterns:
            noisy = corrupt_fixed_count(pattern, 0.15, rng=np.random)
            out = pnp.array(circuit(params, noisy))
            out = pnp.sign(out)
            out[out == 0] = 1
            recalled.append(out)
        full, bit = full_and_bitwise_accuracy(recalled, patterns)
        full_scores.append(float(full))
        bit_scores.append(float(bit))
        trials.append(
            {
                "patterns": np.asarray(patterns).astype(int).tolist(),
                "full_acc": float(full),
                "bit_acc": float(bit),
                "params": np.asarray(params, dtype=float),
            }
        )
    return {
        "trials": trials,
        "full_acc_mean": float(np.mean(np.asarray(full_scores, dtype=float))),
        "bit_acc_mean": float(np.mean(np.asarray(bit_scores, dtype=float))),
    }


def _first_archive_row():
    with ARCHIVE.open(encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle))
