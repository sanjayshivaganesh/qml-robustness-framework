"""Storage-capacity migration. Does not start the 33-cell sweep.

The smoke cell is n=8, m=1, seeds 0..4, and 100 RY-linear steps. Its
critical ratio is empty because that single mean is not below 0.75. It is
not the archived phase-transition table.
"""

import csv
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pytest
from pennylane import numpy as pnp

import experiments.storage_capacity as historical
from benchmark.cli import EXIT_NOT_EXECUTABLE, EXIT_OK, main
from benchmark.config_loader import ConfigNotExecutable, ConfigValidationError, load_benchmark_config
from benchmark.log import LOGGER_NAME
from benchmark.output import ArchivedResultsRefused
from benchmark.protocols.storage_capacity import (
    N_AT_LEAST_16_M,
    SWEEP_COLUMNS,
    historical_cell_count,
    historical_request,
    is_historical_request,
    lightweight_request,
    m_values_for,
    n_flips_for,
    phase_transition_rows,
    run_storage_capacity,
    summarize_cell,
)
from benchmark.registry import InvalidModelParameters, get, prepare
from benchmark.runner import run_benchmark
from benchmark.scoring import WILCOXON_FAILURE, critical_capacity_ratio, selected
from src.data_generation import corrupt_fixed_count
from src.hopfield import Hopfield
from src.metrics import full_and_bitwise_accuracy
from src.paths import PROJECT_ROOT
from src.qam import qam_recall, train_ry_linear_qam

SMOKE = PROJECT_ROOT / "configs" / "storage_capacity.smoke.json"
HISTORICAL = PROJECT_ROOT / "configs" / "storage_capacity.json"
SUMMARY = PROJECT_ROOT / "results" / "storage_capacity" / "summary"
ARCHIVE_SWEEP = SUMMARY / "storage_capacity_sweep.csv"
ARCHIVE_PHASE = SUMMARY / "phase_transition.csv"
ARCHIVE_SIGNIFICANCE = SUMMARY / "statistical_significance_summary.csv"


def _states():
    return np.random.get_state(), pnp.random.get_state()


def _restore(states):
    numpy_state, pennylane_state = states
    np.random.set_state(numpy_state)
    pnp.random.set_state(pennylane_state)


def _same_state(left, right):
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def _archive_bytes():
    return tuple(path.read_bytes() for path in (ARCHIVE_SWEEP, ARCHIVE_PHASE, ARCHIVE_SIGNIFICANCE))


def test_historical_configuration_selects_hopfield_and_ry_linear():
    loaded = load_benchmark_config(HISTORICAL)
    audit = json.loads(
        (PROJECT_ROOT / "baseline" / "configs" / "original_experiments.json").read_text(
            encoding="utf-8"
        )
    )["experiments"]["storage_capacity"]
    assert loaded.protocol == "storage_capacity"
    assert loaded.executable is False
    assert loaded.procedure_ids == ("hopfield_synchronous", "qam_ry_linear")
    assert loaded.train_steps == historical.train_steps == audit["qam"]["train_steps"] == 100
    assert loaded.recall_noise == pytest.approx(historical.recall_noise) == pytest.approx(0.15)
    assert loaded.bootstrap_samples == historical.bootstrap_samples == 2000
    assert loaded.seeds == (0, 1, 2, 3, 4)
    assert loaded.widths == (
        (8, tuple(range(1, 9))),
        (12, tuple(range(1, 13))),
        (16, N_AT_LEAST_16_M),
    )
    assert historical_cell_count() == 33
    assert sum(len(m_values) for _n, m_values in loaded.widths) * len(loaded.seeds) == 165
    assert m_values_for(8) == tuple(range(1, 9))
    assert m_values_for(12) == tuple(range(1, 13))
    assert m_values_for(16) == N_AT_LEAST_16_M
    assert m_values_for(16, reduced_scale_for_large_n=False) == tuple(range(1, 17))
    assert audit["m_values"]["n_at_least_16"] == list(N_AT_LEAST_16_M)
    registration = get("qam_ry_linear")
    assert registration.implementation is train_ry_linear_qam
    assert registration.adapter.recall_implementation is qam_recall
    assert get("hopfield_synchronous").implementation is Hopfield
    assert get("qam_paulix").implementation is not train_ry_linear_qam
    assert get("qam_rot_minibatch").implementation is not train_ry_linear_qam
    prepare("qam_ry_linear", {"n_bits": 8, "depth": 8, "return_tuple": True})
    with pytest.raises(InvalidModelParameters):
        prepare(
            "qam_ry_linear",
            {"n_bits": 8, "depth": 8, "return_tuple": True, "train_steps": 1},
        )
    with pytest.raises(ConfigNotExecutable):
        loaded.to_runner_request()
    allowed = loaded.to_runner_request(allow_historical_grid=True)
    assert is_historical_request(allowed)
    assert len(tuple(allowed.cells())) == 33
    assert "results" not in Path(loaded.output_destination).parts
    source = inspect.getsource(historical.bootstrap_ci)
    assert "rng=onp.random" in source
    assert "seed(" not in source
    assert "confidence_interval(" not in (
        PROJECT_ROOT / "benchmark" / "protocols" / "storage_capacity.py"
    ).read_text(encoding="utf-8")


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_capacity_calculations_use_the_historical_helpers():
    assert n_flips_for(8, 0.15) == max(1, int(8 * 0.15)) == 1
    assert n_flips_for(16, 0.15) == max(1, int(16 * 0.15)) == 2
    assert 1 / 8 == pytest.approx(0.125)
    assert 3 / 8 == pytest.approx(0.375)
    assert critical_capacity_ratio([0.125, 0.25, 0.375], [1.0, 1.0, 0.7]) == pytest.approx(0.375)
    assert critical_capacity_ratio([0.125, 0.25, 0.375], [1.0, 0.75, 0.5]) == pytest.approx(0.375)
    assert critical_capacity_ratio([0.125, 0.25], [1.0, 0.9]) is None

    states = _states()
    try:
        np.random.seed(3)
        summary = summarize_cell(
            hopfield_full=[0.2, 0.4, 0.6, 0.8, 1.0],
            hopfield_bit=[0.5, 0.55, 0.6, 0.65, 0.7],
            hopfield_train_times=[0.01, 0.02, 0.03, 0.04, 0.05],
            hopfield_infer_times=[0.1, 0.1, 0.1, 0.1, 0.1],
            qam_full=[1.0, 0.8, 0.6, 0.4, 0.2],
            qam_bit=[0.9, 0.8, 0.7, 0.6, 0.5],
            qam_train_times=[1.0, 1.1, 1.2, 1.3, 1.4],
            qam_infer_times=[0.2, 0.2, 0.2, 0.2, 0.2],
            qam_param_counts=[64, 64, 64, 64, 64],
            bootstrap_samples=30,
        )
        np.random.seed(3)
        direct_full = historical.bootstrap_ci([0.2, 0.4, 0.6, 0.8, 1.0], n_boot=30)
        direct_bit = historical.bootstrap_ci([0.5, 0.55, 0.6, 0.65, 0.7], n_boot=30)
        direct_qam = historical.bootstrap_ci([1.0, 0.8, 0.6, 0.4, 0.2], n_boot=30)
        direct_qam_bit = historical.bootstrap_ci([0.9, 0.8, 0.7, 0.6, 0.5], n_boot=30)
        direct_train = historical.bootstrap_ci([1.0, 1.1, 1.2, 1.3, 1.4], n_boot=30)
        historical.bootstrap_ci([0.2, 0.2, 0.2, 0.2, 0.2], n_boot=30)
    finally:
        _restore(states)

    assert summary["hopfield_full"] == pytest.approx(direct_full)
    assert summary["hopfield_bit"] == pytest.approx(direct_bit)
    assert summary["qam_full"] == pytest.approx(direct_qam)
    assert summary["qam_bit"] == pytest.approx(direct_qam_bit)
    assert summary["qam_train_mean"] == pytest.approx(direct_train[0])
    assert summary["hopfield_train_mean"] == pytest.approx(0.03)
    assert summary["p_value_full"] == pytest.approx(
        selected("storage_capacity", "wilcoxon_pvalue")(
            [0.2, 0.4, 0.6, 0.8, 1.0],
            [1.0, 0.8, 0.6, 0.4, 0.2],
            on_failure=WILCOXON_FAILURE["storage_capacity"],
        )
    )
    assert WILCOXON_FAILURE["storage_capacity"] is None
    tied = np.ones(1)
    with pytest.raises(ValueError):
        from scipy.stats import wilcoxon

        wilcoxon(tied, tied)
    assert selected("storage_capacity", "wilcoxon_pvalue")(
        tied, tied, on_failure=None
    ) is None

    synthetic = [
        _synthetic_row("Hopfield", "hopfield_synchronous", 8, 1, 1.0),
        _synthetic_row("QAM", "qam_ry_linear", 8, 1, 1.0),
        _synthetic_row("Hopfield", "hopfield_synchronous", 8, 3, 0.7),
        _synthetic_row("QAM", "qam_ry_linear", 8, 2, 0.7),
    ]
    phases = phase_transition_rows(synthetic, grid="controlled")
    assert phases[0]["historical_model_label"] == "Hopfield"
    assert phases[0]["critical_capacity_ratio"] == pytest.approx(0.375)
    assert phases[1]["historical_model_label"] == "QAM"
    assert phases[1]["critical_capacity_ratio"] == pytest.approx(0.25)
    assert phases[0]["comparable_to_archive"] is False


def test_smoke_configuration_keeps_the_module_training_budget():
    loaded = load_benchmark_config(SMOKE)
    assert loaded.executable is True
    assert loaded.widths == ((8, (1,)),)
    assert loaded.seeds == (0, 1, 2, 3, 4)
    assert loaded.train_steps == 100
    assert loaded.bootstrap_samples == 2000
    request = loaded.to_runner_request()
    assert request.grid == "smoke"
    assert not is_historical_request(request)
    assert lightweight_request().widths == ((8, (1,)),)
    assert historical.depth_policy(8) == 8
    assert historical.depth_policy(12) == 8


def test_historical_grid_and_validation_do_not_train(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("storage-capacity training started")

    monkeypatch.setattr(historical, "main", boom)
    monkeypatch.setattr("src.qam.train_ry_linear_qam", boom)
    before = _archive_bytes()
    loaded = load_benchmark_config(HISTORICAL)
    assert loaded.executable is False
    with pytest.raises(ConfigNotExecutable):
        run_benchmark(loaded, output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="only grid='smoke'"):
        run_storage_capacity(historical_request(), output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert _archive_bytes() == before


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_one_cell_matches_the_direct_schedule(tmp_path, monkeypatch, caplog):
    def boom(*_args, **_kwargs):
        raise AssertionError("experiment main was called")

    monkeypatch.setattr(historical, "main", boom)
    caplog.set_level("INFO", logger=LOGGER_NAME)
    before = _archive_bytes()
    states = _states()
    try:
        direct_trials, direct_stats, direct_state = _direct_cell()
        _restore(states)
        result = run_benchmark(load_benchmark_config(SMOKE), output_dir=tmp_path)
        framework_state = np.random.get_state()
    finally:
        _restore(states)

    assert _same_state(direct_state, framework_state)
    assert result["execution_status"] == "succeeded"
    assert result["comparable_to_archive"] is False
    assert result["main_invoked"] is False
    run_dir = Path(result["output_dir"])
    assert run_dir.is_relative_to(tmp_path)
    assert _archive_bytes() == before

    text = caplog.text
    assert "event=trial_start" in text
    assert "event=model_train" in text
    assert "event=model_evaluate" in text
    assert "event=statistics_start" in text
    assert "event=persist_complete" in text
    assert "progress=1/5" in text
    assert "progress=5/5" in text
    assert "event=run_start" in text

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["historical_train_steps"] == 100
    assert manifest["smoke_cells"] == 1
    assert manifest["historical_cells"] == 33
    assert manifest["historical_trials_per_cell"] == 5
    assert manifest["bootstrap_separate_seed"] is False
    assert manifest["critical_threshold"] == 0.75
    assert manifest["wilcoxon_failure_value"] is None
    assert manifest["plotting"] is False
    assert manifest["main_invoked"] is False
    assert manifest["resumable"] is False
    assert manifest["comparable_to_archive"] is False
    assert manifest["procedure_ids"] == ["hopfield_synchronous", "qam_ry_linear"]
    rng = manifest["rng"]
    assert rng["exact_reproducibility_claimed"] is False
    assert rng["pennylane_seed_call"] is False
    assert rng["internal_resets"] == []
    assert rng["bootstrap"]["separate_seed"] is False
    assert rng["bootstrap"]["samples"] == 2000
    assert rng["unseeded_procedures"][0]["name"] == "experiments.storage_capacity.confidence_interval"
    assert [item["value"] for item in rng["seeds_applied"][0]["values"]] == [0, 1, 2, 3, 4]

    with (run_dir / "sweep.csv").open(encoding="utf-8", newline="") as handle:
        table = list(csv.reader(handle))
    assert tuple(table[0]) == SWEEP_COLUMNS
    assert len(table) == 3
    sweep = json.loads((run_dir / "sweep.json").read_text(encoding="utf-8"))
    hopfield, qam = sweep
    assert hopfield["procedure_id"] == "hopfield_synchronous"
    assert hopfield["historical_model_label"] == "Hopfield"
    assert qam["procedure_id"] == "qam_ry_linear"
    assert qam["historical_model_label"] == "QAM"
    assert hopfield["m"] == 1 and hopfield["n_bits"] == 8
    assert hopfield["capacity_ratio"] == pytest.approx(0.125)
    assert hopfield["sample_size"] == 5
    assert hopfield["bootstrap_samples"] == 2000
    assert hopfield["params_count"] == pytest.approx(64)
    assert qam["params_count"] == pytest.approx(64)
    assert qam["extra_notes"] == "Variational QAM | depth=8 | steps=100 | lr=0.05"
    assert hopfield["extra_notes"] == "Associative recall from corrupted inputs"
    assert hopfield["full_acc_mean"] == pytest.approx(direct_stats["hopfield_full"][0])
    assert hopfield["full_acc_ci_low"] == pytest.approx(direct_stats["hopfield_full"][1])
    assert hopfield["full_acc_ci_high"] == pytest.approx(direct_stats["hopfield_full"][2])
    assert qam["full_acc_mean"] == pytest.approx(direct_stats["qam_full"][0])
    assert qam["bit_acc_mean"] == pytest.approx(direct_stats["qam_bit"][0])
    assert hopfield["p_value_full_vs_qam"] == pytest.approx(direct_stats["p_full"])
    assert hopfield["comparable_to_archive"] is False

    trials = json.loads((run_dir / "trials.json").read_text(encoding="utf-8"))
    assert [trial["seed"] for trial in trials] == [0, 1, 2, 3, 4]
    for produced, expected in zip(trials, direct_trials):
        assert produced["patterns"] == expected["patterns"]
        assert produced["hopfield_full"] == pytest.approx(expected["hopfield_full"])
        assert produced["qam_full"] == pytest.approx(expected["qam_full"])
        assert produced["qam_bit"] == pytest.approx(expected["qam_bit"])
        assert produced["depth"] == 8
        assert produced["train_steps"] == 100
        assert np.allclose(produced["qam_params"], expected["params"])

    phase = json.loads((run_dir / "phase_transition.json").read_text(encoding="utf-8"))
    assert phase["threshold"] == 0.75
    assert phase["rows"][0]["critical_capacity_ratio"] is None
    assert phase["rows"][1]["critical_capacity_ratio"] is None
    with ARCHIVE_PHASE.open(encoding="utf-8", newline="") as handle:
        archived_phase = next(csv.DictReader(handle))
    assert float(archived_phase["critical_capacity_ratio"]) == pytest.approx(0.375)
    assert phase["rows"][0]["critical_capacity_ratio"] != float(
        archived_phase["critical_capacity_ratio"]
    )

    with ARCHIVE_SWEEP.open(encoding="utf-8", newline="") as handle:
        archived = next(csv.DictReader(handle))
    assert archived["n_bits"] == "8" and archived["m"] == "1"
    assert float(archived["full_acc_mean"]) == pytest.approx(hopfield["full_acc_mean"])


def test_archived_tables_keep_their_schema():
    with ARCHIVE_SWEEP.open(encoding="utf-8", newline="") as handle:
        sweep = list(csv.reader(handle))
    with ARCHIVE_PHASE.open(encoding="utf-8", newline="") as handle:
        phase = list(csv.reader(handle))
    with ARCHIVE_SIGNIFICANCE.open(encoding="utf-8", newline="") as handle:
        significance = list(csv.reader(handle))
    assert sweep[0][:8] == [
        "model",
        "n_bits",
        "m",
        "capacity_ratio",
        "full_acc_mean",
        "full_acc_ci_low",
        "full_acc_ci_high",
        "bit_acc_mean",
    ]
    assert len(sweep) == 67
    assert phase[0] == ["model", "n_bits", "critical_capacity_ratio"]
    assert len(phase) == 7
    assert significance[0] == [
        "n_bits",
        "m",
        "capacity_ratio",
        "p_value_full_vs_qam",
        "p_value_bit_vs_qam",
    ]
    assert len(significance) == 34


def test_command_line_distinguishes_the_smoke_from_the_historical_grid(capsys, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("training or main started")

    monkeypatch.setattr(historical, "main", boom)
    monkeypatch.setattr("src.qam.train_ry_linear_qam", boom)
    assert main(["validate", str(SMOKE)]) == EXIT_OK
    smoke_plan = capsys.readouterr().out
    assert "executable: yes" in smoke_plan
    assert "qam_ry_linear" in smoke_plan
    assert "train_steps: 100" in smoke_plan
    assert "cells: 1" in smoke_plan
    assert "bootstrap_separate_seed: false" in smoke_plan
    assert main(["validate", str(HISTORICAL)]) == EXIT_OK
    historical_plan = capsys.readouterr().out
    assert "executable: no" in historical_plan
    assert "cells: 33" in historical_plan
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
        run_benchmark(load_benchmark_config(SMOKE), output_dir=SUMMARY)
    assert _archive_bytes() == before


def test_a_shorter_training_budget_is_rejected(tmp_path):
    document = json.loads(SMOKE.read_text(encoding="utf-8"))
    document["models"][1]["parameters"]["train_steps"] = 1
    path = tmp_path / "steps.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="train_steps"):
        load_benchmark_config(path)


def _synthetic_row(label, procedure_id, n_bits, m, full_mean):
    return {
        "historical_model_label": label,
        "procedure_id": procedure_id,
        "n_bits": n_bits,
        "m": m,
        "capacity_ratio": m / n_bits,
        "full_acc_mean": full_mean,
        "p_value_full_vs_qam": 0.5,
        "p_value_bit_vs_qam": 0.5,
        "sample_size": 5,
    }


def _direct_cell():
    n_bits = 8
    count = 1
    depth = min(8, n_bits)
    trials = []
    hop_full, hop_bit = [], []
    qam_full, qam_bit = [], []
    hop_train_times, hop_infer_times = [], []
    qam_train_times, qam_infer_times, qam_param_counts = [], [], []
    for seed in range(5):
        np.random.seed(seed)
        patterns = np.random.choice([-1, 1], size=(count, n_bits))
        hopfield = Hopfield(n_bits)
        started = time.time()
        hopfield.train(patterns)
        hop_train_times.append(time.time() - started)
        started = time.time()
        recalled = pnp.array(
            [
                hopfield.recall(corrupt_fixed_count(pattern, 0.15, rng=np.random))
                for pattern in patterns
            ]
        )
        hop_infer_times.append((time.time() - started) / count)
        full, bit = full_and_bitwise_accuracy(recalled, patterns)
        hop_full.append(float(full))
        hop_bit.append(float(bit))
        params, circuit, train_time = train_ry_linear_qam(
            patterns, n_bits, depth, steps=100, lr=0.05, return_tuple=True
        )
        qam_train_times.append(float(train_time))
        started = time.time()
        recalled_qam = pnp.array(
            [
                qam_recall(
                    corrupt_fixed_count(pattern, 0.15, rng=np.random),
                    circuit,
                    params,
                )
                for pattern in patterns
            ]
        )
        qam_infer_times.append((time.time() - started) / count)
        qam_full_score, qam_bit_score = full_and_bitwise_accuracy(recalled_qam, patterns)
        qam_full.append(float(qam_full_score))
        qam_bit.append(float(qam_bit_score))
        qam_param_counts.append(int(np.asarray(params).size))
        trials.append(
            {
                "patterns": np.asarray(patterns).astype(int).tolist(),
                "hopfield_full": float(full),
                "qam_full": float(qam_full_score),
                "qam_bit": float(qam_bit_score),
                "params": np.asarray(params, dtype=float),
            }
        )
    hopfield_full_stats = historical.bootstrap_ci(hop_full)
    historical.bootstrap_ci(hop_bit)
    qam_full_stats = historical.bootstrap_ci(qam_full)
    qam_bit_stats = historical.bootstrap_ci(qam_bit)
    historical.bootstrap_ci(qam_train_times)
    historical.bootstrap_ci(qam_infer_times)
    from scipy.stats import wilcoxon

    stats = {
        "hopfield_full": hopfield_full_stats,
        "qam_full": qam_full_stats,
        "qam_bit": qam_bit_stats,
        "p_full": float(wilcoxon(hop_full, qam_full).pvalue),
    }
    return trials, stats, np.random.get_state()
