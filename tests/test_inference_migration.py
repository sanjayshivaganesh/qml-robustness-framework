"""Inference migration. Does not start the 216,000-task grid.

The smoke cell is n=8, capacity 0.1, the second module noise level, and
trials 0..299. Trial 300 stays the module count because the summary divides
its Gaussian columns by that constant. The cell is not the archived summary.
"""

import csv
import hashlib
import inspect
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.noise_robustness_inference as historical
from benchmark.cli import EXIT_NOT_EXECUTABLE, EXIT_OK, main
from benchmark.config_loader import ConfigNotExecutable, ConfigValidationError, load_benchmark_config
from benchmark.log import LOGGER_NAME
from benchmark.output import ArchivedResultsRefused
from benchmark.protocols.noise_robustness_inference import (
    CSV_MODEL_LABEL,
    HistoricalGridRefused,
    historical_noise_levels,
    historical_request,
    historical_task_count,
    is_historical_inference_request,
    lightweight_request,
    pattern_count,
    run_inference,
    task_seed,
)
from benchmark.registry import ModelHasNoConstructor, UnknownModel, get, prepare
from benchmark.runner import run_benchmark
from src.hopfield import hopfield_async_sequential_recall
from src.paths import PROJECT_ROOT
from src.qam import nearest_neighbor_recall

SMOKE = PROJECT_ROOT / "configs" / "noise_robustness_inference.smoke.json"
HISTORICAL = PROJECT_ROOT / "configs" / "noise_robustness_inference.json"
ARCHIVE_DIR = PROJECT_ROOT / "results" / "noise_robustness_inference"
ARCHIVE_RAW = ARCHIVE_DIR / "raw" / "noise_inference_rows.csv"
ARCHIVE_SUMMARY = ARCHIVE_DIR / "summary" / "noise_inference_summary.csv"
ARCHIVE_FILES = (
    ARCHIVE_RAW,
    ARCHIVE_SUMMARY,
    ARCHIVE_DIR / "summary" / "critical_noise_thresholds.csv",
    ARCHIVE_DIR / "paper_figures" / "critical_noise_thresholds.csv",
    ARCHIVE_DIR / "experiment.log",
)
METRIC_KEYS = (
    "bit_accuracy",
    "full_accuracy",
    "hamming_error",
    "input_bit_accuracy",
    "input_hamming_error",
    "recovery_gain",
    "energy",
    "energy_drop",
)


def _archive_digest():
    digest = hashlib.sha256()
    for path in ARCHIVE_FILES:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_historical_configuration_names_hopfield_and_hamming():
    config = load_benchmark_config(HISTORICAL)
    audit = json.loads(
        (PROJECT_ROOT / "baseline" / "configs" / "original_experiments.json").read_text(
            encoding="utf-8"
        )
    )["experiments"]["noise_robustness_inference"]
    assert config.protocol == "noise_robustness_inference"
    assert config.procedure_ids == ("hopfield_async_sequential", "hamming_nearest")
    assert config.executable is False
    assert config.n_bits_list == tuple(historical.N_BITS_LIST)
    assert config.capacity_ratios == tuple(historical.CAPACITY_RATIOS)
    assert config.noise_levels == historical_noise_levels()
    assert len(config.noise_levels) == 16
    assert config.trial_ids == tuple(range(300))
    assert historical_task_count() == 9 * 5 * 16 * 300 == 216000
    assert audit["task_count"] == 216000
    assert audit["raw_row_count_expected"] == 432000
    assert audit["model_labeled_qam"]["circuit"] is False
    assert audit["model_labeled_qam"]["function"] == "src.qam.nearest_neighbor_recall"
    assert audit["hopfield"]["steps"] == 20
    request = historical_request()
    assert is_historical_inference_request(request)
    assert get("hopfield_async_sequential").implementation is hopfield_async_sequential_recall
    assert get("hamming_nearest").implementation is nearest_neighbor_recall
    with pytest.raises(UnknownModel):
        get("qam")
    with pytest.raises(ModelHasNoConstructor):
        prepare("hamming_nearest", {})
    source = inspect.getsource(historical.run_trial)
    assert "nearest_neighbor_recall" in source
    assert "hopfield_async_sequential_recall" in source
    assert 'model": "qam"' in source or '"model": "qam"' in source
    assert "qml" not in source
    assert pattern_count(8, 0.1) == 1
    assert pattern_count(8, 0.25) == 2
    assert pattern_count(128, 1.0) == 128
    noise = historical_noise_levels()[1]
    assert task_seed(8, 0.1, noise, 12) == 42 + 12 + int(8 * 1000) + int(0.1 * 100) + int(noise * 1000)


def test_smoke_keeps_three_hundred_trials_and_one_module_noise():
    config = load_benchmark_config(SMOKE)
    smoke = lightweight_request()
    assert config.executable is True
    assert config.grid == "smoke"
    assert config.n_bits_list == smoke.n_bits_list == (8,)
    assert config.capacity_ratios == smoke.capacity_ratios == (0.1,)
    assert config.noise_levels == smoke.noise_levels == (historical_noise_levels()[1],)
    assert config.trial_ids == tuple(range(300))
    assert CSV_MODEL_LABEL["hamming_nearest"] == "qam"
    assert CSV_MODEL_LABEL["hopfield_async_sequential"] == "hopfield"


def test_historical_grid_and_validation_do_not_recall(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("inference trial or main started")

    monkeypatch.setattr(historical, "main", boom)
    monkeypatch.setattr(historical, "run_trial", boom)
    before = _archive_digest()
    historical_config = load_benchmark_config(HISTORICAL)
    assert historical_config.executable is False
    with pytest.raises(ConfigNotExecutable):
        run_benchmark(historical_config, output_dir=tmp_path)
    with pytest.raises(HistoricalGridRefused, match="only grid='smoke'"):
        run_inference(historical_request(), output_dir=tmp_path)
    load_benchmark_config(SMOKE)
    assert list(tmp_path.iterdir()) == []
    assert _archive_digest() == before


def test_one_cell_matches_run_trial_and_one_archived_task(tmp_path, monkeypatch, caplog):
    def boom(*_args, **_kwargs):
        raise AssertionError("experiment main started")

    monkeypatch.setattr(historical, "main", boom)
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    before = _archive_digest()
    request = lightweight_request()
    state = np.random.get_state()
    try:
        np.random.seed(0)
        payload = run_benchmark(load_benchmark_config(SMOKE), output_dir=tmp_path)
        np.random.seed(0)
        direct_rows = []
        for task in request.tasks():
            direct_rows.extend(historical.run_trial(task))
        direct_summary = historical.aggregate_results(pd.DataFrame(direct_rows))
    finally:
        np.random.set_state(state)

    assert payload["comparable_to_archive"] is False
    assert payload["main_invoked"] is False
    assert payload["training"] is False
    assert payload["procedure_ids"] == ["hopfield_async_sequential", "hamming_nearest"]
    assert len(payload["rows"]) == 600
    assert len(payload["summary"]) == 2
    by_key = {(row["model"], row["trial"]): row for row in payload["rows"]}
    for record in direct_rows:
        got = by_key[(record["model"], record["trial"])]
        assert got["procedure_id"] == {
            "hopfield": "hopfield_async_sequential",
            "qam": "hamming_nearest",
        }[record["model"]]
        assert got["model"] == record["model"]
        for key in METRIC_KEYS:
            assert _same_number(got[key], record[key]), key
    qam_rows = [row for row in payload["rows"] if row["model"] == "qam"]
    assert all(row["procedure_id"] == "hamming_nearest" for row in qam_rows)
    assert all(row["energy"] is None and row["energy_drop"] is None for row in qam_rows)
    flipped = by_key[("hopfield", 12)]
    assert flipped["input_hamming_error"] == pytest.approx(0.125)
    _assert_archived_task(by_key, request.noise_levels[0], 12)
    summary_by_model = {row["model"]: row for row in payload["summary"]}
    for item in direct_summary.to_dict(orient="records"):
        got = summary_by_model[item["model"]]
        assert got["procedure_id"] == {
            "hopfield": "hopfield_async_sequential",
            "qam": "hamming_nearest",
        }[item["model"]]
        assert got["sample_size"] == 300
        for key in ("bit_acc_mean", "full_acc_mean", "bit_ci_low", "full_ci_high"):
            assert got[key] == pytest.approx(float(item[key]))
    archived_summary = _summary_row_count()
    assert archived_summary == 1440
    run_dir = Path(payload["output_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["comparable_to_archive"] is False
    assert manifest["resumable"] is False
    assert manifest["training"] is False
    assert manifest["csv_model_labels"]["hamming_nearest"] == "qam"
    assert manifest["bootstrap_seeded"] is False
    assert manifest["gaussian_denominator"] == "TRIALS_PER_CONFIG"
    assert manifest["process_model"] == "single_process"
    assert manifest["historical_process_model"] == "spawn_pool"
    assert manifest["historical_tasks"] == 216000
    assert manifest["smoke_tasks"] == 300
    assert manifest["rng"]["exact_reproducibility_claimed"] is False
    assert manifest["rng"]["unseeded_procedures"][0]["separate_seed"] is False
    text = caplog.text
    assert "task_progress" in text
    assert "progress=50/300" in text
    assert "progress=300/300" in text
    assert "aggregate_start" in text
    assert "persist_complete" in text
    assert _archive_digest() == before


def test_archived_tables_keep_their_schema():
    with ARCHIVE_RAW.open(encoding="utf-8", newline="") as handle:
        header = handle.readline().strip()
    assert header.split(",") == [
        "model",
        "n_bits",
        "capacity_ratio",
        "noise_p",
        "trial",
        "bit_accuracy",
        "full_accuracy",
        "hamming_error",
        "input_bit_accuracy",
        "input_hamming_error",
        "recovery_gain",
        "runtime",
        "energy",
        "energy_drop",
    ]
    assert ARCHIVE_RAW.stat().st_size == 50153034
    assert _summary_row_count() == 1440


def test_command_line_distinguishes_the_smoke_from_the_historical_grid(capsys, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("inference trial or main started")

    monkeypatch.setattr(historical, "main", boom)
    monkeypatch.setattr(historical, "run_trial", boom)
    assert main(["validate", str(SMOKE)]) == EXIT_OK
    smoke_plan = capsys.readouterr().out
    assert "executable: yes" in smoke_plan
    assert "hamming_nearest" in smoke_plan
    assert "csv_model_labels: hopfield qam" in smoke_plan
    assert "training: no" in smoke_plan
    assert "tasks: 300" in smoke_plan
    assert main(["validate", str(HISTORICAL)]) == EXIT_OK
    historical_plan = capsys.readouterr().out
    assert "executable: no" in historical_plan
    assert "tasks: 216000" in historical_plan
    assert main(["run", str(HISTORICAL), "--dry-run"]) == EXIT_OK
    assert "full_grid: not started" in capsys.readouterr().out
    assert main(["run", str(HISTORICAL)]) == EXIT_NOT_EXECUTABLE
    assert main(["run", str(SMOKE), "--full-grid", "--dry-run"]) == EXIT_NOT_EXECUTABLE
    assert "--full-grid applies only" in capsys.readouterr().err


def test_output_under_the_archive_is_refused(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("inference trial started")

    monkeypatch.setattr(historical, "run_trial", boom)
    before = _archive_digest()
    with pytest.raises(ArchivedResultsRefused):
        run_benchmark(load_benchmark_config(SMOKE), output_dir=ARCHIVE_DIR)
    assert _archive_digest() == before


def test_a_changed_inference_identity_is_rejected(tmp_path):
    document = json.loads(SMOKE.read_text(encoding="utf-8"))
    document["trials_per_config"] = 1
    document["declared_task_count"] = 1
    document["declared_raw_rows"] = 2
    path = tmp_path / "trials.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="trials_per_config"):
        load_benchmark_config(path)

    document = json.loads(SMOKE.read_text(encoding="utf-8"))
    document["models"][1]["variational"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="variational"):
        load_benchmark_config(path)

    document = json.loads(SMOKE.read_text(encoding="utf-8"))
    document["models"][1]["csv_model_label"] = "hamming_nearest"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="csv_model_label"):
        load_benchmark_config(path)

    document = json.loads(SMOKE.read_text(encoding="utf-8"))
    document["models"][1]["procedure_id"] = "qam_ry_linear"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="hamming_nearest"):
        load_benchmark_config(path)

    document = json.loads(SMOKE.read_text(encoding="utf-8"))
    document["dataset"]["noise_levels"] = [0.01]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="not a module value"):
        load_benchmark_config(path)


def _same_number(left, right):
    if left is None:
        return isinstance(right, float) and right != right
    if isinstance(right, float) and right != right:
        return left is None
    return float(left) == pytest.approx(float(right))


def _assert_archived_task(by_key, noise, trial):
    found = {}
    with ARCHIVE_RAW.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["n_bits"] == "8"
                and row["capacity_ratio"] == "0.1"
                and row["trial"] == str(trial)
                and abs(float(row["noise_p"]) - noise) < 1e-12
            ):
                found[row["model"]] = row
                if len(found) == 2:
                    break
    assert set(found) == {"hopfield", "qam"}
    for model, archived in found.items():
        got = by_key[(model, trial)]
        for key in METRIC_KEYS:
            if archived[key] == "":
                assert got[key] is None
            else:
                assert float(got[key]) == pytest.approx(float(archived[key]))


def _summary_row_count():
    with ARCHIVE_SUMMARY.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in handle) - 1
