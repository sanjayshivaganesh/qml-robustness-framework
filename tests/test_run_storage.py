"""Run directories stay outside the archived results tree."""

import csv
import json
from pathlib import Path

import pytest

from benchmark.output import (
    STATUS_FAILED,
    STATUS_SUCCEEDED,
    ArchivedResultsRefused,
    RunDirectoryExists,
    allocate_run_directory,
    begin_run,
    complete_run,
    create_run_dir,
    default_runs_root,
    fail_run,
    write_csv_table,
    write_json,
)
from benchmark.protocols import noise_robustness_sweep as protocol
from src.paths import PROJECT_ROOT


def _snapshot(root: Path):
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _record(run_dir, *, status_finish):
    manifest = begin_run(
        run_dir,
        run_id=run_dir.name,
        experiment_id="noise_robustness_sweep_smoke",
        configuration={"train_steps": 1, "protocol": "noise_robustness_sweep"},
        models=("hopfield_synchronous", "qam_rot_fullbatch"),
        dataset={"n_bits_list": [2], "capacity_ratio": 0.3},
        perturbation={"procedure_id": "bernoulli", "levels": [0.0]},
    )
    if status_finish == STATUS_SUCCEEDED:
        write_csv_table(
            run_dir,
            "trials.csv",
            [{"procedure_id": "hopfield_synchronous", "full_accuracy": 1.0}],
            ("procedure_id", "full_accuracy"),
        )
        return complete_run(
            run_dir,
            manifest,
            files=("manifest.json", "configuration.json", "trials.csv"),
        )
    return fail_run(run_dir, manifest, RuntimeError("stopped"))


def test_default_root_is_not_the_archived_tree():
    root = default_runs_root()
    archive = (PROJECT_ROOT / "results").resolve()
    assert root == (PROJECT_ROOT / "runs").resolve()
    assert root != archive
    assert archive not in root.parents
    assert root.name == "runs"
    assert not root.is_relative_to(archive)


def test_new_runs_are_unique_and_an_existing_directory_is_refused(tmp_path):
    archive = _snapshot(PROJECT_ROOT / "results")
    first, first_id = allocate_run_directory(tmp_path, "noise_robustness_sweep_smoke")
    second, second_id = allocate_run_directory(tmp_path, "noise_robustness_sweep_smoke")
    assert first != second
    assert first_id != second_id
    assert first.parent == tmp_path.resolve()
    assert archive not in first.resolve().parents
    (first / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(RunDirectoryExists):
        create_run_dir(first)
    assert (first / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert _snapshot(PROJECT_ROOT / "results") == archive


def test_structured_results_and_configuration_round_trip(tmp_path):
    run_dir = create_run_dir(tmp_path / "exact")
    manifest = _record(run_dir, status_finish=STATUS_SUCCEEDED)
    assert manifest["execution_status"] == STATUS_SUCCEEDED
    stored = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    configuration = json.loads((run_dir / "configuration.json").read_text(encoding="utf-8"))
    assert stored["execution_status"] == STATUS_SUCCEEDED
    assert stored["experiment_id"] == "noise_robustness_sweep_smoke"
    assert stored["models"] == ["hopfield_synchronous", "qam_rot_fullbatch"]
    assert stored["dataset"]["capacity_ratio"] == 0.3
    assert stored["perturbation"]["procedure_id"] == "bernoulli"
    assert stored["configuration"]["train_steps"] == 1
    assert configuration == stored["configuration"]
    assert stored["comparable_to_archive"] is False
    assert stored["output_class"] == "framework_run"
    assert stored["started_at"]
    assert stored["completed_at"]
    assert "python_version" in stored
    assert "numpy" in stored["package_versions"]
    with (run_dir / "trials.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"procedure_id": "hopfield_synchronous", "full_accuracy": "1.0"}]
    assert not (run_dir / ".trials.csv.partial").exists()


def test_a_failed_run_is_not_marked_successful(tmp_path, monkeypatch):
    def boom(_request):
        raise RuntimeError("stopped")

    monkeypatch.setattr(protocol, "_execute", boom)
    parent = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="stopped"):
        protocol.run_noise_sweep(protocol.lightweight_request(), output_dir=parent)
    children = [path for path in parent.iterdir() if path.is_dir()]
    assert len(children) == 1
    manifest = json.loads((children[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution_status"] == STATUS_FAILED
    assert manifest["execution_status"] != STATUS_SUCCEEDED
    assert "stopped" in manifest["error"]
    assert manifest["completed_at"]
    configuration = json.loads((children[0] / "configuration.json").read_text(encoding="utf-8"))
    assert configuration["train_steps"] == 1
    assert configuration["capacity_ratio"] == 0.3
    assert not (children[0] / "trials.csv").exists()


def test_overwrite_replaces_a_framework_directory_and_still_refuses_the_archive(tmp_path):
    archive = PROJECT_ROOT / "results"
    before = _snapshot(archive)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "already.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(RunDirectoryExists):
        create_run_dir(occupied)
    reused = create_run_dir(occupied, overwrite=True)
    write_json(reused, "manifest.json", {"execution_status": STATUS_SUCCEEDED})
    assert (reused / "already.txt").read_text(encoding="utf-8") == "keep"
    assert json.loads((reused / "manifest.json").read_text(encoding="utf-8"))["execution_status"] == (
        STATUS_SUCCEEDED
    )
    with pytest.raises(ArchivedResultsRefused):
        create_run_dir(archive / "noise_robustness_sweep", overwrite=True)
    with pytest.raises(ArchivedResultsRefused):
        allocate_run_directory(archive / "noise_robustness_sweep", "noise_robustness_sweep")
    assert _snapshot(archive) == before


def test_interrupted_replace_does_not_leave_the_finished_name(tmp_path, monkeypatch):
    run_dir = create_run_dir(tmp_path / "partial")

    def boom(*_args, **_kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr("benchmark.output.os.replace", boom)
    with pytest.raises(OSError, match="replace failed"):
        write_json(run_dir, "manifest.json", {"execution_status": STATUS_SUCCEEDED})
    assert not (run_dir / "manifest.json").exists()
