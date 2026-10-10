"""Command line for one configured smoke. Does not call experiment mains."""

import json
import subprocess
import sys
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
from benchmark.cli import EXIT_INVALID, EXIT_NOT_EXECUTABLE, EXIT_OK, main
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


def _forbid_construction(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("QAM was constructed")

    monkeypatch.setattr(noise_robustness_sweep.QAM, "__init__", boom)


def _example(tmp_path, mutate):
    document = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_help_lists_commands_and_the_smoke_example(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == EXIT_OK
    text = capsys.readouterr().out
    assert "validate" in text
    assert "run" in text
    assert "--dry-run" in text
    assert "configs/noise_robustness_sweep.smoke.json" in text


def test_module_help_uses_the_package_entry_point():
    completed = subprocess.run(
        [sys.executable, "-m", "benchmark", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == EXIT_OK
    assert "validate" in completed.stdout
    assert "configs/noise_robustness_sweep.smoke.json" in completed.stdout


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["validate"],
        ["run"],
        ["run", "--dry-run"],
        ["listen"],
    ],
)
def test_missing_or_unknown_arguments_are_rejected(argv):
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == EXIT_INVALID


def test_validate_reports_the_smoke_plan_without_executing(monkeypatch, capsys):
    def boom(*_args, **_kwargs):
        raise AssertionError("run_benchmark was called")

    _forbid_construction(monkeypatch)
    monkeypatch.setattr("benchmark.cli.run_benchmark", boom)
    archive = _snapshot(PROJECT_ROOT / "results")
    code = main(["validate", str(EXAMPLE)])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "experiment_id: noise_robustness_sweep_smoke" in out
    assert "models: hopfield_synchronous qam_rot_fullbatch" in out
    assert "dataset.n_bits_list: 2" in out
    assert "dataset.capacity_ratio: 0.3" in out
    assert "perturbation: bernoulli" in out
    assert "perturbation.levels: 0.0" in out
    assert "trial_count: 2" in out
    assert "executable: yes" in out
    assert "noise_robustness_sweep_smoke" in out
    assert _snapshot(PROJECT_ROOT / "results") == archive
    assert not (PROJECT_ROOT / "runs" / "noise_robustness_sweep_smoke").exists()


def test_invalid_configuration_is_rejected(tmp_path, monkeypatch, capsys):
    _forbid_construction(monkeypatch)
    missing = tmp_path / "absent.json"
    assert main(["validate", str(missing)]) == EXIT_INVALID
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert main(["validate", str(broken)]) == EXIT_INVALID
    unknown_model = _example(
        tmp_path, lambda document: document["models"].__setitem__(1, {**document["models"][1], "procedure_id": "qam"})
    )
    assert main(["validate", str(unknown_model)]) == EXIT_INVALID
    unsupported = _example(
        tmp_path,
        lambda document: document["experiment"].__setitem__("protocol", "convergence"),
    )
    assert main(["run", str(unsupported), "--dry-run"]) == EXIT_INVALID
    err = capsys.readouterr().err
    assert "qam" in err
    assert "convergence" in err
    assert "malformed JSON" in err


def test_dry_run_prints_the_plan_and_does_not_execute(monkeypatch, capsys):
    def boom(*_args, **_kwargs):
        raise AssertionError("run_benchmark was called")

    _forbid_construction(monkeypatch)
    monkeypatch.setattr("benchmark.cli.run_benchmark", boom)
    code = main(["run", str(EXAMPLE), "--dry-run"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "experiment_id: noise_robustness_sweep_smoke" in out
    assert "models: hopfield_synchronous qam_rot_fullbatch" in out
    assert "trial_count: 2" in out
    assert "dry_run: configuration was not executed" in out
    assert not (PROJECT_ROOT / "runs" / "noise_robustness_sweep_smoke").exists()


def test_non_executable_configuration_is_not_started(tmp_path, monkeypatch, capsys):
    def boom(*_args, **_kwargs):
        raise AssertionError("run_benchmark was called")

    _forbid_construction(monkeypatch)
    monkeypatch.setattr("benchmark.cli.run_benchmark", boom)

    def mutate(document):
        document["grid"] = "full"
        document["models"][1]["parameters"]["train_steps"] = 300
        document["setting_origin"]["grid"] = "historical"
        document["setting_origin"]["models[1].parameters.train_steps"] = "historical"

    path = _example(tmp_path, mutate)
    assert main(["validate", str(path)]) == EXIT_OK
    validate_out = capsys.readouterr().out
    assert "executable: no" in validate_out
    assert "train_steps" in validate_out
    code = main(["run", str(path), "--output-dir", str(tmp_path / "blocked")])
    run_out = capsys.readouterr().out
    assert code == EXIT_NOT_EXECUTABLE
    assert "executable: no" in run_out
    assert not (tmp_path / "blocked").exists()


def test_archived_output_is_refused(monkeypatch):
    _forbid_construction(monkeypatch)
    archive = _snapshot(PROJECT_ROOT / "results")
    code = main(
        [
            "run",
            str(EXAMPLE),
            "--output-dir",
            str(PROJECT_ROOT / "results" / "noise_robustness_sweep"),
        ]
    )
    assert code == EXIT_NOT_EXECUTABLE
    assert _snapshot(PROJECT_ROOT / "results") == archive


def test_small_run_uses_temporary_output(tmp_path, monkeypatch, capsys):
    _forbid_mains(monkeypatch)
    archive = _snapshot(PROJECT_ROOT / "results")
    numpy_state = np.random.get_state()
    pennylane_state = pnp.random.get_state()
    seen = {}
    from benchmark.cli import run_benchmark as real_run

    def spy(request, *, output_dir=None, overwrite=False):
        seen["plan"] = capsys.readouterr().out
        return real_run(request, output_dir=output_dir, overwrite=overwrite)

    monkeypatch.setattr("benchmark.cli.run_benchmark", spy)
    output_dir = tmp_path / "smoke"
    try:
        code = main(["run", str(EXAMPLE), "--output-dir", str(output_dir)])
    finally:
        np.random.set_state(numpy_state)
        pnp.random.set_state(pennylane_state)

    assert code == EXIT_OK
    plan = seen["plan"]
    assert "experiment_id: noise_robustness_sweep_smoke" in plan
    assert "models: hopfield_synchronous qam_rot_fullbatch" in plan
    assert "dataset.capacity_ratio: 0.3" in plan
    assert "perturbation: bernoulli" in plan
    assert "perturbation.levels: 0.0" in plan
    assert "trial_count: 2" in plan
    assert str(output_dir) in plan
    manifests = list(output_dir.glob("*/manifest.json"))
    assert len(manifests) == 1
    run_dir = manifests[0].parent
    assert (run_dir / "trials.csv").is_file()
    assert (run_dir / "configuration.json").is_file()
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["procedure_ids"] == ["hopfield_synchronous", "qam_rot_fullbatch"]
    assert manifest["main_invoked"] is False
    assert _snapshot(PROJECT_ROOT / "results") == archive
    assert not (PROJECT_ROOT / "runs" / "noise_robustness_sweep_smoke").exists()
