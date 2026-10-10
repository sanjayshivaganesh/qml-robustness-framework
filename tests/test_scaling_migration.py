"""Scaling request, adapters, and one reduced trial.

The historical grid is 5 widths, 30 trials, and 200 training steps. These
tests do not run it, do not call ``experiments.scaling.main``, and do not
write under ``results/``.
"""

import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from pennylane import numpy as pnp

import experiments.scaling as historical
from benchmark.adapters import qam_rot_fullbatch, qam_rot_minibatch, qam_ry_linear
from benchmark.cli import EXIT_NOT_EXECUTABLE, EXIT_OK, main
from benchmark.config_loader import ConfigNotExecutable, load_benchmark_config
from benchmark.output import STATUS_SUCCEEDED, ArchivedResultsRefused
from benchmark.protocols import scaling as protocol
from benchmark.registry import get, prepare
from benchmark.runner import run_benchmark
from src.hopfield import HopfieldAsyncRandom
from src.paths import PROJECT_ROOT

SMOKE = PROJECT_ROOT / "configs" / "scaling.smoke.json"
HISTORICAL = PROJECT_ROOT / "configs" / "scaling.json"
AUDIT = PROJECT_ROOT / "baseline" / "configs" / "original_experiments.json"
ARCHIVE = PROJECT_ROOT / "results" / "scaling" / "raw" / "scaling_sweep.csv"


def _snapshot(root: Path):
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _direct_smoke_trial():
    """Historical order at n=4, trial 0, noise 0, and one training step."""
    pnp.random.seed(123 + 0 + 4 * 100)
    patterns = pnp.random.choice([-1, 1], size=(2, 4))
    hopfield = HopfieldAsyncRandom(4)
    hopfield.train(patterns)
    model = historical.QAM(4, depth=min(6, 4), steps=1, lr=0.05)
    initial = np.array(np.asarray(model.params), dtype=float, copy=True)
    model.train(patterns)
    hopfield_recalled = []
    qam_recalled = []
    for pattern in patterns:
        noisy = pattern.copy()
        noisy[pnp.random.rand(4) < 0.0] *= -1
        hopfield_recalled.append(np.asarray(hopfield.recall(noisy)))
    for pattern in patterns:
        noisy = pattern.copy()
        noisy[pnp.random.rand(4) < 0.0] *= -1
        qam_recalled.append(np.asarray(model.recall(noisy)))
    return {
        "patterns": np.asarray(patterns),
        "initial": initial,
        "depth": model.depth,
        "param_count": int(model.param_count),
        "hopfield": hopfield_recalled,
        "qam": qam_recalled,
    }


def test_historical_configuration_selects_the_scaling_pair():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))["experiments"]["scaling"]
    config = load_benchmark_config(HISTORICAL)
    assert config.executable is False
    assert config.protocol == "scaling"
    assert config.procedure_ids == ("hopfield_async_random", "qam_rot_minibatch")
    assert "qam_ry_linear" not in config.procedure_ids
    assert "qam_rot_fullbatch" not in config.procedure_ids
    assert list(config.n_bits_list) == audit["n_bits_list"] == list(historical.n_bits_list)
    assert list(config.noise_levels) == [float(level) for level in audit["noise_levels"]]
    assert list(config.noise_levels) == [float(level) for level in historical.noise_levels]
    assert list(config.trial_ids) == list(range(audit["trials"])) == list(range(historical.trials))
    assert config.steps == audit["qam"]["steps"] == 200
    assert config.lr == audit["qam"]["lr"] == 0.05
    assert config.seed_base == audit["random_seed_base"] == historical.RANDOM_SEED_BASE == 123
    assert len(config.n_bits_list) * len(config.trial_ids) == 150
    hopfield = get("hopfield_async_random")
    qam = get("qam_rot_minibatch")
    assert hopfield.adapter.identity.source == "src.hopfield.HopfieldAsyncRandom"
    assert qam.adapter.implementation is historical.QAM
    assert qam.adapter.implementation is not qam_ry_linear.train_implementation
    assert qam.adapter.implementation is not qam_rot_fullbatch.implementation
    factory, args, kwargs = prepare(
        "qam_rot_minibatch",
        {"n_bits": 4, "depth": min(6, 4), "steps": 200, "lr": 0.05},
    )
    assert factory is qam_rot_minibatch.build
    assert args == (4,)
    assert kwargs == {"depth": 4, "steps": 200, "lr": 0.05}
    request = config.to_runner_request(allow_historical_grid=True)
    assert protocol.is_historical_scaling_request(request)
    assert protocol.trial_seed(123, 0, 4) == 523
    assert protocol.n_patterns_for(4) == 2
    assert protocol.n_patterns_for(16) == 6
    with pytest.raises(ConfigNotExecutable):
        config.to_runner_request()
    train_source = inspect.getsource(historical.QAM.train)
    assert "min(2, len(patterns))" in train_source
    assert "1e-4" in train_source
    assert "patience_counter >= 15" in train_source
    destination = config.resolved_output_dir().resolve()
    archive = (PROJECT_ROOT / "results").resolve()
    assert destination == (PROJECT_ROOT / "runs" / "scaling").resolve()
    assert not destination.is_relative_to(archive)


def test_smoke_configuration_is_a_reduced_budget():
    config = load_benchmark_config(SMOKE)
    assert config.executable is True
    assert config.experiment_id == "scaling_smoke"
    assert config.grid == "smoke"
    assert config.n_bits_list == (4,)
    assert config.noise_levels == (0.0,)
    assert config.trial_ids == (0,)
    assert config.steps == 1
    assert config.steps != 200
    request = config.to_runner_request()
    assert request.steps == 1
    assert not protocol.is_historical_scaling_request(request)


def test_historical_grid_is_not_started_without_the_flag(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("scaling QAM was constructed")

    monkeypatch.setattr(historical.QAM, "__init__", boom)
    monkeypatch.setattr(historical, "main", boom)
    config = load_benchmark_config(HISTORICAL)
    archive = _snapshot(PROJECT_ROOT / "results")
    with pytest.raises(ConfigNotExecutable):
        run_benchmark(config, output_dir=tmp_path)
    with pytest.raises(protocol.HistoricalGridRefused):
        protocol.run_scaling(protocol.historical_scaling_request(), output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert _snapshot(PROJECT_ROOT / "results") == archive


def test_small_run_matches_the_historical_order_and_stays_outside_the_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(historical, "main", lambda: (_ for _ in ()).throw(AssertionError("main")))
    numpy_state = np.random.get_state()
    pennylane_state = pnp.random.get_state()
    archive = _snapshot(PROJECT_ROOT / "results")
    try:
        direct = _direct_smoke_trial()
        np.random.set_state(numpy_state)
        pnp.random.set_state(pennylane_state)
        config = load_benchmark_config(SMOKE)
        result = run_benchmark(config, output_dir=tmp_path)
    finally:
        np.random.set_state(numpy_state)
        pnp.random.set_state(pennylane_state)

    assert result["execution_status"] == STATUS_SUCCEEDED
    assert result["comparable_to_archive"] is False
    run_dir = Path(result["output_dir"])
    assert run_dir.resolve().is_relative_to(tmp_path.resolve())
    assert _snapshot(PROJECT_ROOT / "results") == archive
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["historical_steps"] == 200
    assert manifest["smoke_steps"] == 1
    assert manifest["plotting"] is False
    assert manifest["main_invoked"] is False
    assert manifest["resumable"] is False
    assert manifest["comparable_to_archive"] is False
    assert manifest["models"] == ["hopfield_async_random", "qam_rot_minibatch"]
    assert manifest["rng"]["exact_reproducibility_claimed"] is False
    assert manifest["rng"]["internal_resets"] == []
    assert manifest["rng"]["configured_seeds"]["seed_base"] == 123
    assert manifest["rng"]["seeds_applied"][0]["values"] == [
        {"n_bits": 4, "trial_id": 0, "value": 523}
    ]
    assert manifest["rng"]["unseeded_procedures"][0]["name"] == "experiments.scaling.bootstrap_ci"
    with (run_dir / "trials.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["procedure_id"] for row in rows] == [
        "hopfield_async_random",
        "qam_rot_minibatch",
    ]
    assert [row["historical_model_label"] for row in rows] == ["Hopfield", "QAM"]
    assert rows[0]["n_bits"] == "4"
    assert rows[0]["m"] == "2"
    assert rows[0]["trial_id"] == "0"
    assert rows[0]["seed"] == "523"
    assert rows[0]["noise_p"] == "0.0"
    assert rows[0]["params_count"] == "16"
    assert rows[1]["params_count"] == "48"
    assert rows[0]["comparable_to_archive"] == "False"
    payload = json.loads((run_dir / "trials.json").read_text(encoding="utf-8"))
    trial = payload["trials"][0]
    assert trial["depth_used"] == direct["depth"] == 4
    assert trial["steps"] == 1
    assert np.array_equal(trial["patterns"], direct["patterns"])
    assert np.allclose(trial["init_params"], direct["initial"])
    arms = {arm["procedure_id"]: arm for arm in trial["noise"][0]["arms"]}
    assert np.array_equal(arms["hopfield_async_random"]["recalled_patterns"], direct["hopfield"])
    assert np.array_equal(arms["qam_rot_minibatch"]["recalled_patterns"], direct["qam"])
    assert direct["param_count"] == 48


def test_archived_scaling_table_is_not_rewritten_or_treated_as_reproduced():
    text = ARCHIVE.read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.strip()]
    assert rows[0].split(",")[:6] == ["model", "n_bits", "m", "trial_id", "full_acc", "bit_acc"]
    assert len(rows) == 1201
    assert "plt" not in Path(protocol.__file__).read_text(encoding="utf-8")


def test_command_line_distinguishes_the_smoke_from_the_historical_grid(monkeypatch, capsys):
    def boom(*_args, **_kwargs):
        raise AssertionError("scaling was started")

    monkeypatch.setattr("benchmark.cli.run_benchmark", boom)
    monkeypatch.setattr(historical.QAM, "__init__", boom)
    assert main(["validate", str(SMOKE)]) == EXIT_OK
    smoke_out = capsys.readouterr().out
    assert "models: hopfield_async_random qam_rot_minibatch" in smoke_out
    assert "steps: 1" in smoke_out
    assert "executable: yes" in smoke_out
    assert main(["validate", str(HISTORICAL)]) == EXIT_OK
    historical_out = capsys.readouterr().out
    assert "dataset.n_bits_list: 4 6 8 12 16" in historical_out
    assert "steps: 200" in historical_out
    assert "executable: no" in historical_out
    assert main(["run", str(HISTORICAL), "--dry-run"]) == EXIT_OK
    assert "full_grid: not started" in capsys.readouterr().out
    assert main(["run", str(HISTORICAL)]) == EXIT_NOT_EXECUTABLE
    assert main(["run", str(SMOKE), "--full-grid", "--dry-run"]) == EXIT_NOT_EXECUTABLE
    assert "--full-grid applies only" in capsys.readouterr().err


def test_output_under_the_archive_is_refused(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("scaling QAM was constructed")

    monkeypatch.setattr(historical.QAM, "__init__", boom)
    archive = _snapshot(PROJECT_ROOT / "results")
    config = load_benchmark_config(SMOKE)
    with pytest.raises(ArchivedResultsRefused):
        run_benchmark(config, output_dir=PROJECT_ROOT / "results" / "scaling")
    assert _snapshot(PROJECT_ROOT / "results") == archive
