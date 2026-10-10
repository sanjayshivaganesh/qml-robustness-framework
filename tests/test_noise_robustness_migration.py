"""Historical noise-sweep request, wired to the existing adapters.

The 600-cell, 300-step grid is not executed. One controlled cell uses a
single Adam step. Nothing is written under results/, and no experiment
main is called.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from pennylane import numpy as pnp

import experiments.noise_robustness_sweep as historical
from benchmark.adapters import qam_rot_fullbatch, qam_ry_linear
from benchmark.cli import EXIT_NOT_EXECUTABLE, EXIT_OK, main
from benchmark.config_loader import ConfigNotExecutable, load_benchmark_config
from benchmark.output import STATUS_FAILED, ArchivedResultsRefused
from benchmark.protocols import noise_robustness_sweep as protocol
from benchmark.registry import get, prepare
from benchmark.runner import run_benchmark
from benchmark.scoring import batch_recall, gaussian_ci_population
from src.data_generation import corrupt_bernoulli
from src.hopfield import Hopfield
from src.metrics import mean_std_ci
from src.paths import PROJECT_ROOT

CONFIG = PROJECT_ROOT / "configs" / "noise_robustness_sweep.json"
AUDIT = PROJECT_ROOT / "baseline" / "configs" / "original_experiments.json"
ARCHIVE = (
    PROJECT_ROOT
    / "results"
    / "noise_robustness_sweep"
    / "summary"
    / "noise_robustness_sweep.csv"
)


def _snapshot(root: Path):
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _pooled(recalled, patterns):
    patterns = np.asarray(patterns)
    recalled = [np.asarray(row) for row in recalled]
    count = len(patterns)
    width = patterns.shape[1]
    full = sum(np.array_equal(row, target) for row, target in zip(recalled, patterns)) / count
    bits = sum(np.sum(row == target) for row, target in zip(recalled, patterns)) / (count * width)
    return full, bits


def _direct_cell(n_bits, corruption, seed, train_steps):
    np.random.seed(seed)
    pnp.random.seed(seed)
    count = max(1, int(historical.capacity_ratio * n_bits))
    patterns = np.random.choice([-1, 1], size=(count, n_bits))
    hopfield = Hopfield(n_bits)
    hopfield.train(patterns)
    hopfield_queries = []
    hopfield_recalled = []
    for pattern in patterns:
        query = corrupt_bernoulli(pattern, corruption)
        hopfield_queries.append(np.asarray(query))
        recalled, _timing = hopfield.recall_timed(query)
        hopfield_recalled.append(np.asarray(recalled))
    model = historical.QAM(n_bits, train_steps=train_steps)
    initial = np.array(np.asarray(model.params), dtype=float, copy=True)
    model.train(patterns)
    qam_queries = []
    qam_recalled = []
    for pattern in patterns:
        query = corrupt_bernoulli(pattern, corruption)
        qam_queries.append(np.asarray(query))
        recalled, _timing = model.recall(query)
        qam_recalled.append(np.asarray(recalled))
    return {
        "patterns": np.asarray(patterns),
        "hopfield_queries": hopfield_queries,
        "hopfield_recalled": hopfield_recalled,
        "initial": initial,
        "qam_queries": qam_queries,
        "qam_recalled": qam_recalled,
        "depth": model.depth,
        "lr": model.lr,
        "train_steps": model.train_steps,
    }


def test_configuration_selects_the_historical_pair_and_grid():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))["experiments"]["noise_robustness_sweep"]
    config = load_benchmark_config(CONFIG)
    assert config.executable is False
    assert config.experiment_id == "noise_robustness_sweep"
    assert config.protocol == "noise_robustness_sweep"
    assert config.procedure_ids == ("hopfield_synchronous", "qam_rot_fullbatch")
    assert "qam_ry_linear" not in config.procedure_ids
    assert config.grid == "full"
    assert list(config.n_bits_list) == audit["n_bits_list"] == list(historical.n_bits_list)
    assert list(config.corruption_levels) == audit["corruption_rates"] == list(historical.corruption_rates)
    assert list(config.trial_seeds) == list(range(audit["trials"])) == list(range(historical.trials))
    assert config.train_steps == audit["qam"]["train_steps"] == historical.train_steps
    assert config.capacity_ratio == audit["capacity_ratio"] == historical.capacity_ratio
    assert len(config.n_bits_list) * len(config.corruption_levels) * len(config.trial_seeds) == 600
    hopfield = get("hopfield_synchronous")
    qam = get("qam_rot_fullbatch")
    assert hopfield.adapter.identity.source == "src.hopfield.Hopfield"
    assert qam.adapter.identity.source == "experiments.noise_robustness_sweep.QAM"
    assert qam.adapter.implementation is historical.QAM
    assert qam.adapter.implementation is not qam_ry_linear.train_implementation
    hopfield_factory, hopfield_args, hopfield_kwargs = prepare(
        "hopfield_synchronous", {"n_bits": 4, "steps": 10}
    )
    qam_factory, qam_args, qam_kwargs = prepare(
        "qam_rot_fullbatch",
        {"n_bits": 4, "train_steps": 300, "depth_argument": 6, "lr": 0.05},
    )
    assert hopfield_factory is hopfield.adapter.build
    assert hopfield_args == (4,)
    assert hopfield_kwargs == {}
    assert qam_factory is qam_rot_fullbatch.build
    assert qam_args == (4,)
    assert qam_kwargs == {"train_steps": 300}
    request = config.to_runner_request(allow_historical_grid=True)
    assert protocol.is_historical_noise_sweep_request(request)
    with pytest.raises(ConfigNotExecutable):
        config.to_runner_request()
    destination = config.resolved_output_dir().resolve()
    archive = (PROJECT_ROOT / "results").resolve()
    assert destination == (PROJECT_ROOT / "runs" / "noise_robustness_sweep").resolve()
    assert archive not in destination.parents
    assert not destination.is_relative_to(archive)


def test_historical_grid_is_not_started_without_the_explicit_flag(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("QAM was constructed")

    monkeypatch.setattr(historical.QAM, "__init__", boom)
    monkeypatch.setattr(historical, "main", boom)
    config = load_benchmark_config(CONFIG)
    archive = _snapshot(PROJECT_ROOT / "results")
    with pytest.raises(ConfigNotExecutable):
        run_benchmark(config, output_dir=tmp_path)
    with pytest.raises(protocol.HistoricalGridRefused):
        protocol.run_noise_sweep(protocol.historical_noise_sweep_request(), output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert _snapshot(PROJECT_ROOT / "results") == archive


def test_manifest_records_the_historical_seeds_without_running_cells(tmp_path, monkeypatch):
    def boom(_request):
        raise RuntimeError("stopped before cells")

    monkeypatch.setattr(protocol, "_execute", boom)
    monkeypatch.setattr(historical, "main", boom)
    archive = _snapshot(PROJECT_ROOT / "results")
    config = load_benchmark_config(CONFIG)
    with pytest.raises(RuntimeError, match="stopped before cells"):
        run_benchmark(config, output_dir=tmp_path, allow_historical_grid=True)
    with pytest.raises(ArchivedResultsRefused):
        run_benchmark(
            config,
            output_dir=PROJECT_ROOT / "results" / "noise_robustness_sweep",
            allow_historical_grid=True,
        )
    manifests = list(tmp_path.glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["execution_status"] == STATUS_FAILED
    assert manifest["comparable_to_archive"] is False
    assert manifest["resumable"] is False
    assert manifest["main_invoked"] is False
    assert manifest["historical_grid"] is True
    assert manifest["plotting"] is False
    assert manifest["models"] == ["hopfield_synchronous", "qam_rot_fullbatch"]
    assert manifest["rng"]["exact_reproducibility_claimed"] is False
    assert manifest["rng"]["configured_seeds"]["trial_seeds"] == list(range(30))
    assert manifest["rng"]["seeds_applied"][0]["values"] == list(range(30))
    assert manifest["rng"]["internal_resets"][0]["procedure_id"] == "qam_rot_fullbatch"
    assert manifest["rng"]["internal_resets"][0]["value"] == 42
    assert manifest["rng"]["internal_resets"][0]["configured"] is False
    assert manifests[0].resolve().is_relative_to(tmp_path.resolve())
    assert _snapshot(PROJECT_ROOT / "results") == archive


def test_one_cell_matches_the_historical_order_at_one_adam_step():
    numpy_state = np.random.get_state()
    pennylane_state = pnp.random.get_state()
    try:
        direct = _direct_cell(4, 0.0, 0, 1)
        np.random.set_state(numpy_state)
        pnp.random.set_state(pennylane_state)
        cell = protocol._run_cell(
            4,
            0.0,
            0,
            capacity_ratio=0.3,
            train_steps=1,
            score_batch=batch_recall,
        )
    finally:
        np.random.set_state(numpy_state)
        pnp.random.set_state(pennylane_state)

    assert np.array_equal(cell["patterns"], direct["patterns"])
    assert cell["n_patterns"] == 1
    arms = {arm["procedure_id"]: arm for arm in cell["arms"]}
    assert list(arms) == ["hopfield_synchronous", "qam_rot_fullbatch"]
    assert arms["qam_rot_fullbatch"]["source"] == "experiments.noise_robustness_sweep.QAM"
    assert arms["qam_rot_fullbatch"]["train_steps"] == 1
    assert arms["qam_rot_fullbatch"]["depth_used"] == direct["depth"] == 4
    assert arms["qam_rot_fullbatch"]["lr"] == direct["lr"] == 0.05
    assert arms["qam_rot_fullbatch"]["constructor_reseed"] == 42
    assert np.allclose(arms["qam_rot_fullbatch"]["init_params"], direct["initial"])
    for procedure_id, queries, recalled in (
        ("hopfield_synchronous", direct["hopfield_queries"], direct["hopfield_recalled"]),
        ("qam_rot_fullbatch", direct["qam_queries"], direct["qam_recalled"]),
    ):
        assert np.array_equal(arms[procedure_id]["queries"][0]["corrupted"], queries[0])
        assert np.array_equal(arms[procedure_id]["recalled_patterns"], recalled)
        full, bits = _pooled(recalled, direct["patterns"])
        assert arms[procedure_id]["metrics"]["full_accuracy"] == pytest.approx(full)
        assert arms[procedure_id]["metrics"]["bit_accuracy"] == pytest.approx(bits)


def test_metrics_match_the_historical_reductions_on_fixed_arrays():
    patterns = np.array([[1, -1, 1, -1], [1, 1, -1, -1]])
    recalled = np.array([[1, -1, 1, 1], [1, 1, -1, -1]])
    full, bits = batch_recall(recalled, patterns)
    pooled_full, pooled_bits = _pooled(recalled, patterns)
    assert full == pytest.approx(pooled_full) == pytest.approx(0.5)
    assert bits == pytest.approx(pooled_bits) == pytest.approx(0.875)
    scores = [1.0, 0.0, 1.0, 0.5]
    mean, std, half_width = gaussian_ci_population(scores)
    historical_mean, historical_std, historical_half_width = mean_std_ci(scores)
    assert mean == pytest.approx(historical_mean)
    assert std == pytest.approx(historical_std)
    assert half_width == pytest.approx(historical_half_width)
    assert half_width == pytest.approx(1.96 * np.std(scores) / np.sqrt(len(scores)))


def test_archived_summary_is_inspected_and_not_treated_as_reproduced():
    text = ARCHIVE.read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.strip()]
    header = rows[0].split(",")
    assert header[:4] == ["n_bits", "corruption", "hopfield_full_acc_mean", "hopfield_full_acc_std"]
    assert "qam_full_acc_mean" in header
    assert len(rows) == 21
    assert "plt" not in Path(protocol.__file__).read_text(encoding="utf-8")


def test_command_line_prints_the_historical_plan_and_does_not_start_it(monkeypatch, capsys):
    def boom(*_args, **_kwargs):
        raise AssertionError("the historical grid was started")

    monkeypatch.setattr("benchmark.cli.run_benchmark", boom)
    monkeypatch.setattr(historical.QAM, "__init__", boom)
    monkeypatch.setattr(historical, "main", boom)
    assert main(["validate", str(CONFIG)]) == EXIT_OK
    validate_out = capsys.readouterr().out
    assert "models: hopfield_synchronous qam_rot_fullbatch" in validate_out
    assert "dataset.n_bits_list: 4 6 8 10" in validate_out
    assert "trial_count: 30" in validate_out
    assert "train_steps: 300" in validate_out
    assert "grid: full" in validate_out
    assert "executable: no" in validate_out
    assert main(["run", str(CONFIG), "--dry-run"]) == EXIT_OK
    dry_out = capsys.readouterr().out
    assert "dry_run: configuration was not executed" in dry_out
    assert "full_grid: not started" in dry_out
    assert main(["run", str(CONFIG)]) == EXIT_NOT_EXECUTABLE
    assert "without --full-grid" in capsys.readouterr().err


def test_full_grid_flag_is_forwarded_and_a_smoke_file_rejects_it(monkeypatch, capsys):
    seen = {}

    def spy(request, *, output_dir=None, overwrite=False, allow_historical_grid=False):
        seen["allow"] = allow_historical_grid
        seen["train_steps"] = request.train_steps
        seen["procedure_ids"] = request.procedure_ids
        return {
            "output_dir": "not-written",
            "procedure_ids": request.procedure_ids,
            "comparable_to_archive": False,
        }

    monkeypatch.setattr("benchmark.cli.run_benchmark", spy)
    monkeypatch.setattr(historical.QAM, "__init__", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("QAM")))
    code = main(["run", str(CONFIG), "--full-grid", "--output-dir", "unused"])
    assert code == EXIT_OK
    assert seen == {
        "allow": True,
        "train_steps": 300,
        "procedure_ids": ("hopfield_synchronous", "qam_rot_fullbatch"),
    }
    smoke = PROJECT_ROOT / "configs" / "noise_robustness_sweep.smoke.json"
    assert main(["run", str(smoke), "--full-grid", "--dry-run"]) == EXIT_NOT_EXECUTABLE
    assert "--full-grid applies only" in capsys.readouterr().err
