"""Lightweight paired noise-sweep smoke. Does not call the historical main."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from pennylane import numpy as pnp

import experiments.noise_robustness_sweep as historical
from benchmark.adapters import qam_rot_fullbatch
from benchmark.output import ArchivedResultsRefused
from benchmark.protocols import noise_robustness_sweep as protocol
from benchmark.runner import UnknownProtocol, run_benchmark
from benchmark.scoring import gaussian_ci_population
from src.paths import PROJECT_ROOT


def _snapshot(root: Path):
    return {
        path.relative_to(root).as_posix(): path.stat().st_mtime_ns
        for path in root.rglob("*")
        if path.is_file()
    }


def _forbid_training(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("QAM construction was called")

    monkeypatch.setattr(protocol.qam_rot_fullbatch, "build", boom)


def test_pattern_count_uses_the_noise_sweep_formula():
    assert protocol.n_patterns_for(2, 0.3) == 1
    assert protocol.n_patterns_for(4, 0.3) == max(1, int(0.3 * 4))
    assert protocol.n_patterns_for(10, 0.3) == max(1, int(0.3 * 10))
    assert historical.capacity_ratio == 0.3


def test_historical_grid_is_refused_before_training(monkeypatch):
    _forbid_training(monkeypatch)
    oversized = protocol.NoiseSweepRequest(
        n_bits_list=tuple(historical.n_bits_list),
        corruption_rates=tuple(historical.corruption_rates),
        seeds=tuple(range(historical.trials)),
        capacity_ratio=historical.capacity_ratio,
        train_steps=1,
        grid="smoke",
    )
    labeled_full = protocol.NoiseSweepRequest(
        n_bits_list=(2,),
        corruption_rates=(0.0,),
        seeds=(0,),
        train_steps=1,
        grid="full",
    )
    long_train = protocol.NoiseSweepRequest(
        n_bits_list=(2,),
        corruption_rates=(0.0,),
        seeds=(0,),
        train_steps=historical.train_steps,
        grid="smoke",
    )
    for request in (oversized, labeled_full, long_train):
        with pytest.raises(protocol.HistoricalGridRefused):
            run_benchmark(request)


def test_archive_output_is_refused_before_training(monkeypatch):
    _forbid_training(monkeypatch)
    archive = PROJECT_ROOT / "results"
    before = _snapshot(archive)
    with pytest.raises(ArchivedResultsRefused):
        run_benchmark(
            protocol.lightweight_request(),
            output_dir=archive / "noise_robustness_sweep",
        )
    assert _snapshot(archive) == before


def test_runner_rejects_a_request_that_is_not_this_protocol():
    with pytest.raises(UnknownProtocol):
        run_benchmark(object())


def test_lightweight_smoke_schema_reseed_and_isolated_output(tmp_path):
    archive = PROJECT_ROOT / "results"
    archived_csv = archive / "noise_robustness_sweep" / "summary" / "noise_robustness_sweep.csv"
    archived_bytes = archived_csv.read_bytes()
    before = _snapshot(archive)
    state = np.random.get_state()
    try:
        result = run_benchmark(
            protocol.lightweight_request(),
            output_dir=tmp_path / "noise-sweep-smoke",
        )
    finally:
        np.random.set_state(state)

    assert _snapshot(archive) == before
    assert archived_csv.read_bytes() == archived_bytes
    assert result["main_invoked"] is False
    assert result["comparable_to_archive"] is False
    assert result["grid"] == "smoke"
    assert result["procedure_ids"] == ["hopfield_synchronous", "qam_rot_fullbatch"]
    assert "qam" not in result["procedure_ids"]
    assert result["historical_train_steps"] == 300
    assert len(result["rng_limitations"]) >= 3

    assert len(result["cells"]) == 2
    init_params = []
    patterns = []
    for cell in result["cells"]:
        assert cell["n_bits"] == 2
        assert cell["n_patterns"] == 1
        assert cell["capacity_ratio"] == 0.3
        assert cell["pattern_generator"] == "numpy.random"
        assert cell["perturbation_id"] == "bernoulli"
        assert cell["perturbation_level"] == 0.0
        assert len(cell["patterns"]) == 1 and len(cell["patterns"][0]) == 2
        patterns.append(cell["patterns"])
        ids = [arm["procedure_id"] for arm in cell["arms"]]
        assert ids == ["hopfield_synchronous", "qam_rot_fullbatch"]
        for arm in cell["arms"]:
            assert arm["procedure_id"] != "qam"
            assert set(arm["metrics"]) == {"full_accuracy", "bit_accuracy"}
            assert set(arm["timings"]) == {
                "train_time_seconds",
                "mean_recall_time_per_bit",
            }
            assert 0.0 <= arm["metrics"]["full_accuracy"] <= 1.0
            assert 0.0 <= arm["metrics"]["bit_accuracy"] <= 1.0
            for query in arm["queries"]:
                assert query["n_flips"] == 0
        hopfield, qam = cell["arms"]
        assert hopfield["constructor_reseed"] is None
        assert hopfield["train_steps"] is None
        assert qam["constructor_reseed"] == 42
        assert qam["train_steps"] == 1
        assert qam["depth_used"] == 2
        assert qam["lr"] == 0.05
        assert qam["source"] == "experiments.noise_robustness_sweep.QAM"
        init_params.append(np.asarray(qam["init_params"], dtype=float))

    assert patterns[0] != patterns[1]
    assert np.allclose(init_params[0], init_params[1])
    _assert_draws_match_historical_order(result, init_params[0])

    by_id = {row["procedure_id"]: row for row in result["aggregates"]}
    assert set(by_id) == {"hopfield_synchronous", "qam_rot_fullbatch"}
    for procedure_id, row in by_id.items():
        scores = [
            arm["metrics"]["full_accuracy"]
            for cell in result["cells"]
            for arm in cell["arms"]
            if arm["procedure_id"] == procedure_id
        ]
        mean, std, half = gaussian_ci_population(scores)
        assert row["n_trials"] == 2
        assert row["full_accuracy_mean"] == float(mean)
        assert row["full_accuracy_std"] == float(std)
        assert row["full_accuracy_ci"] == float(half)
        assert row["comparable_to_archive"] is False

    run_dir = Path(result["output_dir"])
    assert run_dir.is_relative_to(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["comparable_to_archive"] is False
    assert manifest["main_invoked"] is False
    assert manifest["procedure_ids"] == ["hopfield_synchronous", "qam_rot_fullbatch"]
    assert manifest["smoke_train_steps"] == 1
    assert manifest["historical_train_steps"] == 300
    assert manifest["rng"]["exact_reproducibility_claimed"] is False
    assert manifest["rng"]["configured_seeds"]["trial_seeds"] == [0, 1]
    assert manifest["rng"]["seeds_applied"][0]["values"] == [0, 1]
    assert manifest["rng"]["internal_resets"][0]["value"] == 42
    assert manifest["rng"]["internal_resets"][0]["configured"] is False
    assert manifest["execution_status"] == "succeeded"
    assert manifest["output_class"] == "framework_run"
    assert manifest["run_id"] == run_dir.name
    assert manifest["error"] is None
    stored = json.loads((run_dir / "configuration.json").read_text(encoding="utf-8"))
    assert stored["train_steps"] == 1
    assert stored["capacity_ratio"] == 0.3
    with (run_dir / "trials.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(protocol.TRIAL_COLUMNS)
    assert {row["procedure_id"] for row in rows} == {
        "hopfield_synchronous",
        "qam_rot_fullbatch",
    }
    assert all(row["comparable_to_archive"] == "False" for row in rows)
    assert protocol.COLUMN_ROLES["full_accuracy"] == "metric"
    assert protocol.COLUMN_ROLES["train_time_seconds"] == "timing"
    assert "model" not in protocol.TRIAL_COLUMNS

    source = (PROJECT_ROOT / "benchmark" / "protocols" / "noise_robustness_sweep.py").read_text()
    runner = (PROJECT_ROOT / "benchmark" / "runner.py").read_text()
    assert ".main(" not in source and ".main(" not in runner
    assert "savefig" not in source and "Pool(" not in source


def _assert_draws_match_historical_order(result, init_params):
    """Patterns follow the trial seed. QAM init follows the constructor reset."""
    for cell in result["cells"]:
        np.random.seed(cell["trial_seed"])
        pnp.random.seed(cell["trial_seed"])
        expected = np.random.choice([-1, 1], size=(cell["n_patterns"], cell["n_bits"]))
        assert cell["patterns"] == expected.tolist()
    pnp.random.seed(qam_rot_fullbatch.rng_contract.reseed_value)
    expected_init = 0.01 * np.random.randn(*np.shape(init_params))
    assert np.allclose(init_params, expected_init)
