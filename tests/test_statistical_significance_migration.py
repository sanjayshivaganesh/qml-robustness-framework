"""Statistical-significance migration. Does not start the 720-task grid.

The one-cell run uses the module's 100 epochs. Its Wilcoxon value is the
failure sentinel for a single observation. It is not the archived 30-seed
p-value.
"""

import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from pennylane import numpy as pnp
from scipy.stats import wilcoxon

import experiments.statistical_significance as historical
from benchmark.cli import EXIT_NOT_EXECUTABLE, EXIT_OK, main
from benchmark.config_loader import ConfigNotExecutable, ConfigValidationError, load_benchmark_config
from benchmark.output import ArchivedResultsRefused
from benchmark.protocols.statistical_significance import (
    TRIAL_COLUMNS,
    historical_request,
    historical_task_count,
    is_historical_request,
    lightweight_request,
    n_patterns_for,
    run_statistical_significance,
    summarize_groups,
)
from benchmark.registry import InvalidModelParameters, get, prepare
from benchmark.runner import run_benchmark
from benchmark.scoring import WILCOXON_FAILURE, paired_bootstrap_diff, selected
from src.metrics import cliffs_delta, pattern_accuracy_pair
from src.paths import PROJECT_ROOT
from src.qam import qam_recall as ry_linear_recall

SMOKE = PROJECT_ROOT / "configs" / "statistical_significance.smoke.json"
HISTORICAL = PROJECT_ROOT / "configs" / "statistical_significance.json"
ARCHIVE_RAW = PROJECT_ROOT / "results" / "statistical_significance" / "raw" / "statistical_rows.csv"
ARCHIVE_SUMMARY = (
    PROJECT_ROOT / "results" / "statistical_significance" / "summary" / "statistical_summary.csv"
)


def _states():
    return np.random.get_state(), pnp.random.get_state()


def _restore(states):
    numpy_state, pennylane_state = states
    np.random.set_state(numpy_state)
    pnp.random.set_state(pennylane_state)


def _archive_bytes():
    return ARCHIVE_RAW.read_bytes(), ARCHIVE_SUMMARY.read_bytes()


def test_historical_configuration_selects_the_statistical_pair():
    loaded = load_benchmark_config(HISTORICAL)
    audit = json.loads(
        (PROJECT_ROOT / "baseline" / "configs" / "original_experiments.json").read_text(
            encoding="utf-8"
        )
    )["experiments"]["statistical_significance"]
    assert loaded.protocol == "statistical_significance"
    assert loaded.procedure_ids == ("hopfield_synchronous", "qam_paulix")
    assert loaded.n_bits_list == tuple(historical.N_BITS_LIST) == tuple(audit["n_bits_list"])
    assert loaded.capacity_ratios == tuple(float(value) for value in historical.CAPACITY_RATIOS)
    assert loaded.corruption_levels == tuple(float(value) for value in historical.CORRUPTION_LEVELS)
    assert loaded.seeds == tuple(range(30))
    assert loaded.bootstrap_samples == historical.BOOTSTRAP_SAMPLES == audit["bootstrap_samples"] == 5000
    assert loaded.executable is False
    assert historical_task_count() == 720 == loaded.n_bits_list.__len__() * 4 * 2 * 30
    tasks = tuple(historical_request().tasks())
    assert tasks[0] == (8, 0.2, 0.1, 0)
    assert tasks[30] == (8, 0.2, 0.2, 0)
    assert sum(1 for task in tasks if task[3] == 0) == 24
    assert n_patterns_for(8, 0.2) == max(1, int(0.2 * 8)) == 1
    assert n_patterns_for(16, 0.5) == max(1, int(0.5 * 16)) == 8

    registration = get("qam_paulix")
    assert registration.implementation is historical.build_qam
    assert registration.adapter.train_implementation is historical.train_qam
    assert registration.adapter.recall_implementation is historical.qam_recall
    assert registration.adapter.recall_implementation is not ry_linear_recall
    assert get("hopfield_synchronous").implementation.__name__ == "Hopfield"
    assert get("qam_rot_fullbatch").implementation is not historical.build_qam
    assert get("qam_rot_minibatch").implementation is not historical.build_qam
    prepare("qam_paulix", {"n_bits": 8})
    with pytest.raises(InvalidModelParameters):
        prepare("qam_paulix", {"n_bits": 8, "epochs": 1})
    train_source = inspect.getsource(historical.train_qam)
    assert "normal(0, 0.1" in train_source
    assert "EPOCHS" in train_source
    assert "seed(" not in train_source
    bootstrap_source = inspect.getsource(historical.bootstrap_diff)
    assert "seed(" not in bootstrap_source
    assert "randint" in bootstrap_source
    assert "seed(" not in inspect.getsource(summarize_groups)
    with pytest.raises(ConfigNotExecutable):
        loaded.to_runner_request()
    allowed = loaded.to_runner_request(allow_historical_grid=True)
    assert is_historical_request(allowed)
    assert "results" not in Path(loaded.output_destination).parts


def test_smoke_configuration_keeps_the_module_training_budget():
    loaded = load_benchmark_config(SMOKE)
    assert loaded.executable is True
    assert loaded.n_bits_list == (8,)
    assert loaded.capacity_ratios == (0.2,)
    assert loaded.corruption_levels == (0.1,)
    assert loaded.seeds == (0,)
    assert loaded.bootstrap_samples == 5000
    request = loaded.to_runner_request()
    assert request.grid == "smoke"
    assert not is_historical_request(request)
    assert lightweight_request().seeds == (0,)
    assert historical.EPOCHS == 100
    assert historical.DEPTH == 10
    assert float(historical.LR) == 0.02


def test_statistics_match_the_historical_helpers_on_controlled_rows():
    rows = _controlled_rows()
    states = _states()
    try:
        np.random.seed(7)
        summaries, effects = summarize_groups(rows, 20, grid="controlled")
        np.random.seed(7)
        direct = _direct_summary(rows, 20)
    finally:
        _restore(states)

    assert [row["n_bits"] for row in summaries] == [8, 12]
    assert summaries[0]["sample_size"] == 4
    assert summaries[0]["historical_sample_size"] == 30
    assert summaries[0]["bootstrap_seeded"] is False
    for produced, expected in zip(summaries, direct):
        assert produced["mean_diff_full"] == pytest.approx(expected["mean_diff_full"])
        assert produced["full_ci_2p5"] == pytest.approx(expected["full_ci_2p5"])
        assert produced["full_ci_97p5"] == pytest.approx(expected["full_ci_97p5"])
        assert produced["full_p_value"] == pytest.approx(expected["full_p_value"])
        assert produced["bit_ci_2p5"] == pytest.approx(expected["bit_ci_2p5"])
        assert produced["bit_p_value"] == pytest.approx(expected["bit_p_value"])
        assert produced["comparable_to_archive"] is False
    assert effects[0]["delta_full"] == pytest.approx(direct[0]["delta_full"])
    assert effects[0]["full_magnitude"] == direct[0]["full_magnitude"]
    assert effects[0]["x_procedure_id"] == "hopfield_synchronous"
    assert effects[0]["y_procedure_id"] == "qam_paulix"
    assert effects[0]["computed_by_experiment_main"] is False
    assert effects[0]["delta_full"] == pytest.approx(
        cliffs_delta(
            [0.0, 0.5, 1.0, 0.25],
            [1.0, 1.0, 0.0, 0.75],
        )
    )

    left = np.array([1, -1, 1])
    right = np.array([1, 1, 1])
    other = np.array([1, 1, 1])
    full_scores = []
    bit_scores = []
    for pred, true in ((left, right), (other, right)):
        full, bit = selected("statistical_significance", "pattern_accuracy_pair")(pred, true)
        assert (full, bit) == pattern_accuracy_pair(pred, true)
        full_scores.append(full)
        bit_scores.append(bit)
    assert float(np.mean(full_scores)) == pytest.approx(0.5)
    assert float(np.mean(bit_scores)) == pytest.approx((2 / 3 + 1) / 2)


def test_historical_grid_is_not_started(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("historical training started")

    monkeypatch.setattr(historical, "train_qam", boom)
    monkeypatch.setattr(historical, "main", boom)
    before = _archive_bytes()
    loaded = load_benchmark_config(HISTORICAL)
    with pytest.raises(ConfigNotExecutable):
        run_benchmark(loaded, output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="only grid='smoke'"):
        run_statistical_significance(historical_request(), output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert _archive_bytes() == before


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_one_cell_matches_run_trial_and_is_not_the_archived_summary(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("experiment main was called")

    monkeypatch.setattr(historical, "main", boom)
    before = _archive_bytes()
    states = _states()
    try:
        result = run_benchmark(
            load_benchmark_config(SMOKE),
            output_dir=tmp_path,
        )
        _restore(states)
        direct = historical.run_trial((8, 0.2, 0.1, 0))
    finally:
        _restore(states)

    assert result["execution_status"] == "succeeded"
    assert result["comparable_to_archive"] is False
    assert result["main_invoked"] is False
    run_dir = Path(result["output_dir"])
    assert run_dir.is_relative_to(tmp_path)
    assert _archive_bytes() == before

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["historical_epochs"] == 100
    assert manifest["historical_depth"] == 10
    assert manifest["smoke_tasks"] == 1
    assert manifest["historical_tasks"] == 720
    assert manifest["historical_sample_size"] == 30
    assert manifest["bootstrap_seeded"] is False
    assert manifest["alpha_decision"] is None
    assert manifest["plotting"] is False
    assert manifest["main_invoked"] is False
    assert manifest["resumable"] is False
    assert manifest["comparable_to_archive"] is False
    assert manifest["process_model"] == "single_process"
    assert manifest["historical_process_model"] == "multiprocessing_pool"
    assert manifest["procedure_ids"] == ["hopfield_synchronous", "qam_paulix"]
    rng = manifest["rng"]
    assert rng["exact_reproducibility_claimed"] is False
    assert rng["confidence_intervals_reproducible"] is False
    assert rng["p_values_reproducible"] is False
    assert rng["internal_resets"] == []
    assert rng["unseeded_procedures"][0]["name"] == "experiments.statistical_significance.bootstrap_diff"
    assert rng["unseeded_procedures"][0]["seeded"] is False
    applied = rng["seeds_applied"][0]["values"]
    assert applied == [
        {
            "n_bits": 8,
            "capacity_ratio": 0.2,
            "noise": 0.1,
            "value": 0,
            "calls": ["numpy.random.seed", "pennylane.numpy.random.seed"],
        }
    ]

    with (run_dir / "trials.csv").open(encoding="utf-8", newline="") as handle:
        table = list(csv.reader(handle))
    assert tuple(table[0]) == TRIAL_COLUMNS
    assert len(table) == 2
    trial = json.loads((run_dir / "trials.json").read_text(encoding="utf-8"))[0]
    assert trial["m"] == 1
    assert trial["epochs"] == 100
    assert trial["depth"] == 10
    assert trial["hopfield_procedure_id"] == "hopfield_synchronous"
    assert trial["qam_procedure_id"] == "qam_paulix"
    assert trial["hopfield_full"] == pytest.approx(float(direct["hopfield_full"]))
    assert trial["hopfield_bit"] == pytest.approx(float(direct["hopfield_bit"]))
    assert trial["qam_full"] == pytest.approx(float(direct["qam_full"]))
    assert trial["qam_bit"] == pytest.approx(float(direct["qam_bit"]))

    _restore(states)
    try:
        np.random.seed(0)
        pnp.random.seed(0)
        patterns = np.random.choice([-1, 1], size=(1, 8))
    finally:
        _restore(states)
    assert trial["patterns"] == patterns.astype(int).tolist()

    with ARCHIVE_RAW.open(encoding="utf-8", newline="") as handle:
        archived_row = next(csv.DictReader(handle))
    assert float(archived_row["hopfield_full"]) == pytest.approx(trial["hopfield_full"])
    assert float(archived_row["qam_full"]) == pytest.approx(trial["qam_full"])
    assert archived_row["seed"] == "0"
    assert archived_row["n_bits"] == "8"

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["bootstrap_seeded"] is False
    assert summary["p_values_reproducible"] is False
    assert summary["confidence_intervals_reproducible"] is False
    assert summary["alpha_decision"] is None
    row = summary["rows"][0]
    assert row["sample_size"] == 1
    assert row["historical_sample_size"] == 30
    assert row["bootstrap_samples"] == 5000
    assert row["full_p_value"] == WILCOXON_FAILURE["statistical_significance"] == 1.0
    with ARCHIVE_SUMMARY.open(encoding="utf-8", newline="") as handle:
        archived_summary = next(csv.DictReader(handle))
    assert float(archived_summary["full_p_value"]) != row["full_p_value"]

    effect = json.loads((run_dir / "effect_sizes.json").read_text(encoding="utf-8"))
    assert effect["computed_by_experiment_main"] is False
    assert effect["rows"][0]["sample_size"] == 1
    assert effect["rows"][0]["computed_by_experiment_main"] is False
    assert "plt" not in (
        PROJECT_ROOT / "benchmark" / "protocols" / "statistical_significance.py"
    ).read_text(encoding="utf-8")


def test_archived_tables_keep_their_schema():
    with ARCHIVE_RAW.open(encoding="utf-8", newline="") as handle:
        raw = list(csv.reader(handle))
    with ARCHIVE_SUMMARY.open(encoding="utf-8", newline="") as handle:
        summary = list(csv.reader(handle))
    assert raw[0][:8] == [
        "n_bits",
        "capacity_ratio",
        "noise",
        "seed",
        "hopfield_full",
        "hopfield_bit",
        "qam_full",
        "qam_bit",
    ]
    assert len(raw) == 721
    assert summary[0][:9] == [
        "n_bits",
        "capacity_ratio",
        "noise",
        "mean_hopfield_full",
        "mean_qam_full",
        "mean_diff_full",
        "full_ci_2p5",
        "full_ci_97p5",
        "full_p_value",
    ]
    assert len(summary) == 25


def test_command_line_distinguishes_the_smoke_from_the_historical_grid(capsys, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("training or main started")

    monkeypatch.setattr(historical, "train_qam", boom)
    monkeypatch.setattr(historical, "build_qam", boom)
    monkeypatch.setattr(historical, "main", boom)
    assert main(["validate", str(SMOKE)]) == EXIT_OK
    smoke_plan = capsys.readouterr().out
    assert "executable: yes" in smoke_plan
    assert "qam_paulix" in smoke_plan
    assert "epochs: 100" in smoke_plan
    assert "bootstrap_seeded: false" in smoke_plan
    assert main(["validate", str(HISTORICAL)]) == EXIT_OK
    historical_plan = capsys.readouterr().out
    assert "executable: no" in historical_plan
    assert "dataset.n_bits_list: 8 12 16" in historical_plan
    assert main(["run", str(HISTORICAL), "--dry-run"]) == EXIT_OK
    assert "full_grid: not started" in capsys.readouterr().out
    assert main(["run", str(HISTORICAL)]) == EXIT_NOT_EXECUTABLE
    assert main(["run", str(SMOKE), "--full-grid", "--dry-run"]) == EXIT_NOT_EXECUTABLE
    assert "--full-grid applies only" in capsys.readouterr().err


def test_output_under_the_archive_is_refused(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("training started")

    monkeypatch.setattr(historical, "train_qam", boom)
    before = _archive_bytes()
    loaded = load_benchmark_config(SMOKE)
    with pytest.raises(ArchivedResultsRefused):
        run_benchmark(loaded, output_dir=ARCHIVE_RAW.parent)
    assert _archive_bytes() == before
    assert list(tmp_path.iterdir()) == []


def test_a_shorter_training_budget_is_rejected(tmp_path):
    document = json.loads(SMOKE.read_text(encoding="utf-8"))
    document["models"][1]["parameters"]["epochs"] = 1
    path = tmp_path / "epochs.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="epochs"):
        load_benchmark_config(path)

    document = json.loads(SMOKE.read_text(encoding="utf-8"))
    document["statistics"]["bootstrap"]["seeded"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="seed"):
        load_benchmark_config(path)


def _controlled_rows():
    group_a = _group(
        8,
        0.2,
        0.1,
        [0.0, 0.5, 1.0, 0.25],
        [1.0, 1.0, 0.0, 0.75],
        [0.4, 0.5, 0.6, 0.7],
        [0.9, 0.8, 0.2, 0.3],
    )
    group_b = _group(
        12,
        0.3,
        0.2,
        [1.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0, 0.5],
        [0.8, 0.8, 0.4, 0.4],
        [0.2, 0.3, 0.5, 0.6],
    )
    return group_b + group_a


def _group(n_bits, capacity, noise, hopfield_full, qam_full, hopfield_bit, qam_bit):
    rows = []
    for index, scores in enumerate(
        zip(hopfield_full, qam_full, hopfield_bit, qam_bit)
    ):
        hopfield_full_score, qam_full_score, hopfield_bit_score, qam_bit_score = scores
        rows.append(
            {
                "n_bits": n_bits,
                "capacity_ratio": capacity,
                "noise": noise,
                "seed": index,
                "hopfield_full": hopfield_full_score,
                "qam_full": qam_full_score,
                "hopfield_bit": hopfield_bit_score,
                "qam_bit": qam_bit_score,
            }
        )
    return rows


def _direct_summary(rows, bootstrap_samples):
    grouped = {}
    for row in rows:
        key = (row["n_bits"], row["capacity_ratio"], row["noise"])
        grouped.setdefault(key, []).append(row)
    produced = []
    for key in sorted(grouped):
        members = grouped[key]
        hopfield_full = np.asarray([row["hopfield_full"] for row in members], dtype=float)
        qam_full = np.asarray([row["qam_full"] for row in members], dtype=float)
        hopfield_bit = np.asarray([row["hopfield_bit"] for row in members], dtype=float)
        qam_bit = np.asarray([row["qam_bit"] for row in members], dtype=float)
        full_diffs = paired_bootstrap_diff(qam_full, hopfield_full, n=bootstrap_samples)
        full_low, full_high = np.percentile(full_diffs, [2.5, 97.5])
        bit_diffs = paired_bootstrap_diff(qam_bit, hopfield_bit, n=bootstrap_samples)
        bit_low, _bit_high = np.percentile(bit_diffs, [2.5, 97.5])
        delta_full, magnitude = selected("statistical_significance", "cliffs_hopfield_vs_qam")(
            hopfield_full, qam_full
        )
        produced.append(
            {
                "mean_diff_full": float(qam_full.mean() - hopfield_full.mean()),
                "full_ci_2p5": float(full_low),
                "full_ci_97p5": float(full_high),
                "full_p_value": float(wilcoxon(qam_full, hopfield_full).pvalue),
                "bit_ci_2p5": float(bit_low),
                "bit_p_value": float(wilcoxon(qam_bit, hopfield_bit).pvalue),
                "delta_full": float(delta_full),
                "full_magnitude": magnitude,
            }
        )
    return produced
