"""Known-answer checks for recall metrics and interval helpers.

Compares ``benchmark.scoring`` with ``src.metrics`` on the same arrays.
Does not call an experiment ``main`` and does not write under ``results/``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon

from benchmark.scoring import (
    METRICS,
    NOT_SELECTED,
    PROTOCOL_METRICS,
    RUNTIME_QUANTITIES,
    WILCOXON_FAILURE,
    MetricNotInProtocol,
    UnknownMetric,
    batch_recall,
    bootstrap_mean_interval,
    bootstrap_vectorized_percentile,
    cliffs_hopfield_vs_qam,
    critical_capacity_ratio,
    critical_noise_at_least,
    critical_noise_first_below,
    gaussian_ci_population,
    gaussian_ci_sample_std,
    label_pattern_recall,
    loss_checkpoints,
    paired_bootstrap_diff,
    post100_improvement_pct,
    recovery_gain,
    selected,
    wilcoxon_pvalue,
)
from src.metrics import (
    bit_accuracy,
    bootstrap_mean_ci,
    bootstrap_percentile_ci,
    cliffs_delta,
    compute_metrics,
    full_accuracy,
    full_and_bitwise_accuracy,
    hamming_error,
    mean_std_ci,
    pattern_accuracy_pair,
)
from src.paths import PROJECT_ROOT


def _noise_sweep_totals(recalled, true):
    """The reduction written in ``experiments/noise_robustness_sweep.py``."""
    m = len(true)
    n_bits = len(true[0])
    full_correct = 0
    bits_correct = 0
    for rec, target in zip(recalled, true):
        if np.array_equal(rec, target):
            full_correct += 1
        bits_correct += np.sum(rec == target)
    return full_correct / m, bits_correct / (m * n_bits)


def _scaling_means(recalled, true):
    """The reduction written in ``experiments/scaling.py``."""
    full_acc = [int(np.array_equal(rec, target)) for rec, target in zip(recalled, true)]
    bit_acc = [np.mean(rec == target) for rec, target in zip(recalled, true)]
    return np.mean(full_acc), np.mean(bit_acc)


def test_perfect_mismatch_and_partial_pattern_recall():
    true = np.array([1, -1, 1, -1])
    perfect = true.copy()
    mismatch = -true
    partial = np.array([1, -1, 1, 1])

    assert compute_metrics(perfect, true) == (1.0, 1.0, 0.0)
    assert bit_accuracy(perfect, true) == 1.0
    assert full_accuracy(perfect, true) == 1.0
    assert hamming_error(perfect, true) == 0.0
    assert pattern_accuracy_pair(perfect, true) == (1, 1.0)

    assert compute_metrics(mismatch, true) == (0.0, 0.0, 1.0)
    assert pattern_accuracy_pair(mismatch, true) == (0, 0.0)

    bit, full, hamming = compute_metrics(partial, true)
    assert bit == 0.75
    assert full == 0.0
    assert hamming == 0.25
    assert bit + hamming == 1.0
    assert pattern_accuracy_pair(partial, true) == (0, 0.75)


def test_batch_recall_known_answer_and_original_helpers():
    recalled = [
        np.array([1, -1, 1, 1]),
        np.array([-1, -1, 1, 1]),
    ]
    true = [
        np.array([1, -1, 1, -1]),
        np.array([-1, -1, 1, 1]),
    ]
    full, bit = batch_recall(recalled, true)
    direct = full_and_bitwise_accuracy(recalled, true)
    assert full == direct[0]
    assert bit == direct[1]
    assert float(full) == 0.5
    assert float(bit) == 0.875

    pair_full = np.mean(
        [pattern_accuracy_pair(r, t)[0] for r, t in zip(recalled, true)]
    )
    pair_bit = np.mean(
        [pattern_accuracy_pair(r, t)[1] for r, t in zip(recalled, true)]
    )
    assert float(pair_full) == float(full)
    assert float(pair_bit) == float(bit)

    sweep_full, sweep_bit = _noise_sweep_totals(recalled, true)
    scaling_full, scaling_bit = _scaling_means(recalled, true)
    assert sweep_full == float(full)
    assert sweep_bit == float(bit)
    assert float(scaling_full) == float(full)
    assert float(scaling_bit) == float(bit)


def test_equal_length_batch_formulas_match_on_a_larger_bipolar_set():
    rng = np.random.RandomState(0)
    true = [rng.choice([-1, 1], size=6) for _ in range(5)]
    recalled = []
    for pattern in true:
        flipped = pattern.copy()
        flipped[1] *= -1
        recalled.append(flipped)
    recalled[0] = true[0].copy()
    full, bit = batch_recall(recalled, true)
    sweep_full, sweep_bit = _noise_sweep_totals(recalled, true)
    pooled_hits = sum(int(np.sum(r == t)) for r, t in zip(recalled, true))
    assert pooled_hits / (5 * 6) == sweep_bit
    assert sweep_full == float(full)
    assert sweep_bit == pytest.approx(float(bit))


def test_zip_truncation_is_the_helper_convention():
    true = [np.array([1, -1]), np.array([1, 1])]
    recalled = [np.array([1, -1]), np.array([1, 1]), np.array([-1, -1])]
    assert batch_recall(recalled, true) == full_and_bitwise_accuracy(recalled, true)
    assert float(batch_recall(recalled, true)[0]) == 1.0


def test_shape_mismatch_raises_from_the_helper():
    with pytest.raises(ValueError):
        bit_accuracy(np.array([1, -1]), np.array([1, -1, 1]))
    with pytest.raises(ValueError):
        batch_recall([np.array([1, -1])], [np.array([1, -1, 1])])


def test_empty_inputs_follow_the_current_numpy_helpers():
    empty = np.array([])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert np.isnan(bit_accuracy(empty, empty))
        assert full_accuracy(empty, empty) == 1.0
        assert np.isnan(hamming_error(empty, empty))
        mean, std, half = mean_std_ci([])
        assert np.isnan(mean) and np.isnan(std) and np.isnan(half)
        batch_full, batch_bit = full_and_bitwise_accuracy([], [])
        assert np.isnan(batch_full) and np.isnan(batch_bit)
    with pytest.raises(ZeroDivisionError):
        cliffs_delta([], [])


def test_hamming_nearest_keeps_its_procedure_id():
    true = np.array([1, -1, 1, -1])
    pred = np.array([1, -1, -1, -1])
    hamming = label_pattern_recall("hamming_nearest", pred, true)
    circuit = label_pattern_recall("qam_ry_linear", pred, true)
    assert hamming.procedure_id == "hamming_nearest"
    assert circuit.procedure_id == "qam_ry_linear"
    assert hamming.bit_accuracy == circuit.bit_accuracy == 0.75
    assert hamming.full_accuracy == circuit.full_accuracy == 0.0
    assert hamming.hamming_error == circuit.hamming_error == 0.25
    assert hamming.bit_accuracy == compute_metrics(pred, true)[0]
    archive_label = label_pattern_recall("qam", pred, true)
    assert archive_label.procedure_id == "qam"
    assert archive_label.bit_accuracy == hamming.bit_accuracy


def test_recovery_gain_is_input_minus_output_hamming():
    true = np.array([1, 1, 1, 1])
    noisy = np.array([1, 1, -1, -1])
    recalled = np.array([1, 1, 1, -1])
    incoming = hamming_error(noisy, true)
    outgoing = hamming_error(recalled, true)
    assert incoming == 0.5
    assert outgoing == 0.25
    assert recovery_gain(incoming, outgoing) == 0.25
    assert recovery_gain(incoming, 0.0) == incoming
    assert recovery_gain(0.25, 0.5) == -0.25


def test_population_and_sample_gaussian_intervals_differ():
    values = np.array([0.0, 1.0])
    mean, std, half = gaussian_ci_population(values)
    direct = mean_std_ci(values)
    assert mean == direct[0]
    assert std == direct[1] == np.std(values, ddof=0)
    assert half == direct[2]
    assert std == 0.5
    assert half == pytest.approx(1.96 * 0.5 / np.sqrt(2))

    sample_mean, sample_std, sample_half = gaussian_ci_sample_std(values)
    series = pd.Series(values)
    assert sample_mean == series.mean()
    assert sample_std == series.std()
    assert sample_std == np.std(values, ddof=1)
    assert sample_half == pytest.approx(1.96 * float(series.std()) / np.sqrt(len(series)))
    assert sample_std != std


def test_bootstrap_mean_interval_matches_the_helper_when_seeded():
    data = np.array([0.0, 1.0, 0.5, 0.25])
    via = bootstrap_mean_interval(data, rng=np.random.RandomState(4), n_boot=40)
    direct = bootstrap_mean_ci(data, n_boot=40, rng=np.random.RandomState(4))
    assert via == direct
    assert via[0] == float(np.mean(data))
    with pytest.raises(TypeError):
        bootstrap_mean_interval(data, rng=None)


def test_bootstrap_helpers_keep_their_sample_counts_and_return_shapes():
    import inspect

    from experiments import noise_robustness_inference, scaling

    data = np.array([0.0, 1.0, 0.5, 0.25])
    low, high = bootstrap_vectorized_percentile(
        data, rng=np.random.RandomState(2), n_boot=30
    )
    direct = bootstrap_percentile_ci(data, n_boot=30, rng=np.random.RandomState(2))
    assert np.array_equal(np.asarray([low, high]), np.asarray(direct))
    with pytest.raises(TypeError):
        bootstrap_vectorized_percentile(data, rng=None)

    mean_sig = inspect.signature(bootstrap_mean_ci)
    percentile_sig = inspect.signature(bootstrap_percentile_ci)
    assert mean_sig.parameters["n_boot"].default == 2000
    assert percentile_sig.parameters["n_boot"].default == 200
    loop_result = bootstrap_mean_interval(data, rng=np.random.RandomState(1), n_boot=5)
    assert len(loop_result) == 3
    assert len(direct) == 2

    inference_text = inspect.getsource(noise_robustness_inference.aggregate_results)
    assert "n_boot=200" in inference_text
    assert "seed(" not in inference_text
    assert "bootstrap_ci(" not in inspect.getsource(scaling.main)


def test_paired_bootstrap_difference_is_qam_minus_hopfield():
    qam = np.array([1.0, 1.0, 0.0])
    hopfield = np.array([0.0, 0.0, 0.0])
    before = np.random.get_state()
    try:
        np.random.seed(0)
        diffs = paired_bootstrap_diff(qam, hopfield, n=6)
        np.random.seed(0)
        from experiments.statistical_significance import bootstrap_diff

        direct = bootstrap_diff(qam, hopfield, n=6)
        assert np.array_equal(diffs, direct)
        assert np.all(diffs >= 0.0)
        np.random.seed(0)
        swapped = paired_bootstrap_diff(hopfield, qam, n=6)
        assert np.array_equal(swapped, -diffs)
    finally:
        np.random.set_state(before)


def test_cliffs_delta_sign_and_magnitude_bands():
    delta, magnitude = cliffs_hopfield_vs_qam([2, 2], [1, 0])
    assert delta == cliffs_delta([2, 2], [1, 0]) == 1.0
    assert magnitude == "Large"
    swapped, _magnitude = cliffs_hopfield_vs_qam([1, 0], [2, 2])
    assert swapped == -1.0
    assert cliffs_delta_magnitude_name(0.146) == "Negligible"
    assert cliffs_delta_magnitude_name(0.147) == "Small"
    assert cliffs_delta_magnitude_name(0.33) == "Medium"
    assert cliffs_delta_magnitude_name(0.474) == "Large"


def cliffs_delta_magnitude_name(delta):
    from src.metrics import cliffs_delta_magnitude

    return cliffs_delta_magnitude(delta)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_wilcoxon_matches_scipy_and_sentinels_stay_distinct():
    hopfield = np.array([0.2, 0.5, 0.7, 0.9])
    qam = np.array([0.1, 0.4, 0.6, 0.8])
    with np.errstate(invalid="ignore", divide="ignore"):
        assert (
            wilcoxon_pvalue(hopfield, qam, on_failure=WILCOXON_FAILURE["storage_capacity"])
            == wilcoxon(hopfield, qam).pvalue
        )
        assert (
            wilcoxon_pvalue(
                qam, hopfield, on_failure=WILCOXON_FAILURE["statistical_significance"]
            )
            == wilcoxon(qam, hopfield).pvalue
        )

        tied = np.ones(5)
        assert wilcoxon(tied, tied).pvalue == 1.0
        assert wilcoxon_pvalue(tied, tied, on_failure=None) == 1.0

    one = np.array([1.0])
    with pytest.raises(ValueError):
        wilcoxon(one, one)
    assert wilcoxon_pvalue(one, one, on_failure=None) is None
    assert wilcoxon_pvalue(one, one, on_failure=1.0) == 1.0
    assert WILCOXON_FAILURE["storage_capacity"] is None
    assert WILCOXON_FAILURE["statistical_significance"] == 1.0


def test_critical_rules_use_the_existing_thresholds():
    ratios = [0.125, 0.25, 0.375, 0.5]
    accuracies = [1.0, 0.8, 0.75, 0.7]
    assert critical_capacity_ratio(ratios, accuracies) == 0.5
    assert critical_capacity_ratio([0.25, 0.5], [1.0, 0.75]) is None

    noise = [0.0, 0.1, 0.2, 0.3]
    bouncing = [1.0, 0.6, 0.4, 0.55]
    assert critical_noise_at_least(noise, bouncing) == 0.3
    assert critical_noise_at_least(noise, [0.2, 0.2, 0.2, 0.2]) == 0.0

    dip_then_recover = [0.95, 0.91, 0.8, 0.95]
    assert critical_noise_first_below(noise, dip_then_recover) == 0.2
    assert critical_noise_first_below(noise, [0.95, 0.95, 0.91, 0.90]) is None

    capacity_text = (PROJECT_ROOT / "experiments" / "storage_capacity.py").read_text()
    floor_text = (PROJECT_ROOT / "scripts" / "plot_critical_noise_threshold.py").read_text()
    paper_text = (PROJECT_ROOT / "scripts" / "plot_inference_paper_figures.py").read_text()
    assert "full_acc_mean" in capacity_text and "0.75" in capacity_text
    assert "ACCURACY_THRESHOLD = 0.50" in floor_text
    assert "CRITICAL_ACCURACY = 0.90" in paper_text


def test_convergence_checkpoints_and_zero_loss_guard():
    series = np.zeros(301)
    series[0] = 3.0
    series[50] = 2.5
    series[100] = 2.0
    series[200] = 1.5
    series[-1] = 1.0
    points = loss_checkpoints(series)
    assert points["loss_step_0"] == 3.0
    assert points["loss_step_50"] == 2.5
    assert points["loss_step_100"] == 2.0
    assert points["loss_step_200"] == 1.5
    assert points["loss_final"] == 1.0
    assert points["post100_improvement_pct"] == 50.0
    assert post100_improvement_pct(series) == 50.0

    flat = np.zeros(301)
    flat[-1] = 1.0
    assert post100_improvement_pct(flat) == (0.0 - 1.0) / 1e-12 * 100
    with pytest.raises(IndexError):
        loss_checkpoints(np.arange(10))
    text = (PROJECT_ROOT / "experiments" / "convergence.py").read_text()
    assert "max(abs(loss_100), 1e-12)" in text


def test_protocols_select_only_their_own_metrics():
    assert (
        selected("noise_robustness_sweep", "gaussian_ci_population")
        is gaussian_ci_population
    )
    assert selected("storage_capacity", "bootstrap_mean_ci") is bootstrap_mean_interval
    assert selected("noise_robustness_inference", "recovery_gain") is recovery_gain
    assert selected("convergence", "loss_checkpoints") is loss_checkpoints
    with pytest.raises(MetricNotInProtocol):
        selected("noise_robustness_sweep", "recovery_gain")
    with pytest.raises(MetricNotInProtocol):
        selected("noise_robustness_inference", "bootstrap_mean_ci")
    with pytest.raises(MetricNotInProtocol):
        selected("convergence", "batch_recall")
    with pytest.raises(UnknownMetric):
        selected("noise_robustness_sweep", "infer_time")
    with pytest.raises(UnknownMetric):
        selected("noise_robustness_inference", "inference_percentile_bootstrap")
    for name in RUNTIME_QUANTITIES:
        assert name not in METRICS
    for name in NOT_SELECTED:
        assert name not in METRICS
    assert "batch_recall" in PROTOCOL_METRICS["scaling"]
    assert "gaussian_ci_population" not in PROTOCOL_METRICS["noise_robustness_inference"]


def test_scoring_module_does_not_write_results():
    text = (PROJECT_ROOT / "benchmark" / "scoring.py").read_text()
    assert "results_subdir" not in text
    assert "to_csv" not in text
    assert "AdamOptimizer" not in text
