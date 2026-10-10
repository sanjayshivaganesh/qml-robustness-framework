"""Explicit metric selection over the helpers the experiments already use.

Recall scores, interval estimators, and summary rules are separate names.
A protocol asks for the names it reports. Runtime is not one of those names.

Nothing here writes a result file or runs an experiment. Experiment modules
keep their own inline reductions and their own unseeded bootstrap copies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.metrics import (
    bit_accuracy,
    bootstrap_mean_ci,
    bootstrap_percentile_ci,
    cliffs_delta,
    cliffs_delta_magnitude,
    compute_metrics,
    full_accuracy,
    full_and_bitwise_accuracy,
    hamming_error,
    mean_std_ci,
    pattern_accuracy_pair,
)

# Failure values stored by the two Wilcoxon callers. They are not unified.
WILCOXON_FAILURE = {
    "storage_capacity": None,
    "statistical_significance": 1.0,
}

# Quantities the experiments time. They are not recall metrics.
RUNTIME_QUANTITIES = frozenset(
    {
        "train_time",
        "infer_time",
        "recall_timed_per_bit",
        "infer_time_per_pattern",
        "inference_wall_time",
        "energy",
        "energy_drop",
    }
)

# Present in the tree and intentionally not selectable. Wiring any of these
# into a reported column would change what an experiment currently emits.
NOT_SELECTED = {
    "parametric_ci": "src.metrics.parametric_ci has no caller",
    "storage_confidence_interval": (
        "experiments.storage_capacity.confidence_interval has no caller"
    ),
    "scaling_bootstrap_ci": (
        "experiments.scaling.bootstrap_ci is defined and unused; "
        "scaling figures do not read it"
    ),
    "inference_percentile_bootstrap": (
        "experiments.noise_robustness_inference.aggregate_results "
        "draws 200 unseeded resamples and does not accept an rng"
    ),
    "plot_bootstrap_histogram": (
        "scripts.plot_bootstrap_distribution draws 5000 unseeded "
        "resamples and is not a trial metric"
    ),
}


class UnknownMetric(KeyError):
    """The name is not a recall, interval, or summary metric in this tree."""


class MetricNotInProtocol(KeyError):
    """That protocol does not report this metric."""

    def __init__(self, protocol, metric_id):
        self.protocol = protocol
        self.metric_id = metric_id
        super().__init__(f"{protocol} does not report {metric_id}")


@dataclass(frozen=True)
class LabeledPatternRecall:
    """Per-pattern recall scores plus the procedure that produced ``pred``.

    The numbers are ``compute_metrics``. The procedure id is not used in
    the arithmetic. Hamming nearest-neighbour and a variational circuit
    can share these numbers and still be different procedures.
    """

    procedure_id: str
    bit_accuracy: float
    full_accuracy: float
    hamming_error: float


def label_pattern_recall(procedure_id, pred, true):
    """Score one pattern. ``procedure_id`` is stored and not interpreted."""
    bit, full, hamming = compute_metrics(pred, true)
    return LabeledPatternRecall(procedure_id, bit, full, hamming)


def batch_recall(recalled_patterns, true_patterns):
    """Return ``full_and_bitwise_accuracy`` unchanged.

    Full accuracy is the mean of per-pattern exact matches. Bit accuracy is
    the mean of per-pattern match fractions. On equal-length patterns this
    matches the noise-sweep totals ``matches / m`` and
    ``matching_bits / (m * n_bits)``, and the scaling means of the same
    per-pattern indicators. The experiment modules still compute those
    totals themselves.
    """
    return full_and_bitwise_accuracy(recalled_patterns, true_patterns)


def recovery_gain(input_hamming_error, output_hamming_error):
    """Inference only: input Hamming error minus output Hamming error."""
    return input_hamming_error - output_hamming_error


def gaussian_ci_population(values, z=1.96):
    """Noise-sweep interval: mean, population std, ``z * std / sqrt(n)``."""
    return mean_std_ci(values, z=z)


def gaussian_ci_sample_std(values, z=1.96):
    """Inference Gaussian column: mean, sample std (``ddof=1``), half-width.

    Pandas ``Series.std`` defaults to ``ddof=1``. This is not
    ``mean_std_ci``. ``n`` is the length of ``values``, which is the group
    size the inference summary divides by when the cell is complete.
    """
    values = np.asarray(values, dtype=float)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    half = z * std / np.sqrt(len(values))
    return mean, std, half


def bootstrap_mean_interval(data, *, rng, n_boot=2000, confidence=0.95):
    """Storage-capacity bootstrap. ``rng`` is required.

    Pass ``numpy.random`` only when the caller intends the global stream
    the storage-capacity wrapper uses. ``default_rng`` is not a substitute.
    """
    if rng is None:
        raise TypeError(
            "rng is required; pass numpy.random to use the global stream"
        )
    return bootstrap_mean_ci(
        data, n_boot=n_boot, confidence=confidence, rng=rng
    )


def bootstrap_vectorized_percentile(series, *, rng, n_boot=200):
    """Vectorized percentile interval from ``bootstrap_percentile_ci``.

    This is not the inference summary. That summary calls ``numpy.random``
    with no seed. Passing an explicit ``rng`` here makes a repeatable draw
    of the same vectorized algorithm; it does not reproduce an unseeded run.
    """
    if rng is None:
        raise TypeError(
            "rng is required; an omitted seed is not a reproducible interval"
        )
    return bootstrap_percentile_ci(series, n_boot=n_boot, rng=rng)


def paired_bootstrap_diff(a, b, n=5000):
    """Paired resample of ``a - b`` via the statistical-significance function.

    The historical function draws ``numpy.random.randint`` and has no rng
    argument. This wrapper does not add one. ``a`` is the QAM-column score
    and ``b`` is the Hopfield score in that experiment.
    """
    from experiments.statistical_significance import bootstrap_diff

    return bootstrap_diff(a, b, n=n)


def cliffs_hopfield_vs_qam(hopfield_scores, qam_scores):
    """Cliff's delta with Hopfield as ``x`` and the QAM column as ``y``.

    Positive means Hopfield scores tend to be larger. Returns
    ``(delta, magnitude)``.
    """
    delta = cliffs_delta(hopfield_scores, qam_scores)
    return delta, cliffs_delta_magnitude(delta)


def wilcoxon_pvalue(x, y, *, on_failure):
    """Two-sided Wilcoxon p-value, or ``on_failure`` if scipy raises.

    Storage capacity passes Hopfield then QAM and stores ``None`` on
    failure. Statistical significance passes QAM then Hopfield and stores
    ``1.0``. Argument order is the caller's. This function does not swap
    them and does not map one sentinel onto the other.
    """
    from scipy.stats import wilcoxon

    try:
        return wilcoxon(x, y).pvalue
    except Exception:
        return on_failure


def critical_capacity_ratio(capacity_ratios, full_accuracies, threshold=0.75):
    """First ratio whose mean full accuracy is strictly below ``threshold``.

    Rows must already be in the order the caller wants scanned. Storage
    capacity sorts by ratio ascending. A value equal to the threshold is
    not critical. ``None`` if every accuracy stays at or above it.
    """
    for ratio, accuracy in zip(capacity_ratios, full_accuracies):
        if accuracy < threshold:
            return ratio
    return None


def critical_noise_at_least(noise_levels, full_accuracies, threshold=0.50):
    """Largest noise whose full accuracy is still at least ``threshold``.

    This is the 0.50 rule in ``scripts/plot_critical_noise_threshold.py``:
    the maximum qualifying noise, not the last point of a contiguous run.
    If none qualify, the script stores ``0.0``.
    """
    qualifying = [
        noise
        for noise, accuracy in zip(noise_levels, full_accuracies)
        if accuracy >= threshold
    ]
    if not qualifying:
        return 0.0
    return max(qualifying)


def critical_noise_first_below(noise_levels, full_accuracies, threshold=0.90):
    """First noise, in the given order, whose full accuracy is below ``threshold``.

    This is the 0.90 rule in ``scripts/plot_inference_paper_figures.py``.
    ``None`` if accuracy never falls below the threshold. The plot script
    then omits that group.
    """
    for noise, accuracy in zip(noise_levels, full_accuracies):
        if accuracy < threshold:
            return noise
    return None


def post100_improvement_pct(mean_loss):
    """``(loss_100 - loss_final) / max(|loss_100|, 1e-12) * 100``."""
    series = np.asarray(mean_loss, dtype=float)
    loss_100 = float(series[100])
    loss_final = float(series[-1])
    return (loss_100 - loss_final) / max(abs(loss_100), 1e-12) * 100


def loss_checkpoints(mean_loss):
    """Convergence summary points. The series is mean loss of length ``steps + 1``."""
    series = np.asarray(mean_loss, dtype=float)
    return {
        "loss_step_0": float(series[0]),
        "loss_step_50": float(series[50]),
        "loss_step_100": float(series[100]),
        "loss_step_200": float(series[200]),
        "loss_final": float(series[-1]),
        "post100_improvement_pct": post100_improvement_pct(series),
    }


METRICS = {
    "bit_accuracy": bit_accuracy,
    "full_accuracy": full_accuracy,
    "hamming_error": hamming_error,
    "compute_metrics": compute_metrics,
    "pattern_accuracy_pair": pattern_accuracy_pair,
    "label_pattern_recall": label_pattern_recall,
    "batch_recall": batch_recall,
    "recovery_gain": recovery_gain,
    "gaussian_ci_population": gaussian_ci_population,
    "gaussian_ci_sample_std": gaussian_ci_sample_std,
    "bootstrap_mean_ci": bootstrap_mean_interval,
    "bootstrap_vectorized_percentile": bootstrap_vectorized_percentile,
    "paired_bootstrap_diff": paired_bootstrap_diff,
    "cliffs_hopfield_vs_qam": cliffs_hopfield_vs_qam,
    "wilcoxon_pvalue": wilcoxon_pvalue,
    "critical_capacity_ratio": critical_capacity_ratio,
    "critical_noise_at_least_0_50": critical_noise_at_least,
    "critical_noise_below_0_90": critical_noise_first_below,
    "post100_improvement_pct": post100_improvement_pct,
    "loss_checkpoints": loss_checkpoints,
}

PROTOCOL_METRICS = {
    "noise_robustness_sweep": frozenset(
        {"batch_recall", "gaussian_ci_population"}
    ),
    "storage_capacity": frozenset(
        {
            "batch_recall",
            "bootstrap_mean_ci",
            "wilcoxon_pvalue",
            "critical_capacity_ratio",
        }
    ),
    "circuit_depth": frozenset({"batch_recall"}),
    "convergence": frozenset({"post100_improvement_pct", "loss_checkpoints"}),
    "scaling": frozenset({"batch_recall"}),
    "statistical_significance": frozenset(
        {
            "pattern_accuracy_pair",
            "paired_bootstrap_diff",
            "wilcoxon_pvalue",
            "cliffs_hopfield_vs_qam",
        }
    ),
    "noise_robustness_inference": frozenset(
        {
            "compute_metrics",
            "label_pattern_recall",
            "hamming_error",
            "recovery_gain",
            "gaussian_ci_sample_std",
        }
    ),
}


def selected(protocol, metric_id):
    """Return the callable that ``protocol`` is allowed to report.

    Runtime names and the unseeded historical bootstraps are rejected.
    """
    if metric_id in RUNTIME_QUANTITIES:
        raise UnknownMetric(f"{metric_id} is a runtime quantity, not a recall metric")
    if metric_id in NOT_SELECTED:
        raise UnknownMetric(NOT_SELECTED[metric_id])
    if protocol not in PROTOCOL_METRICS:
        raise UnknownMetric(f"unknown protocol {protocol}")
    if metric_id not in PROTOCOL_METRICS[protocol]:
        raise MetricNotInProtocol(protocol, metric_id)
    return METRICS[metric_id]
