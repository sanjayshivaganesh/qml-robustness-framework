# Metric validation

Date: 2026-10-09. This record covers `benchmark/scoring.py` and `tests/test_metrics_regression.py`. No experiment `main` was run. No file under `results/` was written. Archived CSV values were not regenerated, and this document does not claim that a framework call reproduces them.

The framework calls the helpers in `src/metrics.py` or the one-line rules already present in the experiment and plot sources. Experiment modules still contain their own reductions. Those modules were not edited.

## Definitions preserved

Per-pattern scores come from `compute_metrics`, which returns `(bit_accuracy, full_accuracy, hamming_error)` in that order.

| Quantity | Definition | Helper |
|---|---|---|
| Bit accuracy | Fraction of matching positions, `float(mean(pred == true))`. | `bit_accuracy` |
| Full-pattern accuracy | `1.0` only when every position matches, otherwise `0.0`. | `full_accuracy` |
| Hamming error | Fraction of mismatches. On a non-empty pair it is `1 - bit accuracy`. | `hamming_error` |
| Pattern pair | `(1 or 0, bit accuracy)` for one pair. | `pattern_accuracy_pair` |
| Batch recall | Mean of per-pattern exact-match indicators, and mean of per-pattern bit fractions. | `full_and_bitwise_accuracy` |
| Recovery gain | Input Hamming error minus output Hamming error. Inference only. | arithmetic on `hamming_error` |
| Population Gaussian interval | Mean, `std` with `ddof=0`, and `1.96 * std / sqrt(n)`. | `mean_std_ci` |
| Sample Gaussian interval | Mean, `std` with `ddof=1`, and `1.96 * std / sqrt(n)`. This is the inference summary's Gaussian column. Pandas `Series.std` uses `ddof=1`. | `gaussian_ci_sample_std` |
| Bootstrap mean interval | Python loop of `rng.choice` then `mean`, default 2000 resamples, percentile bounds. Returns `(mean, lower, upper)`. | `bootstrap_mean_ci` |
| Vectorized percentile interval | One `rng.choice` of shape `(n_boot, n)`, default 200 resamples, percentiles 2.5 and 97.5. Returns `(low, high)` only. | `bootstrap_percentile_ci` |
| Paired bootstrap | Resample of QAM-column score minus Hopfield score, default 5000, via `numpy.random.randint` inside `experiments.statistical_significance.bootstrap_diff`. | that function |
| Cliff's delta | `(n_{x>y} - n_{x<y}) / (n_x * n_y)` with Hopfield scores as `x`. Positive means Hopfield tends to be larger. Bands at 0.147, 0.33, and 0.474. | `cliffs_delta`, `cliffs_delta_magnitude` |
| Wilcoxon | `scipy.stats.wilcoxon(x, y).pvalue`, or the caller's failure value if scipy raises. Storage capacity's failure value is `None`. Statistical significance's failure value is `1.0`. | `wilcoxon_pvalue` |
| Critical capacity | First scanned ratio whose mean full accuracy is strictly below 0.75. | `critical_capacity_ratio` |
| Critical noise 0.50 | Largest noise whose mean full accuracy is still at least 0.50. If none qualify, `0.0`. | `critical_noise_at_least` |
| Critical noise 0.90 | First noise, in the given order, whose mean full accuracy is below 0.90. `None` if the series never crosses, which is the plot script's omit rule. | `critical_noise_first_below` |
| Convergence | Loss at steps 0, 50, 100, 200, and the last point. `post100_improvement_pct = (loss_100 - loss_final) / max(abs(loss_100), 1e-12) * 100`. | `loss_checkpoints` |

`selected(protocol, metric_id)` returns only a callable that protocol already reports. The noise sweep does not report recovery gain. Inference does not report the 2000-resample loop. Convergence does not report batch recall.

## Recall, identity, and runtime

`label_pattern_recall` stores a procedure id next to `compute_metrics`. The id is not an input to the arithmetic. The same bipolar vectors scored as `hamming_nearest` and as `qam_ry_linear` produce the same three numbers and different ids. Passing the archive column string `qam` does not turn the score into a circuit result.

Train time, inference time, per-bit time, per-pattern time, wall time, energy, and energy drop are rejected as metric names. Energy remains the third return value of sequential Hopfield recall.

## What the tests check

Command:

```bash
./venv/bin/python -m pytest -q tests/test_metrics_regression.py
```

Hand-specified length-4 patterns:

- Exact match: bit `1`, full `1`, Hamming `0`, pair `(1, 1.0)`.
- Sign flip of every bit: bit `0`, full `0`, Hamming `1`, pair `(0, 0.0)`.
- Three of four bits match: bit `0.75`, full `0`, Hamming `0.25`, pair `(0, 0.75)`.

A two-pattern batch (one partial, one exact) has full accuracy `0.5` and bit accuracy `0.875`. On that batch the framework value equals `full_and_bitwise_accuracy`, the mean of `pattern_accuracy_pair`, the noise-sweep pooled counts, and the scaling per-pattern means.

A five-pattern length-6 batch, drawn from `RandomState(0)` and then edited by hand, still matches the noise-sweep full-accuracy count exactly. The pooled bit count `matching_bits / (m * n_bits)` and the mean of per-pattern bit fractions differ by one unit in the last place (`0.8666...667` versus `0.8666...668`). They are the same ratio in exact arithmetic when every pattern has the same length. The framework keeps `full_and_bitwise_accuracy`. The noise-sweep and scaling modules still use their own lines.

Other locked cases:

- `zip` stops at the shorter batch. An extra recalled pattern is ignored. The framework does not add a length check the helper does not have.
- Unequal pattern widths raise `ValueError` from NumPy.
- Empty equal arrays: bit accuracy and Hamming error are `nan`, full accuracy is `1.0` because `np.all` of an empty comparison is true, and `mean_std_ci([])` is `(nan, nan, nan)`. `cliffs_delta([], [])` raises `ZeroDivisionError`. These are the current helper results. They were not given a new policy.
- Population standard deviation of `[0, 1]` is `0.5`. The half-width is `1.96 * 0.5 / sqrt(2)`. The sample standard deviation matches `pandas.Series.std` and is not `0.5`.
- `bootstrap_mean_interval` with `RandomState(4)` and 40 resamples equals `bootstrap_mean_ci` on that same state, and the reported mean is the sample mean. `rng=None` raises.
- The vectorized percentile helper with `RandomState(2)` equals `bootstrap_percentile_ci`. Its default resample count is 200 and it returns two percentiles. The loop helper's default resample count is 2000 and it returns mean, lower, and upper.
- `paired_bootstrap_diff(qam, hopfield)` equals `bootstrap_diff` after the same global `numpy.random.seed(0)`. Swapping the arguments negates a constant-sign difference. The wrapper does not add an rng argument.
- Cliff's delta of `[2, 2]` against `[1, 0]` is `1` (Large). The swap is `-1`. The magnitude boundaries are `< 0.147`, `< 0.33`, and `< 0.474`.
- Wilcoxon on distinct samples equals `scipy.stats.wilcoxon`. On this install (SciPy 1.17.1), five identical scores return p-value `1.0` with a divide warning and do not enter `except`. A one-element tie raises `ValueError`; the wrapper then returns `None` or `1.0` according to the caller. The two failure constants stay distinct.
- Critical capacity uses a strict `< 0.75`, so `0.75` itself is not the threshold. The 0.50 noise rule is the maximum qualifying noise, including when accuracy falls and then rises. The 0.90 rule is the first noise in order below `0.90`, and a series that never crosses returns `None`.
- A length-301 loss series with known entries yields post-100 improvement `50`. A zero loss at step 100 uses the `1e-12` floor. A series shorter than the indexed steps raises `IndexError`.

`selected` rejects `infer_time` and `inference_percentile_bootstrap`.

## Not rewritten

These stay in the modules that own them. Equivalence with the shared helper was checked only for the batch-accuracy formulas above, on equal-length patterns.

| Location | What remains |
|---|---|
| `experiments/noise_robustness_sweep.py` | Trial totals `full_correct / m` and `bits_correct / (m * n_bits)`, then `mean_std_ci`. |
| `experiments/scaling.py` | Means of per-pattern exact matches and per-pattern bit fractions. `bootstrap_ci` is still unused. |
| `experiments/noise_robustness_inference.py` | `compute_metrics`, recovery gain, and the unseeded 200-resample bootstrap inside `aggregate_results`. The Gaussian columns still use the pandas group standard deviation. |
| `experiments/statistical_significance.py` | `pattern_accuracy_pair`, `bootstrap_diff`, and Wilcoxon with failure value `1.0`. |
| `experiments/storage_capacity.py` | `full_and_bitwise_accuracy`, `bootstrap_mean_ci` on the global stream, Wilcoxon with failure value `None`, and the unused `confidence_interval`. |
| `experiments/circuit_depth.py` | `full_and_bitwise_accuracy`. |
| `experiments/convergence.py` | The loss-checkpoint formula. `std_loss` there is NumPy's population `std`. |
| `scripts/plot_critical_noise_threshold.py` | The 0.50 table, still written under `results/` if that script is run. |
| `scripts/plot_inference_paper_figures.py` | The 0.90 table. |
| `scripts/plot_bootstrap_distribution.py` | 5000 unseeded independent resamples. Not a paired bootstrap and not wrapped. |
| `src/metrics.py` `parametric_ci` | No caller. Not selectable. |

## Limitations

- An unseeded bootstrap is not reproducible. The inference summary percentiles and the bootstrap histogram script were not wrapped and were not seeded in order to make them so.
- `bootstrap_vectorized_percentile` with an explicit `RandomState` repeats that vectorized algorithm. It is not a replay of an inference summary that never set a seed.
- `paired_bootstrap_diff` still consumes the global NumPy stream, because `bootstrap_diff` has no rng argument.
- `default_rng` is not used. A legacy `RandomState` with the same integer is the seeded fixture.
- Perfect recall of several equal Wilcoxon pairs does not currently take the `except` branch on SciPy 1.17.1. Older archives that store `None` for that case were not recomputed. The wrapper still catches `Exception` and still has two different failure values.
- Batch bit accuracy from a pooled count and from the mean of per-pattern fractions can differ by one floating-point unit. The small hand batch matched exactly. The noise-sweep source was left on the pooled count.
- Empty full-pattern accuracy is `1.0` under the current NumPy helper. That is not a claim about an empty memory.
- Wall-clock columns are outside this interface. They are not comparable across machines and are not acceptance criteria here.
