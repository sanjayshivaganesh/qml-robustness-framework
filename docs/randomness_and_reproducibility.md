# Randomness and reproducibility

Date: 2026-10-10. This note records the generators the current code actually calls. A seed written into a manifest is a label for those calls. It is not a certificate that a later run will match an archived CSV.

The framework runner executes the noise-sweep smoke, the scaling smoke, one statistical-significance cell, one storage-capacity cell, one circuit-depth cell, and one inference cell. Each manifest field `rng` names the seeds that run applies. This note does not move those calls onto an independent generator, and it does not seed the unseeded bootstrap helpers. The noise-sweep manifest still records the constructor reset to 42.

`numpy.random.seed` and `pennylane.numpy.random.seed` are different functions on one legacy `RandomState`. `numpy.random.default_rng` is a different family. An equal integer does not make those families draw the same sequence.

## Current RNG behaviour

### Shared stream

| Caller | What it draws | Generator |
|---|---|---|
| Pattern `choice` of ±1 | Stored patterns | The module's `numpy.random` or `pennylane.numpy.random` after that trial's seed call. Both are the legacy stream. |
| `corrupt_bernoulli` with `rng=None` | One `rand` per bit | `numpy.random` |
| `corrupt_fixed_count` with `rng=None` | `choice` of bit indices, without replacement | `numpy.random` |
| `src.qam.train_ry_linear_qam` and `train_ry_linear_qam_record_loss` | `randn` for parameters, scale 0.01 | `pennylane.numpy.random`, because that module imports PennyLane NumPy as `np` |
| `HopfieldAsyncRandom.recall` | One `permutation` per sweep | `pennylane.numpy.random` |
| `experiments.noise_robustness_sweep.QAM.__init__` | `pennylane.numpy.random.seed(42)`, then `numpy.random.randn` | Legacy stream, reset to 42 inside the constructor |
| `experiments.scaling.QAM` | `randn` in `__init__`; `choice` of a minibatch inside `loss`, which `opt.step` evaluates a second time | `pennylane.numpy.random`. The constructor does not reseed. |
| `experiments.statistical_significance.train_qam` | `numpy.random.normal(0, 0.1)` | Legacy stream, after both seed functions were set to the trial seed |
| Inference `aggregate_results` and `scripts/plot_bootstrap_distribution.py` | `choice` or `randint` resamples | `numpy.random`, with no seed call in that function |
| `src.metrics.bootstrap_mean_ci` and `bootstrap_percentile_ci` | `choice` resamples | The `rng` argument, or `numpy.random` when `rng` is omitted |
| `experiments.statistical_significance.bootstrap_diff` | `randint` indexes | `numpy.random`, with no seed call in that function |
| Framework run directory id | `uuid.uuid4` | Not the legacy stream, and not a scientific seed |

Synchronous Hopfield recall, sequential Hopfield recall, and Hamming nearest neighbour do not draw. Full-batch Adam on the noise-sweep loss and on the RY-linear loss does not draw inside the loss. The Gaussian interval `mean_std_ci` does not draw.

### Noise-sweep smoke

For each configured trial seed the runner calls `numpy.random.seed` and `pennylane.numpy.random.seed` with that integer, then:

1. `choice` for the stored patterns.
2. Bernoulli corruption and synchronous recall for every stored pattern. Those queries use the trial seed.
3. `QAM(n_bits, train_steps=...)`. The class still executes `pnp.random.seed(42)` and then `np.random.randn`. The adapter does not add a seed call.
4. Full-batch Adam for the requested number of steps. The smoke uses 1 to 5 steps. The class default remains 300. Training does not draw.
5. Bernoulli corruption and recall again. Those queries continue from the reset, not from the trial seed.

The manifest records three different facts:

| Field | Meaning for this smoke |
|---|---|
| `configured_seeds.trial_seeds` | The seeds in the request. |
| `seeds_applied` | Those same integers, passed to both global seed functions at the start of each trial. |
| `internal_resets` | `pennylane.numpy.random.seed(42)` inside `QAM.__init__`. `configured` is false. This value is not a trial seed. |
| `unseeded_procedures` | `gaussian_ci_population`. It does not draw. |
| `state_not_captured` | The `RandomState` is not saved. Draws after the reset are not given their own seed. The run id is not a scientific seed. |
| `exact_reproducibility_claimed` | False. |

### Seed calls that stay on the legacy stream

These rows are the historical call sites. The framework adapters call the same functions and do not insert a generator. Scaling, statistical significance, storage capacity, circuit depth, and inference have a reduced runner. Convergence does not.

| Protocol | Seed that is applied | Later draws on that stream | Reset |
|---|---|---|---|
| Storage capacity, one cell | `numpy.random.seed(seed)` for seeds `0..4`, repeated for every `(n_bits, m)`. The smoke cell is `n=8`, `m=1`. | Pattern `choice`, fixed-count `choice` through `rng=numpy.random`, then RY-linear `randn`. `bootstrap_ci` then resamples with `rng=numpy.random` and no seed call. | None |
| Circuit depth, one cell | `numpy.random.seed(seed)` for seeds `0..4`, repeated for every `(n_bits, depth)`. The smoke cell is `n=8`, `m=4`, depth `1`. | Pattern `choice`, RY-linear `randn`, then fixed-count. There is no bootstrap. | None |
| Convergence | `numpy.random.seed` for seeds `0, 1, 2` | Pattern `choice`, then `randn` inside `train_ry_linear_qam_record_loss` | None |
| Scaling | `pennylane.numpy.random.seed(123 + trial_id + n_bits * 100)` | Pattern `choice`, Rot `randn` with no reseed, two minibatch `choice` draws per Adam step, then inline Bernoulli `rand` and one permutation per random-order sweep | None. The class default depth 3 is not the call; the call passes `min(6, n_bits)`. |
| Statistical significance, one cell | Both global seed functions set to the task seed before the pattern draw. The smoke cell is seed `0` at `n=8`, capacity `0.2`, noise `0.1`. | Pattern `choice`, Bernoulli, `normal(0, 0.1)`, Bernoulli again. `bootstrap_diff` then draws `randint` with no seed. Historical `main` calls that helper on the pool parent after the workers finish. The framework runner is single-process, so the helper continues from the last task. | None |
| Inference, one cell | `numpy.random.seed` of `42 + trial_id + int(n_bits * 1000) + int(capacity_ratio * 100) + int(noise_p * 1000)`, inside `run_trial`. The smoke cell is `n=8`, capacity `0.1`, `NOISE_LEVELS[1]`, trials `0..299`. | Pattern `choice`, one `randint` target, one inline Bernoulli `rand`. Sequential recall and Hamming recall do not draw. `aggregate_results` then resamples with no seed. Historical `main` draws that bootstrap on the pool parent. This runner draws it after the last in-process task. | None |

Capacity and depth pass an explicit `rng` into `corrupt_fixed_count`, and that object is the global NumPy stream (`onp.random`), not a private `RandomState`. The capacity helper `bootstrap_ci` also passes `rng=onp.random` into `bootstrap_mean_ci`, so those 2000 resamples continue the same stream after the trial loop. They are not given a seed of their own, and this task does not add one. Scaling and inference still use an inline `rand(n) < p` mask, so those draws stay at their place in the historical order.

## Known sources of coupling

- One legacy `RandomState` serves `numpy.random` and `pennylane.numpy.random`. A seed call from either name moves every later draw, including a draw in another module.
- The noise-sweep constructor reset to 42 discards the trial seed for every draw after `QAM.__init__`. Hopfield queries and QAM queries from the same trial integer are not the same sequence.
- Capacity, depth, and convergence reapply a small seed list at every cell. The five capacity seeds are not five seeds for the whole sweep.
- Scaling's minibatch index is drawn inside `loss`, and `opt.step` evaluates `loss` again. One extra `choice` changes the rest of that trial, including the Bernoulli masks and the Hopfield permutations.
- Inference's summary bootstrap and `scripts/plot_bootstrap_distribution.py` read whatever the legacy stream happens to be. They do not restore it.
- The historical noise sweep, statistical significance, and inference each seed inside a worker. The smoke runner is single-process. The same trial integers do not reproduce pool scheduling or wall-clock columns.
- `uuid4` for the run directory does not consume the legacy stream. It also does not restore it.

## Reproducibility guarantees and limitations

What the smoke manifest supports:

- The request's trial seeds are the integers passed to both global seed functions before the pattern draw.
- Rot full-batch construction still resets that stream to 42 before the parameter draw. The manifest records the reset. It does not perform a second one.
- The Gaussian interval on this protocol does not add a random draw.

What a recorded seed does not support:

- Equality with an archived CSV. `comparable_to_archive` stays false. The archived noise-sweep table is a 30-trial aggregate at 300 Adam steps.
- A claim that the run can be repeated exactly from the manifest alone. The legacy `RandomState` is not stored, and draws after the reset to 42 are not separately seeded.
- Treating `default_rng(seed)` as a stand-in for the legacy stream.
- Treating the inference column `model="qam"` as a variational sample. That column is Hamming nearest neighbour and does not draw.

The unseeded bootstrap helpers were left unseeded. Seeding `aggregate_results`, `bootstrap_diff`, or `plot_bootstrap_distribution` would be a change to historical protocol behaviour, so it was not done.

## Future migration plan

Isolation stays opt-in. A new helper may accept a `numpy.random.RandomState` or a `Generator`, and a new framework caller may pass one. Legacy adapters and the experiment modules keep the global seed calls until a small-cell comparison shows the same draws.

Order for that work, one protocol at a time:

1. Write down the draw order, including constructor draws, a second loss evaluation, and inline Bernoulli masks.
2. Add an explicit `rng` argument on the framework side only. Do not replace `numpy.random.seed` or `pennylane.numpy.random.seed` inside `src/` or `experiments/` in the same change.
3. Keep the noise-sweep `pnp.random.seed(42)` inside `QAM.__init__`. A wrapper that skipped it would be a different experiment.
4. Leave unseeded bootstrap functions unseeded until a separate change records the old call, the new seed, and a comparison that states the difference. Do not present that seed as the historical protocol.
5. Do not point `default_rng` at a legacy call site. Tests that need the legacy stream keep using `RandomState` or the global functions.
6. After the draws match on a small cell, the manifest can name the generator object. `exact_reproducibility_claimed` stays false until an archived comparison has been measured, and that comparison is a separate request.
