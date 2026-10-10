# Phase 2 validation

Date: 2026-10-10. This is an audit of the framework built on Days 3–10. No adapter, loader, registry, or runner was rewritten. No experiment `main` was called. The historical grids were not run.

Command for the suite: `MPLCONFIGDIR="$PWD/.mplconfig" ./venv/bin/python -m pytest -q`. Result: 142 passed. `git status -- results` was empty after that run. `src/metrics.py` has no diff against `HEAD`.

`docs/phase1_status.md` still says the config loader was not yet validated. That sentence is the Phase 1 record. The loader checks below supersede it. The Phase 1 file was left as it was.

## Verified

These were observed in this audit, either by the test suite or by a direct call that does not enter `main`.

| Check | What was observed |
|---|---|
| Eight ids, eight objects | `registered_ids()` is `hopfield_synchronous`, `hopfield_async_sequential`, `hopfield_async_random`, `qam_ry_linear`, `qam_rot_fullbatch`, `qam_rot_minibatch`, `qam_paulix`, `hamming_nearest`. The eight `Registration.implementation` objects have distinct identities. |
| Three Hopfield procedures | Synchronous registration is `src.hopfield.Hopfield` and its factory is `build`. Sequential factory is `store` (`hopfield_hebbian_weights`); the object stored on that registration is `hopfield_async_sequential_recall`. Random-order registration is `HopfieldAsyncRandom`. Those three objects are not the same. |
| Variational QAM procedures | Noise-sweep `QAM` and scaling `QAM` are different classes. RY-linear factory is `build_circuit`; the stored object is `train_ry_linear_qam`. PauliX factory is `build_qam`. PauliX `recall` is `experiments.statistical_significance.qam_recall` and is not `src.qam.qam_recall`. |
| Hamming | Registration object is `nearest_neighbor_recall`. The adapter has no `train`. It is not `qam_recall` and not either Rot class. |
| Unknown id | `get("qam")` and `prepare("qam", ...)` raise `UnknownModel` before a constructor runs. Covered by `tests/test_registry.py`. |
| Config load does not execute | With `experiments.noise_robustness_sweep.QAM.__init__` and every experiment `main` replaced by functions that raise, `load_benchmark_config("configs/noise_robustness_sweep.smoke.json")` returned an executable `BenchmarkConfig` and neither replacement was called. `results/` bytes were unchanged. `runs/` was not created. The same claim is in `tests/test_config_loader.py::test_loading_does_not_construct_a_circuit`. |
| Small configured run | `tests/test_registry.py::test_configured_smoke_runs_end_to_end` loads that file and calls `run_benchmark(loaded, output_dir=tmp_path / "smoke")`. Manifest, `trials.csv`, and `aggregates.csv` record `hopfield_synchronous` and `qam_rot_fullbatch` for both seeds. `main_invoked` is false. `comparable_to_archive` is false. The output path is under `tmp_path`. |
| Archive overwrite | The same test compares `results/` file bytes before and after and finds them equal. Separately, `run_benchmark` on that loaded config with `output_dir` set to `results/noise_robustness_sweep` raised `ArchivedResultsRefused` before `QAM.__init__`. The directory was not created and the archive bytes were unchanged. `tests/test_smoke_noise_sweep.py` covers the same guard for a `NoiseSweepRequest`. `tests/test_config_loader.py` rejects a destination of `results/noise_robustness_sweep` at load time, including through a symlink. |
| Seed reset, one smoke cell | For `n_bits=2`, corruption `0.0`, seed `0`, `train_steps=1`, the smoke cell's patterns and Hopfield query matched `np.random.choice` and `corrupt_bernoulli` after both global seed calls. Its recorded init parameters matched `0.01 * randn` drawn after `pennylane.numpy.random.seed(42)`, and did not match a draw that continued an unrelated seed. `QAM(2)` with no `train_steps` argument still has `train_steps == 300`. |
| Seed reset, two stored patterns | For `n_bits=8`, corruption `0.25`, seed `3`, `train_steps=1`, `m = max(1, int(0.3 * 8))` is 2. The smoke cell matched a direct loop: Hopfield `train` and per-pattern `corrupt_bernoulli` then `recall_timed`, then `QAM(n_bits, train_steps=1)`, then per-pattern Bernoulli and `recall`. Init parameters matched the direct constructor. Hopfield queries and QAM queries were different arrays. Depth used was 6. Learning rate was 0.05. This call trains one Adam step. It is not `run_trial` at 300 steps and it is not `main`. |
| Fixed-count rule | `git diff` does not change the executable line `n_flip = max(1, int(len(pattern) * noise_level))`. On pattern `[1, -1, 1, -1]` and `RandomState(1)`, levels `0.0`, `0.15`, and `0.5`, `corrupt_fixed_count`, `apply_fixed_count`, and the function from `HEAD:src/data_generation.py` returned the same vectors. Each flip count was `max(1, int(4 * level))`, so level 0 still flips one bit. |
| Scientific diff | Comments, docstrings, and print text differ under `src/hopfield.py`, `src/qam.py`, `src/data_generation.py`, and `experiments/*.py`. Numeric assignments that appear in the diff (`NOISE_LEVELS`, `RANDOM_SEED_BASE`, `TRIALS_PER_CONFIG`, `n_bits_list`, `trials_per_m`, `recall_noise`, and the capacity and Hopfield factors) have the same values on both sides. `pnp.random.seed(42)` is not part of the diff. Archived CSVs under `results/` are untouched. |

The noise-sweep class still executes `pnp.random.seed(42)` inside `QAM.__init__` before `np.random.randn`. The adapter does not add another seed call. The smoke passes `train_steps` and does not pass `depth` or `lr`, so those stay the class defaults 6 and 0.05. Depth is then `min(6, n_bits)` inside the class.

## Assumptions

These were not established by a measurement in this audit.

| Item | Why it is not a verified result |
|---|---|
| Archived CSV equality | No smoke row was compared with `results/noise_robustness_sweep/summary/noise_robustness_sweep.csv`. `comparable_to_archive` stays false. |
| The 300-step, 600-cell noise sweep | Refused by the runner. Not executed. The class default `train_steps=300` was read off a constructed `QAM(2)` that was not trained. |
| Other smoke seeds and widths | The numeric match covers seed 0 at `n_bits=2`, corruption 0, and seed 3 at `n_bits=8`, corruption 0.25. The cell function uses the same order for every seed. That is code inspection, not a sweep. |
| Phase 1 recall and metric checks | Still the evidence in `docs/regression_validation.md` and `docs/phase1_status.md`. This audit did not repeat those hand-pattern comparisons except where the suite runs them. |
| Rot minibatch training, PauliX training, RY-linear training through the runner | Registered and not dispatched. Their earlier small comparisons were not re-run as part of this audit beyond the suite. |
| Inline Bernoulli in scaling and inference | Those modules still contain their own masks. This audit did not replay their full RNG order. |
| Circuit-depth inline `sign` | Still not routed through `qam_recall`. Not re-executed here. |
| A future protocol will refuse `results/` | Only the current writer, `benchmark.output.create_run_dir`, was checked. A new runner that writes somewhere else would not inherit that check automatically. |

## Unresolved defects

No integration defect showed up. Nothing in the registry, loader, smoke runner, or output guard was changed.

The open items are gaps, not failing checks: other protocols have no runner, the historical grid is refused on purpose, and archive equality is unmeasured. `docs/phase2_readiness.md` lists what has to be checked before one of those protocols is moved.

## Ready for execution-layer development

Yes, for the next layer of protocol runners on top of this registry, this loader, and `benchmark.output`.

The next runner should keep the current limits: resolve a real procedure id, call the adapter that already exists, write outside `results/`, and refuse the historical grid. It should not call an experiment `main`, and it should not describe its output as the archived experiment until a comparison has been measured.

It is not a reason to start the 300-step noise sweep, the inference pool, or a general CLI.
