# Execution and failure handling

Date: 2026-10-10. The framework runner logs stages to the `benchmark` logger and writes the outcome on the run manifest. Historical experiment modules still print their own lines. Those prints are unchanged.

A logged seed, a logged stage, or a manifest status is not a claim that the run can be repeated exactly, and it is not a claim that a stopped run can be resumed.

## Logging setup

`benchmark.log.configure_logging` attaches one handler to the logger named `benchmark` and sends it to stderr. The line format is time, level, logger name, then the message. The message starts with `event=` and then `key=value` fields.

The call does not use `logging.basicConfig`, so it does not replace handlers on the root logger. A second call keeps the same handler. `python -m benchmark` configures the logger before it loads a file. `run_benchmark` and `run_noise_sweep` configure it as well, so a library call emits the same events.

INFO is the level used for stages. A failure is logged at ERROR with the traceback attached to the log record.

## What is logged

| Event | When |
|---|---|
| `configuration_validated` | `validate` or `run` loaded a file and resolved its model ids. |
| `configuration_invalid` | The file is not a valid request. The process still prints the message and exits 2. |
| `configuration_not_executable` | The file loaded and the runner will not start it. |
| `dry_run` | The plan was printed and the runner was not called. |
| `configuration_accepted` | `run_benchmark` received a `BenchmarkConfig`. |
| `configuration_rejected` | That object cannot be turned into a smoke request, or a model id fails preparation. The exception is re-raised. |
| `models_prepared` | Factory arguments were checked. No circuit has been constructed. |
| `protocol_rejected` | `run_benchmark` was given an object it does not dispatch. |
| `request_refused` | The smoke guard rejected the grid, the step count, or the cell count. No run directory is created. |
| `output_refused` | The output path was rejected, including a path that resolves to `results/`. |
| `run_start` | A smoke run is about to execute. `resumable=False`. |
| `manifest_written` | `manifest.json` was written with status `running`. |
| `execution_start` | The cell list is known. |
| `trial_start` / `trial_complete` | One configured `(n_bits, corruption, seed)` cell, with `progress=index/total`. |
| `model_construct` | The adapter factory returned. Logged once per model per cell. |
| `model_train` | That model's training call returned. |
| `model_evaluate` | Recall and scoring for that model returned. The line includes the query count. |
| `evaluation_aggregate` | The Gaussian interval over the finished cells. |
| `persist_start` / `persist_complete` | `trials.csv`, `aggregates.csv`, and `trials.json`. |
| `run_recorded` | The manifest was replaced with `succeeded` or `failed`. |
| `run_complete` | The runner is about to return. Status is `succeeded` or, for an in-memory run, `in-memory`. |
| `run_failed` | An exception left the cell loop, persistence, or the completion write. The traceback is on this record. |
| `manifest_update_failed` | The runner tried to store `failed` and the manifest write itself raised. |

The cell loop follows `NoiseSweepRequest.cells`: `n_bits`, then corruption rate, then trial seed. Logging does not reorder those cells and does not call the seed functions. Both global seed calls remain the first statements inside the cell, before the pattern draw.

`model_evaluate` is one line per model after that model's recalls finish. The recall loop does not log each pattern, each Hopfield step, or each circuit evaluation. Historical modules that print a training loss, such as the RY-linear recorder, still print those lines when they run. The smoke path does not call that recorder.

## Failures

An exception inside a run that already has a directory is handled in this order:

1. `run_failed` is logged with the original traceback.
2. The manifest is replaced. `execution_status` becomes `failed`. `error` is the exception type and message. `error_type` is the type name. `traceback` is the formatted traceback. `resumable` is false. `comparable_to_archive` stays false.
3. The original exception is re-raised. The runner does not return a result object after that exception.

That handler covers `Exception` and other `BaseException` values raised inside the cell loop or while the tables and the success manifest are being written, including `KeyboardInterrupt`. The success status is written only after the tables exist and `complete_run` returns.

If the manifest write in step 2 fails, `manifest_update_failed` is logged and the original exception is still the one that propagates. The directory can then still say `running`, because the replacement did not land.

These cases do not get a `failed` manifest, because the runner has no manifest to update:

- The smoke guard rejects the request before a directory is created.
- The output path is refused before `begin_run`.
- The process is killed, or the machine stops, after `running` is written and before the handler runs. A later reader can see `running` for a process that is gone. That status is not a job waiting to continue.

An in-memory run (`output_dir=None`) has no manifest. A failure is logged and re-raised. Nothing is returned in its place.

The command line still prints the validation or refusal message and uses the same exit codes as `docs/cli_usage.md`. The log line does not replace that message.

## Resume

`resumable` is false on every manifest this writer produces, including a successful one. The runner does not look for an earlier directory, does not skip finished cells, and does not read a checkpoint. A failed run stays in its directory. Another launch allocates a new run id.

There is no checkpoint file and no resume command. An interrupted experiment is not resumed.

## What this layer does not change

The smoke cell still seeds both global generators, draws patterns, trains synchronous Hopfield, then constructs Rot full-batch QAM, which still reseeds to 42. Scores and the Gaussian interval are the same functions as before. The runner stays single-process. It does not add a pool, and it does not call an experiment `main`.

The other six historical protocols are not dispatched. Their print statements and their own pools stay in those modules.
