"""Framework progress logs and failure records. The failing run uses a stand-in model."""

import json
import logging

import numpy as np
import pytest
from pennylane import numpy as pnp

from benchmark.adapters import qam_rot_fullbatch
from benchmark.cli import EXIT_INVALID, EXIT_OK, main
from benchmark.log import LOGGER_NAME, configure_logging, log_event
from benchmark.output import STATUS_FAILED, STATUS_RUNNING, STATUS_SUCCEEDED
from benchmark.protocols.noise_robustness_sweep import (
    NoiseSweepRequest,
    lightweight_request,
    run_noise_sweep,
)
from src.paths import PROJECT_ROOT


class _StandInQAM:
    """Enough of the Rot constructor surface for one cell. This is not QAM."""

    def __init__(self, n_bits, depth=6, train_steps=300, lr=0.05):
        self.n_bits = n_bits
        self.depth = min(depth, n_bits)
        self.train_steps = train_steps
        self.lr = lr
        self.params = np.zeros((self.depth, n_bits, 3))

    def train(self, patterns):
        return 0.0

    def recall(self, pattern):
        return np.ones(self.n_bits), 0.0


class _FailingQAM(_StandInQAM):
    def train(self, patterns):
        raise RuntimeError("artificial failure")


def _manifests(parent):
    return [
        json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        for path in parent.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    ]


def test_logging_setup_attaches_one_handler_and_leaves_the_root_logger():
    root = logging.getLogger()
    root_handlers = list(root.handlers)
    logger = configure_logging()
    again = configure_logging()
    assert logger is again
    assert logger.name == LOGGER_NAME
    assert logger.level == logging.INFO
    marked = [handler for handler in logger.handlers if getattr(handler, "_benchmark_handler", False)]
    assert len(marked) == 1
    assert list(root.handlers) == root_handlers
    assert logging.getLogger().handlers == root_handlers


def test_events_use_the_benchmark_logger(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    log_event("run_start", cells=1, resumable=False)
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.name == LOGGER_NAME
    assert record.levelno == logging.INFO
    assert "event=run_start" in record.message
    assert "cells=1" in record.message
    assert "resumable=False" in record.message


def test_validation_failure_is_logged_and_still_reported(tmp_path, caplog, capsys):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    missing = tmp_path / "absent.json"
    assert main(["validate", str(missing)]) == EXIT_INVALID
    err = capsys.readouterr().err
    assert "event=configuration_invalid" in caplog.text
    assert str(missing) in err or "absent.json" in err
    assert any(record.exc_info for record in caplog.records if record.levelno >= logging.ERROR)


def test_validation_success_is_logged_without_a_run(caplog, capsys):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    example = PROJECT_ROOT / "configs" / "noise_robustness_sweep.smoke.json"
    assert main(["validate", str(example)]) == EXIT_OK
    assert "event=configuration_validated" in caplog.text
    assert "executable=True" in caplog.text
    assert "event=run_start" not in caplog.text
    assert "experiment_id: noise_robustness_sweep_smoke" in capsys.readouterr().out


def test_artificial_failure_propagates_and_marks_the_manifest(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    monkeypatch.setattr(qam_rot_fullbatch, "QAM", _FailingQAM)
    numpy_state = np.random.get_state()
    pennylane_state = pnp.random.get_state()
    try:
        with pytest.raises(RuntimeError, match="artificial failure") as caught:
            run_noise_sweep(lightweight_request(), output_dir=tmp_path)
    finally:
        np.random.set_state(numpy_state)
        pnp.random.set_state(pennylane_state)

    assert caught.type is RuntimeError
    assert caught.value.__traceback__ is not None
    text = caplog.text
    assert "event=run_start" in text
    assert "event=trial_start" in text
    assert "progress=1/2" in text
    assert "progress=2/2" not in text
    assert "event=model_construct" in text
    assert "procedure_id=hopfield_synchronous" in text
    assert "procedure_id=qam_rot_fullbatch" in text
    assert text.count("event=model_train") == 1
    assert text.count("event=model_evaluate") == 1
    assert "event=run_failed" in text
    assert "event=run_recorded" in text
    assert "status=failed" in text
    assert "event=run_complete" not in text
    assert "Traceback (most recent call last)" in text
    failure = next(record for record in caplog.records if "event=run_failed" in record.message)
    assert failure.exc_info is not None
    assert failure.exc_info[1] is caught.value

    manifests = _manifests(tmp_path)
    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest["execution_status"] == STATUS_FAILED
    assert manifest["execution_status"] != STATUS_SUCCEEDED
    assert manifest["resumable"] is False
    assert manifest["error_type"] == "RuntimeError"
    assert "artificial failure" in manifest["error"]
    assert "Traceback (most recent call last)" in manifest["traceback"]
    assert "artificial failure" in manifest["traceback"]
    children = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert not (children[0] / "trials.csv").exists()


def test_keyboard_interrupt_is_recorded_and_reraised(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    def boom(_request):
        raise KeyboardInterrupt("stopped by user")

    monkeypatch.setattr(
        "benchmark.protocols.noise_robustness_sweep._execute",
        boom,
    )
    with pytest.raises(KeyboardInterrupt, match="stopped by user"):
        run_noise_sweep(lightweight_request(), output_dir=tmp_path)
    manifest = _manifests(tmp_path)[0]
    assert manifest["execution_status"] == STATUS_FAILED
    assert manifest["resumable"] is False
    assert manifest["error_type"] == "KeyboardInterrupt"
    assert "stopped by user" in manifest["error"]
    assert "event=run_complete" not in caplog.text


def test_in_memory_failure_does_not_return_a_result(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    def boom(_request):
        raise RuntimeError("artificial failure")

    monkeypatch.setattr(
        "benchmark.protocols.noise_robustness_sweep._execute",
        boom,
    )
    with pytest.raises(RuntimeError, match="artificial failure"):
        run_noise_sweep(lightweight_request(), output_dir=None)
    assert "event=run_failed" in caplog.text
    assert "output=in-memory" in caplog.text
    assert "event=run_complete" not in caplog.text


def test_manifest_write_error_does_not_replace_the_original_exception(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    def boom(_request):
        raise RuntimeError("artificial failure")

    monkeypatch.setattr(
        "benchmark.protocols.noise_robustness_sweep._execute",
        boom,
    )
    real_replace = __import__("os").replace
    calls = {"n": 0}

    def fail_later(src, dst):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise OSError("manifest update failed")
        return real_replace(src, dst)

    monkeypatch.setattr("benchmark.output.os.replace", fail_later)
    with pytest.raises(RuntimeError, match="artificial failure"):
        run_noise_sweep(lightweight_request(), output_dir=tmp_path)
    assert "event=manifest_update_failed" in caplog.text
    assert "event=run_failed" in caplog.text
    manifest = _manifests(tmp_path)[0]
    assert manifest["execution_status"] == STATUS_RUNNING


def test_artificial_success_logs_progress_and_persistence(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    monkeypatch.setattr(qam_rot_fullbatch, "QAM", _StandInQAM)
    numpy_state = np.random.get_state()
    pennylane_state = pnp.random.get_state()
    request = NoiseSweepRequest(
        n_bits_list=(2,),
        corruption_rates=(0.0,),
        seeds=(0,),
        capacity_ratio=0.3,
        train_steps=1,
        grid="smoke",
    )
    try:
        result = run_noise_sweep(request, output_dir=tmp_path)
    finally:
        np.random.set_state(numpy_state)
        pnp.random.set_state(pennylane_state)

    text = caplog.text
    assert "event=execution_start" in text
    assert "event=trial_start" in text
    assert "progress=1/1" in text
    assert text.count("event=model_construct") == 2
    assert text.count("event=model_train") == 2
    assert text.count("event=model_evaluate") == 2
    assert "event=evaluation_aggregate" in text
    assert "event=persist_start" in text
    assert "event=persist_complete" in text
    assert "event=run_complete" in text
    assert "status=succeeded" in text
    assert "resumable=False" in text
    assert result["execution_status"] == STATUS_SUCCEEDED
    manifest = _manifests(tmp_path)[0]
    assert manifest["execution_status"] == STATUS_SUCCEEDED
    assert manifest["resumable"] is False
    assert manifest["error"] is None
    assert manifest["traceback"] is None


def test_a_refused_request_does_not_create_a_run(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    request = NoiseSweepRequest(
        n_bits_list=(2,),
        corruption_rates=(0.0,),
        seeds=(0,),
        train_steps=1,
        grid="full",
    )
    with pytest.raises(RuntimeError, match="only grid='smoke'"):
        run_noise_sweep(request, output_dir=tmp_path)
    assert "event=request_refused" in caplog.text
    assert list(tmp_path.iterdir()) == []
