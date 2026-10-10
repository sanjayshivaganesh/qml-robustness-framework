"""Orchestrate one explicit smoke request.

No command-line parser, no plotting, and no call into an experiment ``main``.
The wired protocols are the paired noise sweep, the scaling schedule, the
statistical-significance cell schedule, the storage-capacity cell schedule,
the circuit-depth cell schedule, and the inference task schedule.
A configuration is resolved through the model registry before any adapter
factory is called.
"""

import json
from pathlib import Path

from benchmark.config_loader import BenchmarkConfig
from benchmark.log import configure_logging, log_event, log_failure
from benchmark.protocols.noise_robustness_sweep import (
    PROCEDURE_IDS,
    NoiseSweepRequest,
    run_noise_sweep,
)
from benchmark.protocols.scaling import ScalingRequest, run_scaling
from benchmark.protocols.statistical_significance import (
    StatisticalRequest,
    run_statistical_significance,
)
from benchmark.protocols.circuit_depth import (
    CircuitDepthRequest,
    run_circuit_depth,
)
from benchmark.protocols.storage_capacity import (
    StorageCapacityRequest,
    depth_for,
    run_storage_capacity,
)
from benchmark.circuit_depth_config import CircuitDepthConfig
from benchmark.noise_robustness_inference_config import InferenceConfig
from benchmark.protocols.noise_robustness_inference import (
    InferenceRequest,
    run_inference,
)
from benchmark.registry import get, prepare
from benchmark.scaling_config import ScalingConfig
from benchmark.statistical_significance_config import StatisticalSignificanceConfig
from benchmark.storage_capacity_config import StorageCapacityConfig


class UnknownProtocol(LookupError):
    """The runner has no orchestration for this protocol."""


def run_benchmark(
    request,
    *,
    output_dir=None,
    overwrite: bool = False,
    allow_historical_grid: bool = False,
):
    """Compose dataset, models, corruption, recall, metrics, and optional output.

    ``request`` is a noise-sweep request, a scaling request, a
    statistical-significance request, or the matching configuration object.
    Other protocols are not dispatched. ``output_dir``
    must be outside the archived results tree. For a configuration object,
    omit ``output_dir`` to use the file's destination, or pass a directory to
    override it. Omit both only when the result should stay in memory. A
    directory is a parent for a new run id unless ``overwrite`` is true.
    ``overwrite`` does not allow the archived results tree.
    ``allow_historical_grid`` starts that protocol's module grid. Any other
    request passed with that flag is refused.
    """
    configure_logging()
    if isinstance(request, BenchmarkConfig):
        log_event(
            "configuration_accepted",
            experiment_id=request.experiment_id,
            protocol=request.protocol,
            train_steps=request.train_steps,
            executable=request.executable,
        )
        if output_dir is None:
            output_dir = request.resolved_output_dir()
        try:
            runner_request = request.to_runner_request(
                allow_historical_grid=allow_historical_grid
            )
            _prepare_configured_models(request)
        except Exception as exc:
            log_failure(
                "configuration_rejected",
                exc,
                experiment_id=request.experiment_id,
                protocol=request.protocol,
            )
            raise
        log_event(
            "models_prepared",
            experiment_id=request.experiment_id,
            procedure_ids=" ".join(request.procedure_ids),
        )
        return run_noise_sweep(
            runner_request,
            output_dir=output_dir,
            overwrite=overwrite,
            experiment_id=request.experiment_id,
            configuration=_configuration_record(request),
            allow_historical_grid=allow_historical_grid,
        )
    if isinstance(request, ScalingConfig):
        return _run_scaling_config(
            request,
            output_dir=output_dir,
            overwrite=overwrite,
            allow_historical_grid=allow_historical_grid,
        )
    if isinstance(request, ScalingRequest):
        return run_scaling(
            request,
            output_dir=output_dir,
            overwrite=overwrite,
            allow_historical_grid=allow_historical_grid,
        )
    if isinstance(request, StatisticalSignificanceConfig):
        return _run_statistical_config(
            request,
            output_dir=output_dir,
            overwrite=overwrite,
            allow_historical_grid=allow_historical_grid,
        )
    if isinstance(request, StatisticalRequest):
        return run_statistical_significance(
            request,
            output_dir=output_dir,
            overwrite=overwrite,
            allow_historical_grid=allow_historical_grid,
        )
    if isinstance(request, StorageCapacityConfig):
        return _run_storage_config(
            request,
            output_dir=output_dir,
            overwrite=overwrite,
            allow_historical_grid=allow_historical_grid,
        )
    if isinstance(request, StorageCapacityRequest):
        return run_storage_capacity(
            request,
            output_dir=output_dir,
            overwrite=overwrite,
            allow_historical_grid=allow_historical_grid,
        )
    if isinstance(request, CircuitDepthConfig):
        return _run_circuit_depth_config(
            request,
            output_dir=output_dir,
            overwrite=overwrite,
            allow_historical_grid=allow_historical_grid,
        )
    if isinstance(request, CircuitDepthRequest):
        return run_circuit_depth(
            request,
            output_dir=output_dir,
            overwrite=overwrite,
            allow_historical_grid=allow_historical_grid,
        )
    if isinstance(request, InferenceConfig):
        return _run_inference_config(
            request,
            output_dir=output_dir,
            overwrite=overwrite,
            allow_historical_grid=allow_historical_grid,
        )
    if isinstance(request, InferenceRequest):
        return run_inference(
            request,
            output_dir=output_dir,
            overwrite=overwrite,
            allow_historical_grid=allow_historical_grid,
        )
    if not isinstance(request, NoiseSweepRequest):
        exc = UnknownProtocol(
            "run_benchmark accepts a noise-sweep smoke request or an executable "
            "BenchmarkConfig. Other protocols are not dispatched."
        )
        log_failure("protocol_rejected", exc)
        raise exc
    return run_noise_sweep(
        request,
        output_dir=output_dir,
        overwrite=overwrite,
        allow_historical_grid=allow_historical_grid,
    )


def _run_scaling_config(config, *, output_dir, overwrite, allow_historical_grid):
    log_event(
        "configuration_accepted",
        experiment_id=config.experiment_id,
        protocol=config.protocol,
        steps=config.steps,
        executable=config.executable,
    )
    if output_dir is None:
        output_dir = config.resolved_output_dir()
    try:
        runner_request = config.to_runner_request(allow_historical_grid=allow_historical_grid)
        _prepare_scaling_models(config)
    except Exception as exc:
        log_failure(
            "configuration_rejected",
            exc,
            experiment_id=config.experiment_id,
            protocol=config.protocol,
        )
        raise
    log_event(
        "models_prepared",
        experiment_id=config.experiment_id,
        procedure_ids=" ".join(config.procedure_ids),
    )
    return run_scaling(
        runner_request,
        output_dir=output_dir,
        overwrite=overwrite,
        experiment_id=config.experiment_id,
        configuration=_scaling_configuration_record(config),
        allow_historical_grid=allow_historical_grid,
    )


def _prepare_scaling_models(config):
    """Check scaling factory arguments. Do not construct."""
    if tuple(config.procedure_ids) != ("hopfield_async_random", "qam_rot_minibatch"):
        raise UnknownProtocol(
            "scaling requires hopfield_async_random then qam_rot_minibatch, "
            f"got {config.procedure_ids}"
        )
    for procedure_id in config.procedure_ids:
        get(procedure_id)
    for n_bits in config.n_bits_list:
        prepare("hopfield_async_random", {"n_bits": n_bits, "steps": 10})
        prepare(
            "qam_rot_minibatch",
            {
                "n_bits": n_bits,
                "depth": min(6, n_bits),
                "steps": config.steps,
                "lr": config.lr,
            },
        )


def _scaling_configuration_record(config):
    source = Path(config.source_path)
    document = None
    if source.is_file():
        document = json.loads(source.read_text(encoding="utf-8"))
    return {
        "source_path": config.source_path,
        "experiment_id": config.experiment_id,
        "protocol": config.protocol,
        "procedure_ids": list(config.procedure_ids),
        "grid": config.grid,
        "n_bits_list": list(config.n_bits_list),
        "noise_levels": list(config.noise_levels),
        "trial_ids": list(config.trial_ids),
        "steps": config.steps,
        "lr": config.lr,
        "seed_base": config.seed_base,
        "output_destination": config.output_destination,
        "document": document,
    }


def _run_statistical_config(config, *, output_dir, overwrite, allow_historical_grid):
    log_event(
        "configuration_accepted",
        experiment_id=config.experiment_id,
        protocol=config.protocol,
        tasks=(
            len(config.n_bits_list)
            * len(config.capacity_ratios)
            * len(config.corruption_levels)
            * len(config.seeds)
        ),
        executable=config.executable,
    )
    if output_dir is None:
        output_dir = config.resolved_output_dir()
    try:
        runner_request = config.to_runner_request(allow_historical_grid=allow_historical_grid)
        _prepare_statistical_models(config)
    except Exception as exc:
        log_failure(
            "configuration_rejected",
            exc,
            experiment_id=config.experiment_id,
            protocol=config.protocol,
        )
        raise
    log_event(
        "models_prepared",
        experiment_id=config.experiment_id,
        procedure_ids=" ".join(config.procedure_ids),
    )
    return run_statistical_significance(
        runner_request,
        output_dir=output_dir,
        overwrite=overwrite,
        experiment_id=config.experiment_id,
        configuration=_statistical_configuration_record(config),
        allow_historical_grid=allow_historical_grid,
    )


def _prepare_statistical_models(config):
    """Check PauliX and synchronous Hopfield arguments. Do not construct."""
    if tuple(config.procedure_ids) != ("hopfield_synchronous", "qam_paulix"):
        raise UnknownProtocol(
            "statistical_significance requires hopfield_synchronous then qam_paulix, "
            f"got {config.procedure_ids}"
        )
    for procedure_id in config.procedure_ids:
        get(procedure_id)
    for n_bits in config.n_bits_list:
        prepare("hopfield_synchronous", {"n_bits": n_bits, "steps": 10})
        prepare("qam_paulix", {"n_bits": n_bits})


def _statistical_configuration_record(config):
    source = Path(config.source_path)
    document = None
    if source.is_file():
        document = json.loads(source.read_text(encoding="utf-8"))
    return {
        "source_path": config.source_path,
        "experiment_id": config.experiment_id,
        "protocol": config.protocol,
        "procedure_ids": list(config.procedure_ids),
        "grid": config.grid,
        "n_bits_list": list(config.n_bits_list),
        "capacity_ratios": list(config.capacity_ratios),
        "corruption_levels": list(config.corruption_levels),
        "seeds": list(config.seeds),
        "bootstrap_samples": config.bootstrap_samples,
        "output_destination": config.output_destination,
        "document": document,
    }


def _run_storage_config(config, *, output_dir, overwrite, allow_historical_grid):
    cells = sum(len(m_values) for _n_bits, m_values in config.widths)
    log_event(
        "configuration_accepted",
        experiment_id=config.experiment_id,
        protocol=config.protocol,
        cells=cells,
        train_steps=config.train_steps,
        executable=config.executable,
    )
    if output_dir is None:
        output_dir = config.resolved_output_dir()
    try:
        runner_request = config.to_runner_request(allow_historical_grid=allow_historical_grid)
        _prepare_storage_models(config)
    except Exception as exc:
        log_failure(
            "configuration_rejected",
            exc,
            experiment_id=config.experiment_id,
            protocol=config.protocol,
        )
        raise
    log_event(
        "models_prepared",
        experiment_id=config.experiment_id,
        procedure_ids=" ".join(config.procedure_ids),
    )
    return run_storage_capacity(
        runner_request,
        output_dir=output_dir,
        overwrite=overwrite,
        experiment_id=config.experiment_id,
        configuration=_storage_configuration_record(config),
        allow_historical_grid=allow_historical_grid,
    )


def _prepare_storage_models(config):
    """Check synchronous Hopfield and RY-linear arguments. Do not construct a circuit."""
    if tuple(config.procedure_ids) != ("hopfield_synchronous", "qam_ry_linear"):
        raise UnknownProtocol(
            "storage_capacity requires hopfield_synchronous then qam_ry_linear, "
            f"got {config.procedure_ids}"
        )
    for procedure_id in config.procedure_ids:
        get(procedure_id)
    for n_bits, _m_values in config.widths:
        prepare("hopfield_synchronous", {"n_bits": n_bits, "steps": 10})
        prepare(
            "qam_ry_linear",
            {"n_bits": n_bits, "depth": depth_for(n_bits), "return_tuple": True},
        )


def _run_circuit_depth_config(config, *, output_dir, overwrite, allow_historical_grid):
    cells = len(config.cases) * len(config.depths)
    log_event(
        "configuration_accepted",
        experiment_id=config.experiment_id,
        protocol=config.protocol,
        cells=cells,
        train_steps=config.train_steps,
        executable=config.executable,
    )
    if output_dir is None:
        output_dir = config.resolved_output_dir()
    try:
        runner_request = config.to_runner_request(allow_historical_grid=allow_historical_grid)
        _prepare_circuit_depth_models(config)
    except Exception as exc:
        log_failure(
            "configuration_rejected",
            exc,
            experiment_id=config.experiment_id,
            protocol=config.protocol,
        )
        raise
    log_event(
        "models_prepared",
        experiment_id=config.experiment_id,
        procedure_ids=" ".join(config.procedure_ids),
    )
    return run_circuit_depth(
        runner_request,
        output_dir=output_dir,
        overwrite=overwrite,
        experiment_id=config.experiment_id,
        configuration=_circuit_depth_configuration_record(config),
        allow_historical_grid=allow_historical_grid,
    )


def _prepare_circuit_depth_models(config):
    """Check the depth-script RY-linear arguments. Do not construct a circuit."""
    if tuple(config.procedure_ids) != ("qam_ry_linear",):
        raise UnknownProtocol(
            "circuit_depth requires qam_ry_linear alone, "
            f"got {config.procedure_ids}"
        )
    get("qam_ry_linear")
    for n_bits, _m in config.cases:
        for depth in config.depths:
            prepare(
                "qam_ry_linear",
                {"n_bits": n_bits, "depth": depth, "return_tuple": False},
            )


def _circuit_depth_configuration_record(config):
    source = Path(config.source_path)
    document = None
    if source.is_file():
        document = json.loads(source.read_text(encoding="utf-8"))
    return {
        "source_path": config.source_path,
        "experiment_id": config.experiment_id,
        "protocol": config.protocol,
        "procedure_ids": list(config.procedure_ids),
        "grid": config.grid,
        "cases": [{"n_bits": n_bits, "m": m} for n_bits, m in config.cases],
        "depths": list(config.depths),
        "seeds": list(config.seeds),
        "train_steps": config.train_steps,
        "lr": config.lr,
        "recall_noise": config.recall_noise,
        "return_tuple": False,
        "output_destination": config.output_destination,
        "document": document,
    }


def _run_inference_config(config, *, output_dir, overwrite, allow_historical_grid):
    tasks = (
        len(config.n_bits_list)
        * len(config.capacity_ratios)
        * len(config.noise_levels)
        * len(config.trial_ids)
    )
    log_event(
        "configuration_accepted",
        experiment_id=config.experiment_id,
        protocol=config.protocol,
        tasks=tasks,
        training=False,
        executable=config.executable,
    )
    if output_dir is None:
        output_dir = config.resolved_output_dir()
    try:
        runner_request = config.to_runner_request(allow_historical_grid=allow_historical_grid)
        _prepare_inference_models(config)
    except Exception as exc:
        log_failure(
            "configuration_rejected",
            exc,
            experiment_id=config.experiment_id,
            protocol=config.protocol,
        )
        raise
    log_event(
        "models_prepared",
        experiment_id=config.experiment_id,
        procedure_ids=" ".join(config.procedure_ids),
        training=False,
    )
    return run_inference(
        runner_request,
        output_dir=output_dir,
        overwrite=overwrite,
        experiment_id=config.experiment_id,
        configuration=_inference_configuration_record(config),
        allow_historical_grid=allow_historical_grid,
    )


def _prepare_inference_models(config):
    """Check the two inference procedures. Do not store patterns or recall."""
    if tuple(config.procedure_ids) != ("hopfield_async_sequential", "hamming_nearest"):
        raise UnknownProtocol(
            "noise_robustness_inference requires hopfield_async_sequential "
            f"then hamming_nearest, got {config.procedure_ids}"
        )
    get("hopfield_async_sequential")
    get("hamming_nearest")


def _inference_configuration_record(config):
    source = Path(config.source_path)
    document = None
    if source.is_file():
        document = json.loads(source.read_text(encoding="utf-8"))
    return {
        "source_path": config.source_path,
        "experiment_id": config.experiment_id,
        "protocol": config.protocol,
        "procedure_ids": list(config.procedure_ids),
        "grid": config.grid,
        "n_bits_list": list(config.n_bits_list),
        "capacity_ratios": list(config.capacity_ratios),
        "noise_levels": list(config.noise_levels),
        "trial_ids": list(config.trial_ids),
        "output_destination": config.output_destination,
        "document": document,
    }


def _storage_configuration_record(config):
    source = Path(config.source_path)
    document = None
    if source.is_file():
        document = json.loads(source.read_text(encoding="utf-8"))
    return {
        "source_path": config.source_path,
        "experiment_id": config.experiment_id,
        "protocol": config.protocol,
        "procedure_ids": list(config.procedure_ids),
        "grid": config.grid,
        "widths": [
            {"n_bits": n_bits, "m_values": list(m_values)} for n_bits, m_values in config.widths
        ],
        "seeds": list(config.seeds),
        "train_steps": config.train_steps,
        "lr": config.lr,
        "recall_noise": config.recall_noise,
        "bootstrap_samples": config.bootstrap_samples,
        "output_destination": config.output_destination,
        "document": document,
    }


def _configuration_record(config):
    """Parsed fields plus the request document, when the source file is readable."""
    source = Path(config.source_path)
    document = None
    if source.is_file():
        document = json.loads(source.read_text(encoding="utf-8"))
    return {
        "source_path": config.source_path,
        "experiment_id": config.experiment_id,
        "protocol": config.protocol,
        "procedure_ids": list(config.procedure_ids),
        "grid": config.grid,
        "n_bits_list": list(config.n_bits_list),
        "capacity_ratio": config.capacity_ratio,
        "corruption_levels": list(config.corruption_levels),
        "trial_seeds": list(config.trial_seeds),
        "train_steps": config.train_steps,
        "output_destination": config.output_destination,
        "document": document,
    }


def _prepare_configured_models(config):
    """Resolve config procedure ids and check factory arguments. Do not construct."""
    if tuple(config.procedure_ids) != PROCEDURE_IDS:
        raise UnknownProtocol(
            f"noise_robustness_sweep requires {PROCEDURE_IDS}, got {config.procedure_ids}"
        )
    for procedure_id in config.procedure_ids:
        get(procedure_id)
    for n_bits in config.n_bits_list:
        prepare("hopfield_synchronous", {"n_bits": n_bits, "steps": 10})
        prepare(
            "qam_rot_fullbatch",
            {
                "n_bits": n_bits,
                "train_steps": config.train_steps,
                "depth_argument": 6,
                "lr": 0.05,
            },
        )
