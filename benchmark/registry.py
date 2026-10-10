"""Map procedure ids to the adapters that already exist.

``prepare`` checks model-specific parameters and returns the adapter
factory without calling it. Nothing here trains a model or copies a circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

from benchmark.adapters import (
    hamming_nearest,
    hopfield_async_random,
    hopfield_async_sequential,
    hopfield_synchronous,
    qam_paulix,
    qam_rot_fullbatch,
    qam_rot_minibatch,
    qam_ry_linear,
)


class UnknownModel(KeyError):
    """The procedure id is not a registered adapter."""

    def __init__(self, procedure_id):
        self.procedure_id = procedure_id
        known = ", ".join(sorted(_REGISTRY))
        super().__init__(
            f"unknown model {procedure_id!r}. Registered models are: {known}. "
            "'qam' is not a procedure id."
        )


class InvalidModelParameters(ValueError):
    """Parameters do not match this adapter's factory. Construction did not start."""


class ModelHasNoConstructor(LookupError):
    """This procedure is registered and is not built with a constructor."""

    def __init__(self, procedure_id, how):
        self.procedure_id = procedure_id
        super().__init__(f"{procedure_id} has no constructor. {how}")


@dataclass(frozen=True)
class Registration:
    """One adapted procedure. ``implementation`` is the original callable or class."""

    procedure_id: str
    adapter: ModuleType
    constructor_name: str | None
    implementation: object

    def factory(self):
        if self.constructor_name is None:
            raise ModelHasNoConstructor(
                self.procedure_id,
                "Use recall(patterns, query). There is no training method.",
            )
        return getattr(self.adapter, self.constructor_name)


def _registration(adapter, constructor_name, implementation):
    procedure_id = adapter.identity.procedure_id
    if adapter.identity.source is None:
        raise RuntimeError(f"{procedure_id} has no source")
    return Registration(
        procedure_id=procedure_id,
        adapter=adapter,
        constructor_name=constructor_name,
        implementation=implementation,
    )


_REGISTRY = {
    entry.procedure_id: entry
    for entry in (
        _registration(hopfield_synchronous, "build", hopfield_synchronous.implementation),
        _registration(
            hopfield_async_sequential,
            "store",
            hopfield_async_sequential.recall_implementation,
        ),
        _registration(hopfield_async_random, "build", hopfield_async_random.implementation),
        _registration(
            qam_ry_linear,
            "build_circuit",
            qam_ry_linear.train_implementation,
        ),
        _registration(qam_rot_fullbatch, "build", qam_rot_fullbatch.implementation),
        _registration(qam_rot_minibatch, "build", qam_rot_minibatch.implementation),
        _registration(qam_paulix, "build", qam_paulix.build_implementation),
        _registration(hamming_nearest, None, hamming_nearest.implementation),
    )
}


def registered_ids():
    """Procedure ids in registry order."""
    return tuple(_REGISTRY)


def get(procedure_id):
    """Return the registration or raise ``UnknownModel``."""
    try:
        return _REGISTRY[procedure_id]
    except KeyError:
        raise UnknownModel(procedure_id) from None


def prepare(procedure_id, parameters):
    """Validate ``parameters`` and return ``(factory, args, kwargs)``.

    The factory is not called. Unknown ids fail before any parameter check
    that would construct a model.
    """
    registration = get(procedure_id)
    if not isinstance(parameters, dict):
        raise InvalidModelParameters(
            f"{procedure_id}: parameters must be a dict, got {type(parameters).__name__}"
        )
    args, kwargs = _arguments(procedure_id, parameters)
    return registration.factory(), args, kwargs


def _arguments(procedure_id, parameters):
    checkers = {
        "hopfield_synchronous": _hopfield_synchronous_arguments,
        "hopfield_async_sequential": _sequential_arguments,
        "hopfield_async_random": _async_random_arguments,
        "qam_ry_linear": _ry_linear_arguments,
        "qam_rot_fullbatch": _rot_fullbatch_arguments,
        "qam_rot_minibatch": _rot_minibatch_arguments,
        "qam_paulix": _paulix_arguments,
        "hamming_nearest": _hamming_arguments,
    }
    return checkers[procedure_id](parameters)


def _reject_unknown(procedure_id, parameters, allowed):
    unknown = sorted(set(parameters) - set(allowed))
    if unknown:
        raise InvalidModelParameters(
            f"{procedure_id}: unknown parameter(s) {unknown}. Allowed keys are {sorted(allowed)}."
        )


def _require(procedure_id, parameters, key):
    if key not in parameters:
        raise InvalidModelParameters(f"{procedure_id}: missing required parameter {key!r}")


def _positive_int(procedure_id, parameters, key):
    value = parameters[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidModelParameters(
            f"{procedure_id}: {key} must be an integer, got {value!r}. Values are not coerced."
        )
    if value < 1:
        raise InvalidModelParameters(f"{procedure_id}: {key} must be >= 1, got {value}")
    return value


def _hopfield_synchronous_arguments(parameters):
    _reject_unknown("hopfield_synchronous", parameters, {"n_bits", "steps"})
    _require("hopfield_synchronous", parameters, "n_bits")
    _require("hopfield_synchronous", parameters, "steps")
    n_bits = _positive_int("hopfield_synchronous", parameters, "n_bits")
    steps = _positive_int("hopfield_synchronous", parameters, "steps")
    if steps == 10:
        return (n_bits,), {}
    return (n_bits,), {"steps": steps}


def _sequential_arguments(parameters):
    _reject_unknown("hopfield_async_sequential", parameters, {"patterns"})
    _require("hopfield_async_sequential", parameters, "patterns")
    return (parameters["patterns"],), {}


def _async_random_arguments(parameters):
    _reject_unknown("hopfield_async_random", parameters, {"n_bits", "steps"})
    _require("hopfield_async_random", parameters, "n_bits")
    n_bits = _positive_int("hopfield_async_random", parameters, "n_bits")
    if "steps" not in parameters:
        return (n_bits,), {}
    steps = _positive_int("hopfield_async_random", parameters, "steps")
    if steps == 10:
        return (n_bits,), {}
    return (n_bits,), {"steps": steps}


def _ry_linear_arguments(parameters):
    _reject_unknown("qam_ry_linear", parameters, {"n_bits", "depth", "return_tuple"})
    _require("qam_ry_linear", parameters, "n_bits")
    _require("qam_ry_linear", parameters, "depth")
    n_bits = _positive_int("qam_ry_linear", parameters, "n_bits")
    depth = _positive_int("qam_ry_linear", parameters, "depth")
    kwargs = {}
    if "return_tuple" in parameters:
        value = parameters["return_tuple"]
        if not isinstance(value, bool):
            raise InvalidModelParameters(
                f"qam_ry_linear: return_tuple must be a boolean, got {value!r}"
            )
        kwargs["return_tuple"] = value
    return (n_bits, depth), kwargs


def _rot_fullbatch_arguments(parameters):
    _reject_unknown(
        "qam_rot_fullbatch",
        parameters,
        {"n_bits", "train_steps", "depth_argument", "lr"},
    )
    _require("qam_rot_fullbatch", parameters, "n_bits")
    _require("qam_rot_fullbatch", parameters, "train_steps")
    n_bits = _positive_int("qam_rot_fullbatch", parameters, "n_bits")
    train_steps = _positive_int("qam_rot_fullbatch", parameters, "train_steps")
    if "depth_argument" in parameters and parameters["depth_argument"] != 6:
        raise InvalidModelParameters(
            "qam_rot_fullbatch: depth_argument must be 6, "
            f"got {parameters['depth_argument']!r}. "
            "The noise-sweep call does not pass a different depth."
        )
    if "depth_argument" in parameters and (
        isinstance(parameters["depth_argument"], bool)
        or not isinstance(parameters["depth_argument"], int)
    ):
        raise InvalidModelParameters(
            "qam_rot_fullbatch: depth_argument must be the integer 6, "
            f"got {parameters['depth_argument']!r}"
        )
    if "lr" in parameters and parameters["lr"] != 0.05:
        raise InvalidModelParameters(
            f"qam_rot_fullbatch: lr must be 0.05, got {parameters['lr']!r}. "
            "The noise-sweep call does not pass a different learning rate."
        )
    return (n_bits,), {"train_steps": train_steps}


def _rot_minibatch_arguments(parameters):
    allowed = {"n_bits", "depth", "steps", "lr"}
    _reject_unknown("qam_rot_minibatch", parameters, allowed)
    for key in allowed:
        _require("qam_rot_minibatch", parameters, key)
    n_bits = _positive_int("qam_rot_minibatch", parameters, "n_bits")
    depth = _positive_int("qam_rot_minibatch", parameters, "depth")
    steps = _positive_int("qam_rot_minibatch", parameters, "steps")
    lr = parameters["lr"]
    if isinstance(lr, bool) or not isinstance(lr, (int, float)):
        raise InvalidModelParameters(
            f"qam_rot_minibatch: lr must be a number, got {lr!r}. Values are not coerced."
        )
    return (n_bits,), {"depth": depth, "steps": steps, "lr": lr}


def _paulix_arguments(parameters):
    _reject_unknown("qam_paulix", parameters, {"n_bits"})
    _require("qam_paulix", parameters, "n_bits")
    n_bits = _positive_int("qam_paulix", parameters, "n_bits")
    return (n_bits,), {}


def _hamming_arguments(parameters):
    raise ModelHasNoConstructor(
        "hamming_nearest",
        "Use recall(patterns, query). There is no training method.",
    )
