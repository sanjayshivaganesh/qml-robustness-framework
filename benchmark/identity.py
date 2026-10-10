"""Shared adapter contract.

Every adapter exposes ``identity``, ``rng_contract``, and ``describe()``.
Those three members are the whole shared surface.

Construction, training, and recall stay on the adapter that has them.
``capabilities`` names the operations that adapter actually implements.
``call`` raises ``UnsupportedOperation`` for anything else. An adapter that
does not train does not grow a ``train`` method.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RngContract:
    """How this procedure uses the legacy global random stream.

    ``construction_reseeds`` is true only when the wrapped constructor itself
    calls a global seed function. ``construction_draws`` is true when that
    constructor draws from the stream without necessarily seeding it.
    ``numpy.random`` and ``pennylane.numpy.random`` are one shared stream.
    """

    stream: str
    construction_reseeds: bool
    construction_draws: bool
    reseed_value: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class ProcedureIdentity:
    """Stable name for one existing procedure. Not a CSV ``model`` label."""

    procedure_id: str
    family: str
    source: str
    capabilities: tuple[str, ...]


class UnsupportedOperation(LookupError):
    """The procedure has no such operation. Callers do not get a substitute."""

    def __init__(self, procedure_id: str, operation: str):
        self.procedure_id = procedure_id
        self.operation = operation
        super().__init__(f"{procedure_id} does not support {operation}")


def procedure_description(
    identity: ProcedureIdentity,
    rng_contract: RngContract,
    **hyperparameters: Any,
) -> dict[str, Any]:
    """JSON-serializable identity, RNG contract, and passthrough settings."""
    payload: dict[str, Any] = {
        "procedure_id": identity.procedure_id,
        "family": identity.family,
        "source": identity.source,
        "capabilities": list(identity.capabilities),
        "rng": {
            "stream": rng_contract.stream,
            "construction_reseeds": rng_contract.construction_reseeds,
            "construction_draws": rng_contract.construction_draws,
            "reseed_value": rng_contract.reseed_value,
            "notes": rng_contract.notes,
        },
        "hyperparameters": hyperparameters,
    }
    try:
        json.dumps(payload)
    except TypeError as exc:
        raise TypeError(
            "describe() values must be JSON-serializable"
        ) from exc
    return payload


def call(adapter: Any, operation: str, /, *args: Any, **kwargs: Any) -> Any:
    """Invoke one declared capability and return the source function's result."""
    if operation not in adapter.identity.capabilities:
        raise UnsupportedOperation(adapter.identity.procedure_id, operation)
    method = getattr(adapter, operation, None)
    if not callable(method):
        raise UnsupportedOperation(adapter.identity.procedure_id, operation)
    return method(*args, **kwargs)
