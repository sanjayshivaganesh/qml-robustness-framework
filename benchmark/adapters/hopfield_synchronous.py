"""Synchronous Hopfield. Delegates to ``src.hopfield.Hopfield``.

Used by the noise sweep, storage capacity, and statistical significance.
Default recall is 10 synchronous ``sign(Wx)`` steps. Zeros become ``+1``.
``recall_timed`` returns ``(pattern, wall_time / n_bits)``.
"""

import inspect

from src.hopfield import Hopfield

from benchmark.identity import ProcedureIdentity, RngContract, procedure_description

implementation = Hopfield

identity = ProcedureIdentity(
    procedure_id="hopfield_synchronous",
    family="hopfield",
    source="src.hopfield.Hopfield",
    capabilities=("build", "train", "recall", "recall_timed"),
)

rng_contract = RngContract(
    stream="none",
    construction_reseeds=False,
    construction_draws=False,
    notes=(
        "Training and synchronous recall are deterministic given the "
        "patterns and the query. This adapter does not seed."
    ),
)


def build(n_bits, steps=None):
    """Return a real ``Hopfield``. Omitted ``steps`` keeps the class default."""
    if steps is None:
        return Hopfield(n_bits)
    return Hopfield(n_bits, steps=steps)


def train(model, patterns):
    """Return whatever ``Hopfield.train`` returns (elapsed seconds)."""
    return model.train(patterns)


def recall(model, pattern, steps=None):
    """Return the recalled pattern from ``Hopfield.recall``."""
    return model.recall(pattern, steps=steps)


def recall_timed(model, pattern, steps=None):
    """Return ``(pattern, wall_time / n_bits)`` from ``Hopfield.recall_timed``."""
    return model.recall_timed(pattern, steps=steps)


def describe(**hyperparameters):
    base = {
        "default_steps": inspect.signature(Hopfield.__init__).parameters["steps"].default,
        "zero_local_field": 1,
        "recall_timed_divisor": "n_bits",
    }
    base.update(hyperparameters)
    return procedure_description(identity, rng_contract, **base)
