"""Sequential asynchronous Hopfield for the inference experiment.

``store`` calls ``hopfield_hebbian_weights``. ``recall`` calls
``hopfield_async_sequential_recall`` and returns
``(state, final_energy, energy_drop)``. There is no ``train`` method:
this is not the synchronous ``Hopfield`` class.
"""

import inspect

from src.hopfield import hopfield_async_sequential_recall, hopfield_hebbian_weights

from benchmark.identity import ProcedureIdentity, RngContract, procedure_description

store_implementation = hopfield_hebbian_weights
recall_implementation = hopfield_async_sequential_recall

identity = ProcedureIdentity(
    procedure_id="hopfield_async_sequential",
    family="hopfield",
    source="src.hopfield.hopfield_async_sequential_recall",
    capabilities=("store", "recall"),
)

rng_contract = RngContract(
    stream="none",
    construction_reseeds=False,
    construction_draws=False,
    notes=(
        "Index-order updates. No random permutation. "
        "This adapter does not seed."
    ),
)


def store(patterns):
    """Return the Hebbian weight matrix from ``hopfield_hebbian_weights``."""
    return hopfield_hebbian_weights(patterns)


def recall(W, noisy_input, steps=None):
    """Return ``(state, final_energy, energy_drop)``.

    Omitted ``steps`` keeps the source default (20).
    """
    if steps is None:
        return hopfield_async_sequential_recall(W, noisy_input)
    return hopfield_async_sequential_recall(W, noisy_input, steps=steps)


def describe(**hyperparameters):
    default_steps = inspect.signature(hopfield_async_sequential_recall).parameters[
        "steps"
    ].default
    base = {
        "weights": "src.hopfield.hopfield_hebbian_weights",
        "default_steps": default_steps,
        "energy": "E = -0.5 * x^T W x",
        "zero_local_field": 1,
        "recall_returns": "(state, final_energy, energy_drop)",
    }
    base.update(hyperparameters)
    return procedure_description(identity, rng_contract, **base)
