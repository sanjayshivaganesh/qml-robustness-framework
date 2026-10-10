"""Random-order asynchronous Hopfield for the scaling experiment.

Delegates to ``HopfieldAsyncRandom``. Permutations use PennyLane NumPy
during ``recall``. This adapter does not accept a generator and does not
reseed. The scaling module imports this class under the local name
``Hopfield``; this adapter does not reuse that alias.
"""

import inspect

from src.hopfield import HopfieldAsyncRandom

from benchmark.identity import ProcedureIdentity, RngContract, procedure_description

implementation = HopfieldAsyncRandom

identity = ProcedureIdentity(
    procedure_id="hopfield_async_random",
    family="hopfield",
    source="src.hopfield.HopfieldAsyncRandom",
    capabilities=("build", "train", "recall"),
)

rng_contract = RngContract(
    stream="pennylane_numpy_global",
    construction_reseeds=False,
    construction_draws=False,
    notes=(
        "Recall draws index permutations from pennylane.numpy.random. "
        "Construction and Hebbian training do not draw. "
        "No generator argument is accepted."
    ),
)


def build(n_bits, steps=None):
    """Return a real ``HopfieldAsyncRandom``. Omitted ``steps`` keeps the default."""
    if steps is None:
        return HopfieldAsyncRandom(n_bits)
    return HopfieldAsyncRandom(n_bits, steps=steps)


def train(model, patterns):
    """Return whatever ``HopfieldAsyncRandom.train`` returns (``None``)."""
    return model.train(patterns)


def recall(model, x, steps=None):
    """Return the state from ``HopfieldAsyncRandom.recall``. No ``rng`` argument."""
    return model.recall(x, steps=steps)


def describe(**hyperparameters):
    base = {
        "default_steps": inspect.signature(HopfieldAsyncRandom.__init__).parameters[
            "steps"
        ].default,
        "positive_or_zero_local_field": 1,
        "permutation_rng": "pennylane.numpy.random",
        "accepts_rng_argument": False,
    }
    base.update(hyperparameters)
    return procedure_description(identity, rng_contract, **base)
