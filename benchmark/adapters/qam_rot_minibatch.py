"""Rot minibatch QAM. The class stays in the scaling module.

The scaling experiment calls
``QAM(n_bits, depth=min(6, n_bits), steps=200, lr=0.05)``.
The class default ``depth=3`` is not that call. ``build`` applies the
experiment's depth expression when ``depth`` is omitted, then passes
``steps`` and ``lr`` through to the original constructor.

``train`` and ``recall`` are the methods on that instance. This adapter
does not cache a minibatch and does not remove the second ``loss`` evaluation
inside ``opt.step``.
"""

import inspect

from experiments.scaling import QAM

from benchmark.identity import ProcedureIdentity, RngContract, procedure_description

implementation = QAM

identity = ProcedureIdentity(
    procedure_id="qam_rot_minibatch",
    family="variational_qam",
    source="experiments.scaling.QAM",
    capabilities=("build", "train", "recall"),
)

rng_contract = RngContract(
    stream="pennylane_numpy_global",
    construction_reseeds=False,
    construction_draws=True,
    notes=(
        "QAM.__init__ draws pennylane.numpy.random.randn and does not reseed. "
        "Training draws a new minibatch on every loss evaluation, and "
        "opt.step evaluates loss again. This adapter does not seed and does "
        "not accept a generator."
    ),
)

# Literals from the scaling experiment's constructor call, not from QAM.__init__.
PROTOCOL_STEPS = 200
PROTOCOL_LR = 0.05


def build(n_bits, *, depth=None, steps=PROTOCOL_STEPS, lr=PROTOCOL_LR):
    """Construct the scaling ``QAM`` with the experiment's depth expression."""
    if depth is None:
        depth = min(6, n_bits)
    return QAM(n_bits, depth=depth, steps=steps, lr=lr)


def train(model, patterns):
    """Return the elapsed seconds from ``QAM.train``."""
    return model.train(patterns)


def recall(model, x):
    """Return the signed pattern from ``QAM.recall`` (no per-bit time)."""
    return model.recall(x)


def describe(**hyperparameters):
    class_defaults = inspect.signature(QAM.__init__).parameters
    base = {
        "protocol_depth": "min(6, n_bits)",
        "class_default_depth": class_defaults["depth"].default,
        "steps": PROTOCOL_STEPS,
        "lr": PROTOCOL_LR,
        "class_default_steps": class_defaults["steps"].default,
        "class_default_lr": class_defaults["lr"].default,
        "encoding": "RY(x * pi)",
        "ansatz": "Rot + linear CNOT",
        "minibatch": 2,
        "early_stop_patience": 15,
        "early_stop_delta": 1e-4,
        "recall_returns": "signed pattern",
        "accepts_rng_argument": False,
    }
    base.update(hyperparameters)
    return procedure_description(identity, rng_contract, **base)
