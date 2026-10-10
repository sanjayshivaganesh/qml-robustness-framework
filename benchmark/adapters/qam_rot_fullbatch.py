"""Rot full-batch QAM. The class stays in the noise-sweep module.

``build`` calls ``experiments.noise_robustness_sweep.QAM``. That constructor
reseeds the shared global stream to 42 and then draws parameters. This
adapter does not seed and does not draw. ``train`` and ``recall`` are the
methods on the constructed instance. ``recall`` returns
``(pattern, wall_time / n_bits)``.
"""

import inspect

from experiments.noise_robustness_sweep import QAM

from benchmark.identity import ProcedureIdentity, RngContract, procedure_description

implementation = QAM

identity = ProcedureIdentity(
    procedure_id="qam_rot_fullbatch",
    family="variational_qam",
    source="experiments.noise_robustness_sweep.QAM",
    capabilities=("build", "train", "recall"),
)

rng_contract = RngContract(
    stream="pennylane_numpy_global",
    construction_reseeds=True,
    construction_draws=True,
    reseed_value=42,
    notes=(
        "experiments.noise_robustness_sweep.QAM.__init__ calls "
        "pennylane.numpy.random.seed(42) before np.random.randn. "
        "This adapter does not seed."
    ),
)


def build(n_bits, **constructor_kwargs):
    """Call ``QAM(n_bits, **constructor_kwargs)`` with no extra seeding.

    ``QAM(n_bits)`` is the noise-sweep call. Class defaults are depth 6,
    300 Adam steps, and lr 0.05. The class then uses ``depth = min(6, n_bits)``.
    """
    return QAM(n_bits, **constructor_kwargs)


def train(model, patterns):
    """Return the elapsed seconds from ``QAM.train``."""
    return model.train(patterns)


def recall(model, x):
    """Return ``(pattern, wall_time / n_bits)`` from ``QAM.recall``."""
    return model.recall(x)


def describe(**hyperparameters):
    params = inspect.signature(QAM.__init__).parameters
    base = {
        "class_default_depth": params["depth"].default,
        "class_default_train_steps": params["train_steps"].default,
        "class_default_lr": params["lr"].default,
        "encoding": "RX(x * pi) and RY(x * pi / 2)",
        "ansatz": "Rot + linear CNOT",
        "loss": "mean over patterns of the sum of squared Pauli-Z errors",
        "recall_returns": "(pattern, wall_time / n_bits)",
        "protocol_call": "QAM(n_bits)",
    }
    base.update(hyperparameters)
    return procedure_description(identity, rng_contract, **base)
