"""PauliX-encoding QAM. Functions stay in the statistical-significance module.

``recall`` is that module's ``qam_recall``, imported under an alias so it is
not ``src.qam.qam_recall``. Depth, epoch count, and learning rate stay the
module constants those functions already close over. This adapter does not
seed. ``train_qam`` draws ``numpy.random.normal(0, 0.1)``.
"""

from experiments.statistical_significance import DEPTH, EPOCHS, LR
from experiments.statistical_significance import build_qam
from experiments.statistical_significance import qam_recall as paulix_qam_recall
from experiments.statistical_significance import train_qam

from benchmark.identity import ProcedureIdentity, RngContract, procedure_description

build_implementation = build_qam
train_implementation = train_qam
recall_implementation = paulix_qam_recall

identity = ProcedureIdentity(
    procedure_id="qam_paulix",
    family="variational_qam",
    source="experiments.statistical_significance.qam_recall",
    capabilities=("build", "train", "recall"),
)

rng_contract = RngContract(
    stream="global_numpy",
    construction_reseeds=False,
    construction_draws=False,
    notes=(
        "build_qam does not draw. train_qam draws numpy.random.normal(0, 0.1) "
        "on the shared global stream. This adapter does not seed."
    ),
)


def build(n_bits):
    """Return the QNode from ``build_qam``."""
    return build_qam(n_bits)


def train(patterns, circuit):
    """Return trained parameters from ``train_qam``."""
    return train_qam(patterns, circuit)


def recall(params, circuit, x):
    """Return the signed pattern from the statistical-significance ``qam_recall``."""
    return paulix_qam_recall(params, circuit, x)


def describe(**hyperparameters):
    base = {
        "build": "experiments.statistical_significance.build_qam",
        "train": "experiments.statistical_significance.train_qam",
        "encoding": "PauliX where the bit is -1",
        "ansatz": "RY + linear CNOT",
        "depth": DEPTH,
        "epochs": EPOCHS,
        "lr": LR,
        "init": "normal(0, 0.1)",
        "recall_returns": "signed expectations, zeros mapped to +1",
        "distinct_from": "src.qam.qam_recall",
    }
    base.update(hyperparameters)
    return procedure_description(identity, rng_contract, **base)
