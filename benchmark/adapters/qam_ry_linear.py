"""RY-linear variational QAM from ``src.qam``.

Capacity calls ``train`` with ``return_tuple=True``. Convergence calls
``train_recording_loss``. Both then use the circuit those functions return.

Circuit-depth recall is not wrapped here. ``experiments/circuit_depth.py``
signs the circuit output inline. ``recall`` on this adapter is
``src.qam.qam_recall``, which is what storage capacity calls. Routing depth
through ``qam_recall`` waits on a comparison that is not part of this layer.
"""

from src.qam import (
    build_ry_linear_circuit,
    qam_recall,
    train_ry_linear_qam,
    train_ry_linear_qam_record_loss,
)

from benchmark.identity import ProcedureIdentity, RngContract, procedure_description

train_implementation = train_ry_linear_qam
train_recording_loss_implementation = train_ry_linear_qam_record_loss
recall_implementation = qam_recall
build_circuit_implementation = build_ry_linear_circuit

identity = ProcedureIdentity(
    procedure_id="qam_ry_linear",
    family="variational_qam",
    source="src.qam.train_ry_linear_qam",
    capabilities=("build_circuit", "train", "train_recording_loss", "recall"),
)

rng_contract = RngContract(
    stream="pennylane_numpy_global",
    construction_reseeds=False,
    construction_draws=False,
    notes=(
        "Parameter init is pennylane.numpy.random.randn inside the trainer. "
        "This adapter does not seed. The caller owns the stream position."
    ),
)


def build_circuit(n_bits, depth, return_tuple=False):
    """Passthrough to ``build_ry_linear_circuit``. Default ``return_tuple`` is False."""
    return build_ry_linear_circuit(n_bits, depth, return_tuple=return_tuple)


def train(patterns, n_bits, depth, steps=100, lr=0.05, return_tuple=True):
    """Passthrough to ``train_ry_linear_qam``. Returns ``(params, circuit, train_time)``."""
    return train_ry_linear_qam(
        patterns,
        n_bits,
        depth,
        steps=steps,
        lr=lr,
        return_tuple=return_tuple,
    )


def train_recording_loss(patterns, n_bits, depth, steps=300, lr=0.05):
    """Passthrough to ``train_ry_linear_qam_record_loss``.

    Returns ``(params, circuit, loss_history)``. Does not recall.
    """
    return train_ry_linear_qam_record_loss(
        patterns,
        n_bits,
        depth,
        steps=steps,
        lr=lr,
    )


def recall(x, circuit, params):
    """Passthrough to ``src.qam.qam_recall``. Not the circuit-depth inline sign map."""
    return qam_recall(x, circuit, params)


def describe(**hyperparameters):
    base = {
        "record_loss": "src.qam.train_ry_linear_qam_record_loss",
        "recall": "src.qam.qam_recall",
        "circuit": "src.qam.build_ry_linear_circuit",
        "encoding": "RY(x * pi)",
        "ansatz": "RY + linear CNOT",
        "train_returns": "(params, circuit, train_time)",
        "train_recording_loss_returns": "(params, circuit, loss_history)",
        "recall_returns": "signed expectations, zeros mapped to +1",
        "circuit_depth_recall": "not_wrapped",
        "circuit_depth_recall_note": (
            "experiments.circuit_depth signs the circuit output inline. "
            "This adapter does not replace that block."
        ),
    }
    base.update(hyperparameters)
    return procedure_description(identity, rng_contract, **base)
