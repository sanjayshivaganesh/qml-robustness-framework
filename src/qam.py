"""RY-linear variational QAM used by capacity, depth, and convergence.

Other experiments use different circuits (or nearest neighbor) and keep
those implementations locally.

Per layer: RY-encode the input, then ``depth`` layers of per-qubit RY and
nearest-neighbor CNOTs. Readout is Pauli-Z on each qubit; recall applies
``sign``.
"""

import time
import numpy as onp
import pennylane as qml
from pennylane import numpy as np


def qam_ansatz(params, wires):
    """One RY + linear-CNOT layer. ``wires`` is unused except for length."""
    n_qubits = len(wires)
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])


def build_ry_linear_circuit(n_bits, depth, return_tuple=False):
    """Build the RY-linear QNode.

    ``return_tuple=True`` matches capacity/convergence; ``False`` matches
    the depth script. Both are valid PennyLane return styles.
    """
    dev = qml.device("default.qubit", wires=n_bits)

    if return_tuple:

        @qml.qnode(dev)
        def circuit(param, x):
            for i in range(n_bits):
                qml.RY(x[i] * np.pi, wires=i)
            for d in range(depth):
                qam_ansatz(param[d], wires=range(n_bits))
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_bits))

    else:

        @qml.qnode(dev)
        def circuit(param, x):
            for i in range(n_bits):
                qml.RY(x[i] * np.pi, wires=i)
            for d in range(depth):
                qam_ansatz(param[d], range(n_bits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_bits)]

    return circuit


def train_ry_linear_qam(
    patterns,
    n_bits,
    depth,
    steps=100,
    lr=0.05,
    return_tuple=True,
):
    """Adam-train the RY-linear QAM. Returns ``(params, circuit, train_time)``."""
    circuit = build_ry_linear_circuit(n_bits, depth, return_tuple=return_tuple)
    params = np.random.randn(depth, n_bits) * 0.01
    opt = qml.AdamOptimizer(lr)

    start_train = time.time()
    for _ in range(steps):

        def loss(p):
            loss_val = 0.0
            for pat in patterns:
                out = np.array(circuit(p, pat))
                target = np.array(pat)
                loss_val += np.mean((target - out) ** 2)
            return loss_val / len(patterns)

        params = opt.step(loss, params)
    train_time = time.time() - start_train
    return params, circuit, train_time


def train_ry_linear_qam_record_loss(patterns, n_bits, depth, steps=300, lr=0.05):
    """Same training as capacity, but records loss before each step and once after.

    Matches the original convergence script.
    """
    circuit = build_ry_linear_circuit(n_bits, depth, return_tuple=True)
    params = np.random.randn(depth, n_bits) * 0.01
    opt = qml.AdamOptimizer(lr)
    loss_history = []

    def loss_fn(p):
        loss_val = 0.0
        for pat in patterns:
            out = np.array(circuit(p, pat))
            target = np.array(pat)
            loss_val += np.mean((target - out) ** 2)
        return loss_val / len(patterns)

    for step in range(steps):
        current_loss = float(loss_fn(params))
        loss_history.append(current_loss)
        if step % 25 == 0:
            print(f"step={step:3d} | loss={current_loss:.6f}")
        params = opt.step(loss_fn, params)

    loss_history.append(float(loss_fn(params)))
    return params, circuit, loss_history


def qam_recall(x, circuit, params):
    """Map measured expectations to ±1 (zeros → +1)."""
    out = np.array(circuit(params, x))
    out = np.sign(out)
    out[out == 0] = 1
    return out


def nearest_neighbor_recall(patterns, query):
    """Hamming nearest neighbor. Used as the inference experiment's ``qam`` column."""
    distances = onp.sum(patterns != query, axis=1) / query.shape[0]
    best_idx = int(onp.argmin(distances))
    return patterns[best_idx]
