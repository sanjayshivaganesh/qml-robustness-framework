"""Hebbian Hopfield network.

Recall is not the same in every experiment:

- ``Hopfield``: synchronous ``sign(Wx)`` (noise sweep, capacity, stats).
- ``hopfield_async_sequential_recall``: sequential async updates + energy
  (inference experiment).
- ``HopfieldAsyncRandom``: random-order async updates using PennyLane's
  numpy RNG, so the scaling experiment keeps the same random stream as the
  original script (patterns and permutations shared that RNG).
"""

import time
import numpy as np


class Hopfield:
    """Synchronous discrete Hopfield network."""

    def __init__(self, n_bits, steps=10):
        self.n_bits = n_bits
        self.steps = steps
        self.W = np.zeros((n_bits, n_bits))

    def train(self, patterns):
        start_time = time.time()
        self.W = np.zeros((self.n_bits, self.n_bits))
        for p in patterns:
            self.W += np.outer(p, p)
        np.fill_diagonal(self.W, 0)
        self.W /= len(patterns)
        return time.time() - start_time

    def recall(self, pattern, steps=None):
        n_steps = self.steps if steps is None else steps
        x = np.asarray(pattern).copy()
        for _ in range(n_steps):
            x = np.sign(self.W @ x)
            x[x == 0] = 1
        return x

    def recall_timed(self, pattern, steps=None):
        """Recall and return ``(pattern, time / n_bits)``.

        Same timing convention as the original noise-robustness sweep.
        """
        start_time = time.time()
        x = self.recall(pattern, steps=steps)
        return x, (time.time() - start_time) / len(pattern)


def hopfield_hebbian_weights(patterns):
    """Hebbian W with zero diagonal, divided by the number of patterns."""
    n_patterns = len(patterns)
    W = (np.asarray(patterns).T @ np.asarray(patterns)) / n_patterns
    np.fill_diagonal(W, 0)
    return W


def hopfield_async_sequential_recall(W, noisy_input, steps=20):
    """Sequential async recall for the inference experiment.

    Returns ``(final_state, final_energy, energy_drop)``.
    """
    state = noisy_input.copy()
    input_energy = -0.5 * noisy_input @ W @ noisy_input
    n_bits = len(state)
    for _ in range(steps):
        for i in range(n_bits):
            s = np.sign(np.dot(W[i], state))
            state[i] = 1 if s == 0 else s
    final_energy = -0.5 * state @ W @ state
    energy_drop = input_energy - final_energy
    return state, final_energy, energy_drop


class HopfieldAsyncRandom:
    """Random-order async Hopfield (scaling experiment).

    Permutations use PennyLane numpy so they share the RNG with pattern
    generation, matching the original ``pennylane.numpy as np`` script.
    """

    def __init__(self, n_bits, steps=10):
        from pennylane import numpy as pnp

        self.n_bits = n_bits
        self.steps = steps
        self.W = pnp.zeros((n_bits, n_bits))

    def train(self, patterns):
        from pennylane import numpy as pnp

        self.W = pnp.zeros((self.n_bits, self.n_bits))
        for p in patterns:
            self.W += pnp.outer(p, p)
        pnp.fill_diagonal(self.W, 0)
        self.W /= len(patterns)

    def recall(self, x, steps=None):
        from pennylane import numpy as pnp

        n_steps = self.steps if steps is None else steps
        v = x.copy()
        for _ in range(n_steps):
            for i in pnp.random.permutation(self.n_bits):
                s = np.dot(self.W[i], v)
                v[i] = 1 if s >= 0 else -1
        return v
