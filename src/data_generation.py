"""Random ±1 patterns and query corruption."""

import numpy as np


def random_bipolar_patterns(n_patterns, n_bits, rng=None):
    """Independent random ±1 patterns, shape ``(n_patterns, n_bits)``."""
    if rng is None:
        rng = np.random
    return rng.choice([-1, 1], size=(n_patterns, n_bits))


def corrupt_bernoulli(pattern, rate, rng=None):
    """Flip each bit independently with probability ``rate``."""
    if rng is None:
        rng = np.random
    corrupted = pattern.copy()
    mask = rng.rand(len(pattern)) < rate
    corrupted[mask] *= -1
    return corrupted


def corrupt_fixed_count(pattern, noise_level, rng=None):
    """Flip ``max(1, int(len(pattern) * noise_level))`` bits.

    Capacity and depth use this with ``p=0.15``. At ``p=0`` it would still
    flip one bit; that matches the original helper.
    """
    if rng is None:
        rng = np.random
    corrupted = pattern.copy()
    n_flip = max(1, int(len(pattern) * noise_level))
    flip_idx = rng.choice(len(pattern), size=n_flip, replace=False)
    corrupted[flip_idx] *= -1
    return corrupted
