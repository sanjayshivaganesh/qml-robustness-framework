"""Pattern batches for the associative-memory experiments.

Draws go through ``src.data_generation.random_bipolar_patterns``. This module
does not seed and it does not corrupt. Historical experiment loops still call
``choice`` themselves; nothing here retargets those call sites.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data_generation import random_bipolar_patterns

GENERATOR_NUMPY_GLOBAL = "numpy.random"
GENERATOR_PENNYLANE_NUMPY = "pennylane.numpy.random"


@dataclass(frozen=True)
class PatternBatch:
    """Patterns one trial stores. Shape is ``(n_patterns, n_bits)``, values ±1.

    ``seed_applied`` records a seed the caller already used. Drawing does not
    apply it. ``generator`` names the random stream the caller drew from.
    """

    patterns: np.ndarray
    n_bits: int
    n_patterns: int
    generator: str
    seed_applied: int | None = None

    def __post_init__(self):
        array = np.asarray(self.patterns)
        if array.ndim != 2:
            raise ValueError(
                f"patterns must have shape (n_patterns, n_bits), got ndim {array.ndim}"
            )
        n_patterns, n_bits = array.shape
        if n_patterns != self.n_patterns or n_bits != self.n_bits:
            raise ValueError(
                f"shape {array.shape} does not match "
                f"n_patterns={self.n_patterns}, n_bits={self.n_bits}"
            )
        if n_patterns < 1 or n_bits < 1:
            raise ValueError("n_patterns and n_bits must be at least 1")
        if not np.isin(array, (-1, 1)).all():
            raise ValueError("patterns must contain only -1 and 1")
        if not isinstance(self.generator, str) or not self.generator:
            raise ValueError("generator must name the random stream that drew the batch")
        object.__setattr__(self, "patterns", array)


def pattern_batch(patterns, *, generator, seed_applied=None):
    """Wrap an already drawn ±1 array. Does not draw and does not seed."""
    array = np.asarray(patterns)
    if array.ndim != 2:
        raise ValueError(
            f"patterns must have shape (n_patterns, n_bits), got ndim {array.ndim}"
        )
    n_patterns, n_bits = array.shape
    return PatternBatch(
        patterns=array,
        n_bits=int(n_bits),
        n_patterns=int(n_patterns),
        generator=generator,
        seed_applied=seed_applied,
    )


def draw_bipolar_patterns(
    n_patterns,
    n_bits,
    rng=None,
    *,
    generator=None,
    seed_applied=None,
):
    """Draw independent ±1 patterns via ``random_bipolar_patterns``.

    ``rng=None`` uses the global ``numpy.random`` stream, which is what the
    helper does. This function does not call ``seed``. ``default_rng`` is a
    different generator family: the same seed integer does not reproduce
    the legacy ``RandomState`` sequence, and Bernoulli corruption calls
    ``rng.rand``, which that family does not provide.
    """
    if generator is None:
        generator = GENERATOR_NUMPY_GLOBAL if rng is None else "caller_rng"
    drawn = random_bipolar_patterns(n_patterns, n_bits, rng=rng)
    return pattern_batch(drawn, generator=generator, seed_applied=seed_applied)


def draw_inference_target_index(n_patterns, rng=None):
    """Return one ``randint(0, n_patterns)`` index.

    The inference trial draws this after the pattern matrix and before the
    Bernoulli mask. Other protocols do not. This function does not copy a
    row and does not corrupt one.
    """
    if rng is None:
        rng = np.random
    return rng.randint(0, n_patterns)
