"""Bernoulli and fixed-count corruption, as two procedures.

Both call the helpers in ``src.data_generation``. ``rng=None`` keeps those
helpers on the global ``numpy.random`` stream. This module does not seed,
does not build a model, and does not compute a metric.

``experiments/scaling.py`` and ``experiments/noise_robustness_inference.py``
still inline ``rand(n) < p`` then ``*= -1``. Those copies stay in the
experiment modules so their position in the historical random stream does
not move. A benchmark runner may call ``apply_bernoulli`` only where that
call replaces the inline mask with the same ``rand`` draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data_generation import corrupt_bernoulli, corrupt_fixed_count

BERNOULLI = "bernoulli"
FIXED_COUNT = "fixed_count"

# Experiment modules that still own an inline Bernoulli mask.
INLINE_BERNOULLI_MODULES = (
    "experiments/scaling.py",
    "experiments/noise_robustness_inference.py",
)


class UnknownPerturbation(ValueError):
    """The procedure id is not Bernoulli or fixed-count."""

    def __init__(self, procedure_id):
        self.procedure_id = procedure_id
        super().__init__(
            f"unknown perturbation {procedure_id!r}; "
            f"expected {BERNOULLI!r} or {FIXED_COUNT!r}"
        )


@dataclass(frozen=True)
class CorruptionSpec:
    """Which corruption procedure to run. The level does not select it.

    ``level`` is the Bernoulli probability, or the ``p`` in
    ``max(1, int(n * p))`` for fixed-count. The same number is a different
    procedure under the other id.
    """

    procedure_id: str
    level: float


@dataclass(frozen=True)
class Query:
    """One corrupted vector and the procedure that produced it.

    ``n_flips`` is the number of positions that differ from ``target``.
    For Bernoulli that count is whatever the mask did. For fixed-count it
    is ``max(1, int(n * level))``, including when ``level`` is 0.
    """

    target: np.ndarray
    corrupted: np.ndarray
    procedure_id: str
    level: float
    n_flips: int


def apply_bernoulli(pattern, rate, rng=None):
    """Flip each bit independently with probability ``rate``.

    Rate 0 flips nothing. Delegates to ``corrupt_bernoulli``.
    """
    return corrupt_bernoulli(pattern, rate, rng=rng)


def apply_fixed_count(pattern, noise_level, rng=None):
    """Flip ``max(1, int(n * noise_level))`` distinct bits.

    ``noise_level`` 0 still flips one bit. Delegates to ``corrupt_fixed_count``.
    """
    return corrupt_fixed_count(pattern, noise_level, rng=rng)


def perturb(pattern, spec, rng=None):
    """Corrupt one bipolar vector. Does not train a model or score a metric."""
    if not isinstance(spec, CorruptionSpec):
        raise TypeError("perturb requires a CorruptionSpec; a numeric level is not a procedure")
    target = np.array(pattern, copy=True)
    if spec.procedure_id == BERNOULLI:
        corrupted = apply_bernoulli(target, spec.level, rng=rng)
    elif spec.procedure_id == FIXED_COUNT:
        corrupted = apply_fixed_count(target, spec.level, rng=rng)
    else:
        raise UnknownPerturbation(spec.procedure_id)
    n_flips = int(np.count_nonzero(np.asarray(corrupted) != target))
    return Query(
        target=target,
        corrupted=np.asarray(corrupted),
        procedure_id=spec.procedure_id,
        level=float(spec.level),
        n_flips=n_flips,
    )
