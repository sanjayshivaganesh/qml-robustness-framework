"""Thin adapters over the associative-memory procedures already in this repo.

Importing this package imports those implementations. It does not call an
experiment ``main`` and it does not write result files.
"""

from benchmark.adapters import (
    hamming_nearest,
    hopfield_async_random,
    hopfield_async_sequential,
    hopfield_synchronous,
    qam_paulix,
    qam_rot_fullbatch,
    qam_rot_minibatch,
    qam_ry_linear,
)

ALL = (
    hopfield_synchronous,
    hopfield_async_sequential,
    hopfield_async_random,
    qam_ry_linear,
    qam_rot_fullbatch,
    qam_rot_minibatch,
    qam_paulix,
    hamming_nearest,
)

__all__ = [
    "ALL",
    "hamming_nearest",
    "hopfield_async_random",
    "hopfield_async_sequential",
    "hopfield_synchronous",
    "qam_paulix",
    "qam_rot_fullbatch",
    "qam_rot_minibatch",
    "qam_ry_linear",
]
