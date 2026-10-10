"""Dataset draws and the two corruption procedures.

Importing this package does not seed, does not run an experiment, and does
not write under ``results/``.
"""

from benchmark.data.corruption import (
    BERNOULLI,
    FIXED_COUNT,
    INLINE_BERNOULLI_MODULES,
    CorruptionSpec,
    Query,
    UnknownPerturbation,
    apply_bernoulli,
    apply_fixed_count,
    perturb,
)
from benchmark.data.patterns import (
    GENERATOR_NUMPY_GLOBAL,
    GENERATOR_PENNYLANE_NUMPY,
    PatternBatch,
    draw_bipolar_patterns,
    draw_inference_target_index,
    pattern_batch,
)

__all__ = [
    "BERNOULLI",
    "FIXED_COUNT",
    "GENERATOR_NUMPY_GLOBAL",
    "GENERATOR_PENNYLANE_NUMPY",
    "INLINE_BERNOULLI_MODULES",
    "CorruptionSpec",
    "PatternBatch",
    "Query",
    "UnknownPerturbation",
    "apply_bernoulli",
    "apply_fixed_count",
    "draw_bipolar_patterns",
    "draw_inference_target_index",
    "pattern_batch",
    "perturb",
]
