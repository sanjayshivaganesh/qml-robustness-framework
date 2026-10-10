"""Pattern batches and the two corruption procedures.

Uses ``numpy.random.RandomState`` or a saved global seed. Does not call an
experiment ``main`` and does not write under ``results/``.
"""

import subprocess
import sys

import numpy as np
import pytest

from benchmark.data import (
    BERNOULLI,
    FIXED_COUNT,
    INLINE_BERNOULLI_MODULES,
    CorruptionSpec,
    UnknownPerturbation,
    apply_bernoulli,
    apply_fixed_count,
    draw_bipolar_patterns,
    draw_inference_target_index,
    pattern_batch,
    perturb,
)
from src.data_generation import (
    corrupt_bernoulli,
    corrupt_fixed_count,
    random_bipolar_patterns,
)
from src.paths import PROJECT_ROOT


def _states_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _inline_bernoulli(pattern, rate, rng):
    """The mask still written in scaling and inference. Not a second rule."""
    noisy = pattern.copy()
    flip_mask = rng.rand(len(pattern)) < rate
    noisy[flip_mask] *= -1
    return noisy


def test_drawn_batch_has_bipolar_shape_and_records_the_generator():
    batch = draw_bipolar_patterns(
        4,
        6,
        rng=np.random.RandomState(1),
        generator="numpy.random.choice",
        seed_applied=1,
    )
    assert batch.patterns.shape == (4, 6)
    assert batch.n_patterns == 4
    assert batch.n_bits == 6
    assert set(np.unique(batch.patterns).tolist()).issubset({-1, 1})
    assert batch.generator == "numpy.random.choice"
    assert batch.seed_applied == 1
    single = draw_bipolar_patterns(1, 1, rng=np.random.RandomState(2))
    assert single.patterns.shape == (1, 1)
    assert single.patterns[0, 0] in (-1, 1)


def test_pattern_batch_rejects_wrong_shape_and_values():
    with pytest.raises(ValueError):
        pattern_batch([1, -1, 1], generator="literal")
    with pytest.raises(ValueError):
        pattern_batch([[0, 1], [1, -1]], generator="literal")
    with pytest.raises(ValueError):
        pattern_batch([[1, -1], [1, 2]], generator="literal")


def test_draw_matches_choice_and_does_not_insert_extra_draws():
    via = random_bipolar_patterns(3, 5, rng=np.random.RandomState(6))
    direct = np.random.RandomState(6).choice([-1, 1], size=(3, 5))
    assert np.array_equal(via, direct)

    left = np.random.RandomState(6)
    right = np.random.RandomState(6)
    batch = draw_bipolar_patterns(3, 5, rng=left, seed_applied=99)
    other = random_bipolar_patterns(3, 5, rng=right)
    assert np.array_equal(batch.patterns, other)
    assert batch.seed_applied == 99
    assert left.randint(0, 7) == right.randint(0, 7)
    assert np.array_equal(left.rand(4), right.rand(4))


def test_seed_label_does_not_change_the_draw_or_reseed():
    labeled = draw_bipolar_patterns(
        2, 4, rng=np.random.RandomState(9), seed_applied=123
    )
    unlabeled = draw_bipolar_patterns(2, 4, rng=np.random.RandomState(9))
    assert np.array_equal(labeled.patterns, unlabeled.patterns)


def test_explicit_rng_does_not_consume_the_global_stream():
    before = np.random.get_state()
    draw_bipolar_patterns(3, 4, rng=np.random.RandomState(1))
    apply_bernoulli(np.array([1, -1, 1, -1]), 0.5, rng=np.random.RandomState(1))
    apply_fixed_count(np.array([1, -1, 1, -1]), 0.15, rng=np.random.RandomState(1))
    assert _states_equal(np.random.get_state(), before)


def test_global_seed_reproduces_the_recorded_pattern_then_bernoulli_draw():
    """Same legacy stream the baseline smoke check recorded. Restores state."""
    before = np.random.get_state()
    try:
        np.random.seed(0)
        batch = draw_bipolar_patterns(2, 4, seed_applied=0)
        noisy = apply_bernoulli(batch.patterns[0], 0.25)
        expected = np.array([[-1, 1, 1, -1], [1, 1, 1, 1]])
        assert np.array_equal(batch.patterns, expected)
        assert np.array_equal(noisy, expected[0])
        assert batch.generator == "numpy.random"

        fresh = np.random.RandomState(0)
        assert np.array_equal(
            random_bipolar_patterns(2, 4, rng=fresh),
            expected,
        )
        assert np.array_equal(corrupt_bernoulli(expected[0], 0.25, rng=fresh), expected[0])
    finally:
        np.random.set_state(before)


def test_helper_matches_pennylane_numpy_choice_on_the_shared_stream():
    """Scaling draws with PennyLane NumPy. The helper can take that object.

    The scaling module itself still calls ``choice`` directly.
    """
    from pennylane import numpy as pnp

    before = np.random.get_state()
    try:
        pnp.random.seed(5)
        via_helper = random_bipolar_patterns(2, 3, rng=pnp.random)
        after_helper = np.random.rand(4)

        pnp.random.seed(5)
        direct = pnp.random.choice([-1, 1], size=(2, 3))
        after_direct = np.random.rand(4)
        assert np.array_equal(np.asarray(via_helper), np.asarray(direct))
        assert np.array_equal(after_helper, after_direct)
    finally:
        np.random.set_state(before)


def test_default_rng_is_not_the_legacy_stream():
    legacy = np.random.RandomState(0).rand(5)
    modern = np.random.default_rng(0).random(5)
    assert not np.allclose(legacy, modern)
    assert not hasattr(np.random.default_rng(0), "rand")


def test_inference_target_index_is_a_separate_randint():
    patterns_rng = np.random.RandomState(4)
    index_rng = np.random.RandomState(4)
    batch = draw_bipolar_patterns(5, 3, rng=patterns_rng)
    index = draw_inference_target_index(batch.n_patterns, rng=patterns_rng)
    random_bipolar_patterns(5, 3, rng=index_rng)
    assert index == index_rng.randint(0, 5)
    assert 0 <= int(index) < 5


def test_bernoulli_edges_do_not_mutate_the_input():
    pattern = np.array([1, -1, 1, -1])
    original = pattern.copy()
    untouched = apply_bernoulli(pattern, 0.0, rng=np.random.RandomState(0))
    flipped = apply_bernoulli(pattern, 1.0, rng=np.random.RandomState(0))
    assert np.array_equal(pattern, original)
    assert np.array_equal(untouched, original)
    assert untouched is not pattern
    assert np.array_equal(flipped, -original)
    assert int(np.count_nonzero(untouched != original)) == 0
    assert int(np.count_nonzero(flipped != original)) == 4


def test_bernoulli_is_reproducible_and_consumes_one_mask_per_call():
    pattern = np.array([1, -1, 1, 1, -1, -1, 1])
    first = apply_bernoulli(pattern, 0.4, rng=np.random.RandomState(8))
    assert np.array_equal(first, corrupt_bernoulli(pattern, 0.4, rng=np.random.RandomState(8)))

    left = np.random.RandomState(8)
    right = np.random.RandomState(8)
    assert np.array_equal(
        apply_bernoulli(pattern, 0.4, rng=left),
        apply_bernoulli(pattern, 0.4, rng=right),
    )
    second_left = apply_bernoulli(pattern, 0.4, rng=left)
    second_right = apply_bernoulli(pattern, 0.4, rng=right)
    assert np.array_equal(second_left, second_right)


@pytest.mark.parametrize("rate", [0.0, 0.25, 1.0])
def test_inline_bernoulli_matches_the_helper_and_the_leftover_stream(rate):
    pattern = np.array([1, -1, 1, -1, 1, 1, -1, 1])
    left = np.random.RandomState(11)
    right = np.random.RandomState(11)
    assert np.array_equal(
        _inline_bernoulli(pattern, rate, left),
        apply_bernoulli(pattern, rate, right),
    )
    assert np.array_equal(left.rand(5), right.rand(5))


@pytest.mark.parametrize(
    ("n_bits", "level", "n_flip"),
    [
        (1, 0.0, 1),
        (4, 0.0, 1),
        (4, 0.15, 1),
        (7, 0.15, 1),
        (8, 0.15, 1),
        (10, 0.15, 1),
        (16, 0.15, 2),
        (10, 0.2, 2),
    ],
)
def test_fixed_count_flips_max_one_or_int_n_times_p(n_bits, level, n_flip):
    assert n_flip == max(1, int(n_bits * level))
    pattern = np.ones(n_bits, dtype=int)
    original = pattern.copy()
    corrupted = apply_fixed_count(pattern, level, rng=np.random.RandomState(3))
    assert np.array_equal(pattern, original)
    assert int(np.count_nonzero(corrupted != original)) == n_flip
    assert np.array_equal(
        corrupted,
        corrupt_fixed_count(original, level, rng=np.random.RandomState(3)),
    )
    again = apply_fixed_count(original, level, rng=np.random.RandomState(3))
    assert np.array_equal(corrupted, again)


def test_fixed_count_above_length_still_rejects_replacement():
    pattern = np.array([1, -1, 1, -1])
    with pytest.raises(ValueError):
        apply_fixed_count(pattern, 2.0, rng=np.random.RandomState(0))


def test_level_does_not_select_the_procedure():
    pattern = np.array([1, -1, 1, -1, 1, -1])
    bernoulli = perturb(pattern, CorruptionSpec(BERNOULLI, 0.0), rng=np.random.RandomState(0))
    fixed = perturb(pattern, CorruptionSpec(FIXED_COUNT, 0.0), rng=np.random.RandomState(0))
    assert bernoulli.procedure_id == BERNOULLI
    assert fixed.procedure_id == FIXED_COUNT
    assert bernoulli.n_flips == 0
    assert fixed.n_flips == 1
    assert np.array_equal(bernoulli.target, pattern)
    assert np.array_equal(pattern, np.array([1, -1, 1, -1, 1, -1]))
    assert bernoulli.corrupted.shape == pattern.shape
    assert fixed.corrupted.shape == pattern.shape


def test_same_level_at_015_is_still_two_procedures():
    pattern = np.ones(16, dtype=int)
    bernoulli = perturb(
        pattern, CorruptionSpec(BERNOULLI, 0.15), rng=np.random.RandomState(5)
    )
    fixed = perturb(
        pattern, CorruptionSpec(FIXED_COUNT, 0.15), rng=np.random.RandomState(5)
    )
    assert fixed.n_flips == max(1, int(16 * 0.15))
    assert bernoulli.procedure_id != fixed.procedure_id


def test_perturb_rejects_a_bare_level_and_an_unknown_procedure():
    pattern = np.array([1, -1, 1, -1])
    with pytest.raises(TypeError):
        perturb(pattern, 0.15, rng=np.random.RandomState(0))
    with pytest.raises(UnknownPerturbation, match="bit_flip"):
        perturb(pattern, CorruptionSpec("bit_flip", 0.15), rng=np.random.RandomState(0))


def test_inline_bernoulli_modules_are_still_local():
    for relative in INLINE_BERNOULLI_MODULES:
        text = (PROJECT_ROOT / relative).read_text()
        assert "corrupt_bernoulli" not in text
        assert "np.random.rand(" in text
        assert "*= -1" in text


def test_data_package_does_not_seed_or_call_a_model():
    root = PROJECT_ROOT / "benchmark" / "data"
    for path in root.rglob("*.py"):
        text = path.read_text()
        assert "seed(" not in text
        assert "Hopfield" not in text
        assert "qml." not in text
        assert "results_subdir" not in text


def test_importing_data_does_not_touch_archived_results():
    script = r"""
from pathlib import Path
root = Path("results")

def snapshot():
    if not root.exists():
        return {}
    return {
        p.relative_to(root).as_posix(): p.stat().st_mtime_ns
        for p in root.rglob("*")
        if p.is_file()
    }

before = snapshot()
import benchmark.data
after = snapshot()
assert before == after
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
