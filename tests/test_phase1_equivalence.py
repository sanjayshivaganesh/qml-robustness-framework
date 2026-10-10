"""Small-input comparisons of framework wrappers with the original functions.

Global-stream tests save and restore NumPy state. A matching integer seed
is not treated as proof that two generator families match. No experiment
``main`` is called and nothing is written under ``results/``.
"""

import numpy as np
import pytest
from pennylane import numpy as pnp

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
from experiments.noise_robustness_sweep import QAM as RotFullBatchQAM
from experiments.scaling import QAM as RotMinibatchQAM
from experiments.statistical_significance import build_qam, train_qam
from experiments.statistical_significance import qam_recall as paulix_recall
from src.hopfield import Hopfield, HopfieldAsyncRandom, hopfield_async_sequential_recall
from src.qam import (
    nearest_neighbor_recall,
    qam_recall,
    train_ry_linear_qam,
    train_ry_linear_qam_record_loss,
)


def _states_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_synchronous_recall_matches_the_class_on_hand_patterns():
    stored = np.array([[1.0, -1.0, 1.0, -1.0]])
    via = hopfield_synchronous.build(4)
    direct = Hopfield(4)
    hopfield_synchronous.train(via, stored)
    direct.train(stored)
    assert np.array_equal(via.W, direct.W)
    assert np.array_equal(hopfield_synchronous.recall(via, stored[0]), stored[0])
    assert np.array_equal(
        hopfield_synchronous.recall(via, stored[0]),
        direct.recall(stored[0]),
    )

    query = np.array([-1.0, -1.0, 1.0])
    blank = hopfield_synchronous.build(3)
    assert np.all(blank.W == 0.0)
    recalled = hopfield_synchronous.recall(blank, query)
    assert np.array_equal(recalled, np.array([1.0, 1.0, 1.0]))
    assert np.array_equal(recalled, Hopfield(3).recall(query))


def test_sequential_recall_matches_energy_and_index_order():
    weights = np.array([[0.0, 1.0], [0.0, 0.0]])
    query = np.array([-1.0, -1.0])
    via = hopfield_async_sequential.recall(weights, query.copy(), steps=1)
    direct = hopfield_async_sequential_recall(weights, query.copy(), steps=1)
    assert np.array_equal(via[0], direct[0])
    assert np.array_equal(via[0], np.array([-1.0, 1.0]))

    input_energy = -0.5 * query @ weights @ query
    final_energy = -0.5 * via[0] @ weights @ via[0]
    assert via[1] == pytest.approx(final_energy)
    assert via[2] == pytest.approx(input_energy - final_energy)
    assert direct[1] == pytest.approx(via[1])
    assert direct[2] == pytest.approx(via[2])

    zeros = np.zeros((2, 2))
    noisy = np.array([-1.0, 1.0])
    assert np.array_equal(
        hopfield_async_sequential.recall(zeros, noisy.copy(), steps=1)[0],
        np.array([1.0, 1.0]),
    )
    default_via = hopfield_async_sequential.recall(zeros, noisy.copy())
    default_direct = hopfield_async_sequential_recall(zeros, noisy.copy())
    assert np.array_equal(default_via[0], default_direct[0])
    assert np.array_equal(default_via[0], np.array([1.0, 1.0]))


def test_random_order_recall_matches_under_one_pinned_seed():
    patterns = np.array([[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]])
    query = np.array([1.0, 1.0, -1.0, -1.0])
    before = np.random.get_state()
    try:
        np.random.seed(7)
        pnp.random.seed(7)
        via_model = hopfield_async_random.build(4)
        hopfield_async_random.train(via_model, patterns)
        via = hopfield_async_random.recall(via_model, query.copy(), steps=1)

        np.random.seed(7)
        pnp.random.seed(7)
        direct_model = HopfieldAsyncRandom(4)
        direct_model.train(patterns)
        direct = direct_model.recall(query.copy(), steps=1)
        assert np.array_equal(via, direct)

        np.random.seed(7)
        pnp.random.seed(7)
        permutation = np.asarray(pnp.random.permutation(4))
        after_permutation = np.random.get_state()

        np.random.seed(7)
        pnp.random.seed(7)
        numpy_permutation = np.asarray(np.random.permutation(4))
        assert np.array_equal(permutation, numpy_permutation)
        assert not np.array_equal(permutation, np.random.default_rng(7).permutation(4))

        np.random.seed(7)
        pnp.random.seed(7)
        probed = HopfieldAsyncRandom(4)
        probed.train(patterns)
        probed.recall(query.copy(), steps=1)
        assert _states_equal(np.random.get_state(), after_permutation)
        assert permutation.tolist() == [2, 1, 0, 3]
    finally:
        np.random.set_state(before)


def test_hamming_adapter_matches_nearest_neighbor_recall():
    patterns = np.array([[1, -1, 1, -1], [-1, -1, -1, 1]])
    query = np.array([1, -1, -1, -1])
    via = hamming_nearest.recall(patterns, query)
    direct = nearest_neighbor_recall(patterns, query)
    assert np.array_equal(via, direct)
    assert np.shares_memory(via, patterns)
    assert hamming_nearest.identity.procedure_id == "hamming_nearest"


def test_sign_maps_send_a_controlled_zero_to_plus_one():
    def circuit(_params, _x):
        return np.array([0.0, 0.2, -0.3])

    expected = np.array([1.0, 1.0, -1.0])
    assert np.array_equal(qam_recall(np.array([1, -1, 1]), circuit, None), expected)
    assert np.array_equal(
        qam_ry_linear.recall(np.array([1, -1, 1]), circuit, None),
        expected,
    )
    assert np.array_equal(
        paulix_recall(None, circuit, np.array([1, -1, 1])),
        expected,
    )
    assert np.array_equal(
        qam_paulix.recall(None, circuit, np.array([1, -1, 1])),
        expected,
    )


def test_rot_fullbatch_init_follows_reseed_42_not_the_preceding_stream():
    before = np.random.get_state()
    try:
        np.random.seed(0)
        pnp.random.seed(0)
        np.random.randn(2, 2, 3)
        continued = 0.01 * np.random.randn(2, 2, 3)

        pnp.random.seed(42)
        reset = 0.01 * np.random.randn(2, 2, 3)
        assert not np.allclose(continued, reset)

        np.random.seed(0)
        pnp.random.seed(0)
        np.random.randn(2, 2, 3)
        via = qam_rot_fullbatch.build(2, train_steps=1)
        assert np.allclose(np.asarray(via.params, dtype=float), reset)

        np.random.seed(0)
        pnp.random.seed(0)
        np.random.randn(2, 2, 3)
        direct = RotFullBatchQAM(2, train_steps=1)
        query = np.array([1.0, -1.0])
        via_pattern, _via_time = qam_rot_fullbatch.recall(via, query)
        direct_pattern, _direct_time = direct.recall(query)
        assert np.allclose(np.asarray(via.params, dtype=float), np.asarray(direct.params, dtype=float))
        assert np.array_equal(via_pattern, direct_pattern)
        assert qam_rot_fullbatch.rng_contract.reseed_value == 42
    finally:
        np.random.set_state(before)


def test_rot_minibatch_init_continues_the_stream_and_matches_the_class():
    before = np.random.get_state()
    try:
        np.random.seed(3)
        pnp.random.seed(3)
        untouched = qam_rot_minibatch.build(2, depth=1, steps=1, lr=0.05)

        np.random.seed(3)
        pnp.random.seed(3)
        np.random.rand()
        shifted = qam_rot_minibatch.build(2, depth=1, steps=1, lr=0.05)
        assert not np.allclose(
            np.asarray(untouched.params, dtype=float),
            np.asarray(shifted.params, dtype=float),
        )

        np.random.seed(3)
        pnp.random.seed(3)
        np.random.rand()
        direct = RotMinibatchQAM(2, depth=1, steps=1, lr=0.05)
        assert np.allclose(
            np.asarray(shifted.params, dtype=float),
            np.asarray(direct.params, dtype=float),
        )
        assert shifted.depth == direct.depth == 1
        assert shifted.steps == 1
        assert qam_rot_minibatch.rng_contract.construction_reseeds is False
    finally:
        np.random.set_state(before)


def test_ry_linear_two_step_train_matches_the_function_and_recorded_recall():
    patterns = np.array([[1.0, 1.0]])
    before = np.random.get_state()
    try:
        np.random.seed(1)
        pnp.random.seed(1)
        via_params, via_circuit, _via_time = qam_ry_linear.train(
            patterns, 2, 1, steps=2, lr=0.05, return_tuple=True
        )
        np.random.seed(1)
        pnp.random.seed(1)
        direct_params, direct_circuit, _direct_time = train_ry_linear_qam(
            patterns, 2, 1, steps=2, lr=0.05, return_tuple=True
        )
        assert np.allclose(np.asarray(via_params, dtype=float), np.asarray(direct_params, dtype=float))
        via_recall = qam_ry_linear.recall(patterns[0], via_circuit, via_params)
        direct_recall = qam_recall(patterns[0], direct_circuit, direct_params)
        assert np.array_equal(via_recall, direct_recall)
        assert np.array_equal(via_recall, np.array([-1.0, 1.0]))

        np.random.seed(1)
        pnp.random.seed(1)
        _params, _circuit, via_history = qam_ry_linear.train_recording_loss(
            patterns, 2, 1, steps=1, lr=0.05
        )
        np.random.seed(1)
        pnp.random.seed(1)
        _params, _circuit, direct_history = train_ry_linear_qam_record_loss(
            patterns, 2, 1, steps=1, lr=0.05
        )
        assert len(via_history) == len(direct_history) == 2
        assert via_history == pytest.approx(direct_history)
    finally:
        np.random.set_state(before)


def test_paulix_train_and_recall_match_on_a_two_bit_pattern():
    patterns = np.array([[1.0, -1.0]])
    circuit = build_qam(2)
    before = np.random.get_state()
    try:
        np.random.seed(1)
        pnp.random.seed(1)
        via = qam_paulix.train(patterns, circuit)
        np.random.seed(1)
        pnp.random.seed(1)
        direct = train_qam(patterns, circuit)
        assert np.allclose(np.asarray(via, dtype=float), np.asarray(direct, dtype=float))
        query = np.array([1.0, -1.0])
        assert np.array_equal(
            qam_paulix.recall(via, circuit, query),
            paulix_recall(direct, circuit, query),
        )
        assert np.shape(via) == (10, 2)
    finally:
        np.random.set_state(before)
