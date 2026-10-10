"""Manifest RNG records name the calls that already exist. They do not reseed."""

import inspect

import experiments.noise_robustness_sweep as historical
from benchmark.adapters import qam_rot_fullbatch
from benchmark.protocols.noise_robustness_sweep import (
    NoiseSweepRequest,
    rng_metadata,
)


def _request(seeds):
    return NoiseSweepRequest(
        n_bits_list=(2,),
        corruption_rates=(0.0,),
        seeds=tuple(seeds),
        train_steps=1,
        grid="smoke",
    )


def test_metadata_records_configured_seeds_and_the_known_reset():
    record = rng_metadata(_request((3, 4)))
    assert record["exact_reproducibility_claimed"] is False
    assert record["default_rng_used"] is False
    assert record["configured_seeds"]["trial_seeds"] == [3, 4]
    assert record["seeds_applied"][0]["values"] == [3, 4]
    assert record["seeds_applied"][0]["calls"] == [
        "numpy.random.seed",
        "pennylane.numpy.random.seed",
    ]
    reset = record["internal_resets"][0]
    assert reset["procedure_id"] == "qam_rot_fullbatch"
    assert reset["value"] == 42
    assert reset["configured"] is False
    assert reset["value"] == qam_rot_fullbatch.rng_contract.reseed_value
    assert "seed(42)" in inspect.getsource(historical.QAM.__init__)
    assert record["unseeded_procedures"][0]["name"] == "gaussian_ci_population"
    assert any("RandomState" in item for item in record["state_not_captured"])


def test_the_reset_is_not_recorded_as_a_configured_trial_seed():
    record = rng_metadata(_request((0,)))
    assert 42 not in record["configured_seeds"]["trial_seeds"]
    assert record["seeds_applied"][0]["values"] == [0]
    assert record["internal_resets"][0]["value"] == 42
