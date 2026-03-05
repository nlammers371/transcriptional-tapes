import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from transcriptional_tapes.batch import simulate_dataset
from transcriptional_tapes.gillespie import simulate_trace
from transcriptional_tapes.model import PromoterModel, SimulationConfig
from transcriptional_tapes.ms2 import kernel_from_pattern, kernels_from_cassette_map


def _constant_emission_model(emission=1.0):
    return PromoterModel(
        rate_matrix=np.array([[-1.0, 1.0], [1.0, -1.0]]),
        emission_rates=np.array([emission, emission]),
        sigma=0.0,
        pi0=np.array([1.0, 0.0]),
    )


def test_custom_ms2_pattern_changes_trace_shape():
    model = _constant_emission_model()
    config = SimulationConfig(
        seq_length=6,
        delta_t=1.0,
        unit_size_bp=50,
        ms2_pattern=np.array([1, 1, 0, 0], dtype=int),
    )

    tr = simulate_trace(model, config, rng=np.random.default_rng(0))
    expected = np.array([0.0, 0.0, 1.0, 3.0, 3.0, 3.0])
    assert np.allclose(tr.fluo_ms2, expected)
    assert tr.fluo_ms2_channels.shape == (1, 6)


def test_direct_ms2_kernel_override_is_used():
    model = _constant_emission_model()
    config = SimulationConfig(seq_length=5, delta_t=1.0, ms2_kernel=np.array([1.0, 0.5]))

    tr = simulate_trace(model, config, rng=np.random.default_rng(0))
    expected = np.array([1.0, 1.5, 1.5, 1.5, 1.5])
    assert np.allclose(tr.fluo_ms2, expected)


def test_legacy_alpha_memory_path_still_works_without_custom_kernel():
    model = _constant_emission_model()
    config = SimulationConfig(seq_length=4, delta_t=1.0, memory_steps=4, alpha=1.0)
    tr = simulate_trace(model, config, rng=np.random.default_rng(0))
    assert np.allclose(tr.fluo_ms2, np.array([0.5, 1.5, 2.5, 3.5]))


def test_kernel_from_pattern_supports_multiple_blocks():
    pattern = np.array([0, 1, 1, 0, 1, 0], dtype=int)
    kernel = kernel_from_pattern(pattern)
    assert np.array_equal(kernel, np.array([0.0, 1.0, 1.0, 2.0, 3.0, 3.0]))


def test_kernels_from_cassette_map_for_three_channels():
    cassette = np.array([0, 1, 2, 2, 3, 1], dtype=int)
    kernels = kernels_from_cassette_map(cassette, n_channels=3)
    assert kernels.shape == (3, 6)
    assert np.array_equal(kernels[0], np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0]))
    assert np.array_equal(kernels[1], np.array([0.0, 0.0, 1.0, 2.0, 2.0, 2.0]))
    assert np.array_equal(kernels[2], np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0]))


def test_multi_channel_simulation_and_batch_output_shape():
    model = _constant_emission_model()
    config = SimulationConfig(
        seq_length=5,
        delta_t=1.0,
        n_channels=3,
        cassette_map=np.array([0, 1, 2, 3], dtype=int),
    )
    tr = simulate_trace(model, config, rng=np.random.default_rng(1))
    assert tr.fluo_ms2_channels.shape == (3, 5)

    ds = simulate_dataset(model, config, n_traces=4, seed=1)
    assert ds["observed_fluo"].shape == (5, 4)
    assert ds["observed_fluo_by_channel"].shape == (3, 5, 4)


def test_channel_noise_sigma_and_snr_are_supported():
    model = _constant_emission_model(emission=2.0)
    cfg_sigma = SimulationConfig(
        seq_length=40,
        delta_t=1.0,
        n_channels=2,
        cassette_map=np.array([1, 0, 2, 0, 1], dtype=int),
        channel_noise_sigma=np.array([0.1, 0.2]),
    )
    tr_sigma = simulate_trace(model, cfg_sigma, rng=np.random.default_rng(2))
    assert tr_sigma.fluo_ms2_channels.shape == (2, 40)

    cfg_snr = SimulationConfig(
        seq_length=40,
        delta_t=1.0,
        n_channels=2,
        cassette_map=np.array([1, 2, 0, 1, 2], dtype=int),
        channel_snr=np.array([5.0, 10.0]),
        snr_mode="peak",
    )
    tr_snr = simulate_trace(model, cfg_snr, rng=np.random.default_rng(3))
    assert tr_snr.fluo_ms2_channels.shape == (2, 40)


def test_invalid_cassette_map_values_raise():
    with pytest.raises(ValueError):
        kernels_from_cassette_map(np.array([0, 4], dtype=int), n_channels=3)


def test_multi_channel_requires_cassette_map():
    model = _constant_emission_model()
    config = SimulationConfig(seq_length=5, delta_t=1.0, n_channels=2, memory_steps=3, alpha=1.0)
    with pytest.raises(ValueError):
        simulate_trace(model, config, rng=np.random.default_rng(0))
