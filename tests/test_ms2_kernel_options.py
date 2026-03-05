import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from transcriptional_tapes.gillespie import simulate_trace
from transcriptional_tapes.model import PromoterModel, SimulationConfig
from transcriptional_tapes.ms2 import kernel_from_pattern


def _constant_emission_model(emission=1.0):
    # Two-state model with identical emission so trajectories are deterministic in expectation.
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

    # For constant emission=1 and no noise, pattern [1,1,0,0], kernel bins are [0,0,1,2].
    # Integrated value over each 1-second sample should be [0,0,1,3,3,3].
    expected = np.array([0.0, 0.0, 1.0, 3.0, 3.0, 3.0])
    assert np.allclose(tr.fluo_ms2, expected)


def test_direct_ms2_kernel_override_is_used():
    model = _constant_emission_model()
    config = SimulationConfig(
        seq_length=5,
        delta_t=1.0,
        ms2_kernel=np.array([1.0, 0.5]),
    )

    tr = simulate_trace(model, config, rng=np.random.default_rng(0))
    expected = np.array([1.0, 1.5, 1.5, 1.5, 1.5])
    assert np.allclose(tr.fluo_ms2, expected)


def test_legacy_alpha_memory_path_still_works_without_custom_kernel():
    model = _constant_emission_model()
    config = SimulationConfig(seq_length=4, delta_t=1.0, memory_steps=4, alpha=1.0)
    tr = simulate_trace(model, config, rng=np.random.default_rng(0))

    # For constant emission 1 and alpha=1 with dt=1,
    # first sample ramps to 0.5 then accumulates full windows.
    assert np.allclose(tr.fluo_ms2, np.array([0.5, 1.5, 2.5, 3.5]))


def test_kernel_from_pattern_supports_multiple_blocks():
    pattern = np.array([0, 1, 1, 0, 1, 0], dtype=int)
    kernel = kernel_from_pattern(pattern)
    assert np.array_equal(kernel, np.array([0.0, 1.0, 1.0, 2.0, 3.0, 3.0]))
