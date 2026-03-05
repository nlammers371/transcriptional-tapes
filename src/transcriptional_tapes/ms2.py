import numpy as np


def ms2_loading_coeff_integral(alpha: float, w: int, delta_t: float, t1: float, t2: float) -> float:
    t_ms2 = alpha * delta_t
    if t2 <= t_ms2:
        return (t2 * t2 - t1 * t1) / (2.0 * t_ms2) if t_ms2 > 0 else 0.0
    if t1 >= t_ms2:
        return t2 - t1
    return (t_ms2 * t_ms2 - t1 * t1) / (2.0 * t_ms2) + (t2 - t_ms2)


def kernel_from_pattern(ms2_pattern, normalize: bool = False) -> np.ndarray:
    """Build a discrete age-kernel from a 0/1 occupancy pattern along gene bins."""
    pattern = np.asarray(ms2_pattern, dtype=float)
    if pattern.ndim != 1 or pattern.size == 0:
        raise ValueError("ms2_pattern must be a non-empty 1D array")
    if np.any((pattern != 0) & (pattern != 1)):
        raise ValueError("ms2_pattern entries must be 0 or 1")

    kernel = np.cumsum(pattern[::-1])
    if normalize and kernel[-1] > 0:
        kernel = kernel / kernel[-1]
    return kernel


def kernels_from_cassette_map(cassette_map, n_channels: int, normalize: bool = False) -> np.ndarray:
    """Build per-channel kernels from a single gene-body cassette map.

    cassette_map values are integers in [0, n_channels], where 0 means no cassette
    and i means the bin contributes to channel i.
    """
    cassette = np.asarray(cassette_map)
    if cassette.ndim != 1 or cassette.size == 0:
        raise ValueError("cassette_map must be a non-empty 1D array")
    if not np.issubdtype(cassette.dtype, np.integer):
        raise ValueError("cassette_map entries must be integers")
    if n_channels < 1:
        raise ValueError("n_channels must be >= 1")
    if np.any(cassette < 0) or np.any(cassette > n_channels):
        raise ValueError("cassette_map entries must lie in [0, n_channels]")

    kernels = np.zeros((n_channels, cassette.size), dtype=float)
    for ch in range(1, n_channels + 1):
        kernels[ch - 1] = kernel_from_pattern((cassette == ch).astype(float), normalize=normalize)
    return kernels


def integrate_step_kernel(kernel: np.ndarray, delta_t: float, t1: float, t2: float) -> float:
    """Integrate a stepwise-constant kernel over [t1, t2]."""
    if t2 <= t1:
        return 0.0

    k = np.asarray(kernel, dtype=float)
    w = k.size
    if w == 0:
        return 0.0

    t_low = max(0.0, t1)
    t_high = min(t2, w * delta_t)
    if t_high <= t_low:
        return 0.0

    acc = 0.0
    left = t_low
    while left < t_high:
        i = min(int(left // delta_t), w - 1)
        right = min((i + 1) * delta_t, t_high)
        acc += k[i] * (right - left)
        left = right
    return acc
