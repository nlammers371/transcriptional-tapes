import numpy as np


def ms2_loading_coeff_integral(alpha: float, w: int, delta_t: float, t1: float, t2: float) -> float:
    t_ms2 = alpha * delta_t
    if t2 <= t_ms2:
        return (t2 * t2 - t1 * t1) / (2.0 * t_ms2) if t_ms2 > 0 else 0.0
    if t1 >= t_ms2:
        return t2 - t1
    return (t_ms2 * t_ms2 - t1 * t1) / (2.0 * t_ms2) + (t2 - t_ms2)


def kernel_from_pattern(ms2_pattern, normalize: bool = False) -> np.ndarray:
    """Build a discrete age-kernel from a 0/1 MS2 occupancy pattern along gene bins.

    The i-th kernel bin represents the expected cumulative MS2 contribution during
    age interval [i*delta_t, (i+1)*delta_t), assuming one gene bin is traversed
    per acquisition interval.
    """
    pattern = np.asarray(ms2_pattern, dtype=float)
    if pattern.ndim != 1 or pattern.size == 0:
        raise ValueError("ms2_pattern must be a non-empty 1D array")
    if np.any((pattern != 0) & (pattern != 1)):
        raise ValueError("ms2_pattern entries must be 0 or 1")

    # Age kernel in reverse transcription order: newest ages correspond to 3' bins.
    kernel = np.cumsum(pattern[::-1])
    if normalize and kernel[-1] > 0:
        kernel = kernel / kernel[-1]
    return kernel


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
