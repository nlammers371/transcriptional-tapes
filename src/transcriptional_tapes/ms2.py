def ms2_loading_coeff_integral(alpha: float, w: int, delta_t: float, t1: float, t2: float) -> float:
    t_ms2 = alpha * delta_t
    if t2 <= t_ms2:
        return (t2 * t2 - t1 * t1) / (2.0 * t_ms2) if t_ms2 > 0 else 0.0
    if t1 >= t_ms2:
        return t2 - t1
    return (t_ms2 * t_ms2 - t1 * t1) / (2.0 * t_ms2) + (t2 - t_ms2)
