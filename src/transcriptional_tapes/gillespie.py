from dataclasses import dataclass
import numpy as np

from .model import PromoterModel, SimulationConfig
from .ms2 import (
    integrate_step_kernel,
    kernel_from_pattern,
    kernels_from_cassette_map,
    ms2_loading_coeff_integral,
)


@dataclass(slots=True)
class TraceResult:
    fluo: np.ndarray
    fluo_ms2: np.ndarray
    transition_times: np.ndarray
    naive_states: np.ndarray
    fluo_ms2_channels: np.ndarray | None = None


def _resolve_channel_sigmas(config: SimulationConfig, clean_signal: np.ndarray, default_sigma: float) -> np.ndarray:
    n_channels = clean_signal.shape[0]

    if config.channel_noise_sigma is not None:
        sigmas = np.asarray(config.channel_noise_sigma, dtype=float)
        if sigmas.shape != (n_channels,):
            raise ValueError("channel_noise_sigma must have shape (n_channels,)")
        if np.any(sigmas < 0):
            raise ValueError("channel_noise_sigma entries must be non-negative")
        return sigmas

    if config.channel_snr is not None:
        snr = np.asarray(config.channel_snr, dtype=float)
        if snr.shape != (n_channels,):
            raise ValueError("channel_snr must have shape (n_channels,)")
        if np.any(snr <= 0):
            raise ValueError("channel_snr entries must be positive")

        if config.snr_mode == "std":
            scale = np.std(clean_signal, axis=1)
        elif config.snr_mode == "peak":
            scale = np.max(np.abs(clean_signal), axis=1)
        else:
            raise ValueError("snr_mode must be one of {'std', 'peak'}")
        return np.where(scale > 0, scale / snr, 0.0)

    return np.full(n_channels, default_sigma, dtype=float)


def _resolve_kernels(config: SimulationConfig) -> tuple[np.ndarray | None, int, bool]:
    if config.unit_size_bp is not None and config.unit_size_bp <= 0:
        raise ValueError("unit_size_bp must be positive when provided")

    if config.cassette_map is not None:
        if config.ms2_pattern is not None or config.ms2_kernel is not None:
            raise ValueError("cassette_map cannot be combined with ms2_pattern or ms2_kernel")
        kernels = kernels_from_cassette_map(config.cassette_map, config.n_channels)
        return kernels, kernels.shape[1], True

    if config.n_channels > 1:
        raise ValueError("n_channels > 1 requires cassette_map")

    if config.ms2_kernel is not None:
        kernel = np.asarray(config.ms2_kernel, dtype=float)
        if kernel.ndim != 1 or kernel.size == 0:
            raise ValueError("ms2_kernel must be a non-empty 1D array")
        return kernel[None, :], kernel.size, True

    if config.ms2_pattern is not None:
        kernel = kernel_from_pattern(config.ms2_pattern)
        return kernel[None, :], kernel.size, True

    if config.memory_steps is None or config.alpha is None:
        raise ValueError("memory_steps and alpha are required when no kernel/pattern/cassette_map is provided")
    return None, config.memory_steps, False


def simulate_trace(model: PromoterModel, config: SimulationConfig, rng=None) -> TraceResult:
    rng = rng or np.random.default_rng()
    k = model.rate_matrix.shape[0]
    t_max = config.seq_length * config.delta_t
    times_unif = np.arange(1, config.seq_length + 1) * config.delta_t

    kernels, memory_steps, use_step_kernel = _resolve_kernels(config)
    n_channels = 1 if kernels is None else kernels.shape[0]

    state = rng.choice(np.arange(k), p=model.pi0)
    states = [state]
    times = [0.0]
    t = 0.0

    while t < t_max:
        lam = -model.rate_matrix[state, state]
        t += rng.exponential(1.0 / lam)
        rates = model.rate_matrix[:, state].copy()
        rates[state] = 0
        state = int(rng.choice(np.arange(k), p=rates / lam))
        states.append(state)
        times.append(t)

    tt = np.asarray(times)
    ss = np.asarray(states)
    fluo_clean = np.zeros(config.seq_length)
    fluo_ms2_clean = np.zeros((n_channels, config.seq_length))

    for idx, t_end in enumerate(times_unif):
        t_start = max(0.0, t_end - memory_steps * config.delta_t)
        i_start = np.searchsorted(tt, t_start, side="right") - 1
        i_end = np.searchsorted(tt, t_end, side="right") - 1
        tw = tt[i_start : i_end + 2].copy()
        tw[0] = t_start
        tw[-1] = t_end
        sw = ss[i_start : i_end + 2]

        for i in range(len(sw) - 1):
            t1 = t_end - tw[i + 1]
            t2 = t_end - tw[i]
            emit = model.emission_rates[sw[i]]
            if use_step_kernel:
                for ch in range(n_channels):
                    fluo_ms2_clean[ch, idx] += emit * integrate_step_kernel(kernels[ch], config.delta_t, t1, t2)
            else:
                fluo_ms2_clean[0, idx] += emit * ms2_loading_coeff_integral(
                    config.alpha, memory_steps, config.delta_t, t1, t2
                )
            fluo_clean[idx] += emit * (t2 - t1)

    fluo_noise = rng.normal(0, model.sigma, size=config.seq_length)
    fluo = fluo_clean + fluo_noise

    sigmas = _resolve_channel_sigmas(config, fluo_ms2_clean, model.sigma)
    channel_noise = np.vstack([rng.normal(0, sigmas[ch], size=config.seq_length) for ch in range(n_channels)])
    fluo_ms2_channels = fluo_ms2_clean + channel_noise

    if n_channels == 1:
        return TraceResult(
            fluo=fluo,
            fluo_ms2=fluo_ms2_channels[0],
            transition_times=tt,
            naive_states=ss,
            fluo_ms2_channels=fluo_ms2_channels,
        )

    return TraceResult(
        fluo=fluo,
        fluo_ms2=fluo_ms2_channels[0],
        transition_times=tt,
        naive_states=ss,
        fluo_ms2_channels=fluo_ms2_channels,
    )
