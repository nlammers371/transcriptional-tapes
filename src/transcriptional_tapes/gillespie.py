from dataclasses import dataclass
import numpy as np

from .model import PromoterModel, SimulationConfig
from .ms2 import ms2_loading_coeff_integral


@dataclass(slots=True)
class TraceResult:
    fluo: np.ndarray
    fluo_ms2: np.ndarray
    transition_times: np.ndarray
    naive_states: np.ndarray


def simulate_trace(model: PromoterModel, config: SimulationConfig, rng=None) -> TraceResult:
    rng = rng or np.random.default_rng()
    k = model.rate_matrix.shape[0]
    t_max = config.seq_length * config.delta_t
    times_unif = np.arange(1, config.seq_length + 1) * config.delta_t

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
    fluo = np.zeros(config.seq_length)
    fluo_ms2 = np.zeros(config.seq_length)

    for idx, t_end in enumerate(times_unif):
        t_start = max(0.0, t_end - config.memory_steps * config.delta_t)
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
            fluo_ms2[idx] += emit * ms2_loading_coeff_integral(
                config.alpha, config.memory_steps, config.delta_t, t1, t2
            )
            fluo[idx] += emit * (t2 - t1)

    noise = rng.normal(0, model.sigma, size=config.seq_length)
    return TraceResult(fluo=fluo + noise, fluo_ms2=fluo_ms2 + noise, transition_times=tt, naive_states=ss)
