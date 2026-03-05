import numpy as np

from .gillespie import simulate_trace


def simulate_dataset(model, config, n_traces, seed=None):
    rng = np.random.default_rng(seed)
    traces = []

    first = simulate_trace(model, config, rng)
    traces.append(first)

    channels = first.fluo_ms2_channels.shape[0] if first.fluo_ms2_channels is not None else 1
    observed_by_channel = np.zeros((channels, config.seq_length, n_traces))
    observed_by_channel[:, :, 0] = first.fluo_ms2_channels

    for i in range(1, n_traces):
        tr = simulate_trace(model, config, rng)
        traces.append(tr)
        observed_by_channel[:, :, i] = tr.fluo_ms2_channels

    out = {
        "observed_fluo": observed_by_channel[0],
        "traces": traces,
    }
    if channels > 1:
        out["observed_fluo_by_channel"] = observed_by_channel
    return out
