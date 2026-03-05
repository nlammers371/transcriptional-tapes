import numpy as np

from .gillespie import simulate_trace


def simulate_dataset(model, config, n_traces, seed=None):
    rng = np.random.default_rng(seed)
    observed = np.zeros((config.seq_length, n_traces))
    traces = []
    for i in range(n_traces):
        tr = simulate_trace(model, config, rng)
        traces.append(tr)
        observed[:, i] = tr.fluo_ms2
    return {"observed_fluo": observed, "traces": traces}
