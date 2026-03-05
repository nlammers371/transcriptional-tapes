# transcriptional-tapes

Simulation-only Python scaffold for Gillespie + MS2 trace generation.

## Conceptual overview (Gillespie simulation assumptions)

1. Promoter states evolve as a continuous-time Markov chain (CTMC) with rate matrix `R`.
2. Transition times are sampled with Gillespie SSA using exponential dwell times.
3. Emission rate depends on promoter state.
4. MS2 signal has finite memory (`w`) and finite loop-loading ramp (`alpha`).
5. Continuous-time latent dynamics are sampled onto a discrete acquisition grid (`delta_t`).
6. Gaussian noise is added to generate observed fluorescence.

## Contents

- `src/transcriptional_tapes/`: package source.
- `notebooks/simulation_quickstart.ipynb`: starter notebook.
- `tests/`: basic smoke test.
