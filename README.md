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

## MS2 kernel options

`SimulationConfig` supports three ways to define MS2 convolution behavior:

1. **Legacy ramp model** using `alpha` + `memory_steps`.
2. **Pattern-derived kernel** using `ms2_pattern` (0/1 gene-body bins) and optional `unit_size_bp`.
3. **Direct kernel override** using `ms2_kernel` for a custom age kernel.

## Multi-channel cassette maps

To simulate multiple channels from one shared gene body, use:

- `n_channels`: number of fluorescence channels.
- `cassette_map`: integer vector across bins with values in `[0, n_channels]`.
  - `0`: no cassette in this bin
  - `1..n_channels`: cassette assigned to that channel

The simulator converts this shared map into one convolution kernel per channel.

## Per-channel noise and SNR

Optional channel-level observation noise controls:

- `channel_noise_sigma`: explicit per-channel Gaussian noise sigma.
- `channel_snr`: optional per-channel SNR target, converted into sigma using:
  - `snr_mode="std"` (default): `sigma = std(clean_signal) / SNR`
  - `snr_mode="peak"`: `sigma = max(abs(clean_signal)) / SNR`

When omitted, each channel defaults to `PromoterModel.sigma`.
