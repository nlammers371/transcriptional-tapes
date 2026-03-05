from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class PromoterModel:
    rate_matrix: np.ndarray
    emission_rates: np.ndarray
    sigma: float
    pi0: np.ndarray


@dataclass(slots=True)
class SimulationConfig:
    seq_length: int
    delta_t: float
    memory_steps: int | None = None
    alpha: float | None = None
    unit_size_bp: float | None = None
    ms2_pattern: np.ndarray | None = None
    ms2_kernel: np.ndarray | None = None
