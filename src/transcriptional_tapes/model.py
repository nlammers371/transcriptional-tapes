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
    memory_steps: int
    alpha: float
