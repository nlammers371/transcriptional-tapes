import numpy as np

from .model import PromoterModel


def two_state_default():
    r = np.array([[-0.015, 0.02], [0.015, -0.02]], dtype=float)
    v = np.array([0.0, 4.0]) / 20.0
    pi = np.array([0.57142857, 0.42857143])
    return PromoterModel(rate_matrix=r, emission_rates=v, sigma=1.5, pi0=pi)
