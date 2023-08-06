import numpy as np
from scipy.stats import norm


def price_BS(K, T, sigma, S0, r):
    """ Black-Scholes option pricing formula. """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / sigma / T ** 0.5
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / sigma / T ** 0.5

    price = (S0 * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))

    return price


def delta_BS(K, T, sigma, S0, r):
    """ Black-Scholes option delta hedge. """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / sigma / T ** 0.5

    return norm.cdf(d1, 0.0, 1.0)
