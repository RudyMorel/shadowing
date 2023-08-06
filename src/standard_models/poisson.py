""" Poisson process. """
import numpy as np
from numpy.random import default_rng
from scipy.stats import poisson


def poisson_t(t, n, signed):
    """ Poisson process on [0,t] with n jumps. """
    x = np.zeros(t)
    positions = default_rng().choice(t, size=n, replace=False)
    sign = 2 * (np.random.rand(n) > (0.5 if signed else 0.0)) - 1
    x[positions] = sign
    return x.cumsum()


def poisson_mu(T, mu, signed):
    """ Poisson process on [0,t] with intensity mu. """
    proba = 1 - sum([poisson.pmf(k, mu * T) for k in range(T+1)])  # proba of having pbs
    if proba > 0.025:
        raise ValueError("Impossible to simulate Poisson. Increase the interval or decrease intensity.")
    n = poisson.rvs(T * mu)
    X = poisson_t(T, min(n, T), signed)
    return X
