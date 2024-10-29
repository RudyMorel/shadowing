from typing import Iterable
import numpy as np


def realized_variance(x: np.ndarray, Ts: Iterable, vol: bool):
    """ Computes the realized variance of x at different maturities.

    :param x: log-return array (..., T)
    :param Ts: the different maturities
    :param vol: if True, returns the realized volatility
    :return: array (..., len(Ts))
    """
    x2 = x ** 2
    r_variance = np.stack([x2[..., :T].mean(-1) for T in Ts], -1) * 252
    if vol:
        return r_variance ** 0.5
    return r_variance