from typing import Iterable
import numpy as np


def realized_variance(x: np.ndarray,
                      Ts: Iterable,
                      vol: bool):
    x2 = x ** 2
    r_variance = np.stack([x2[..., :T].mean((-1,-2)) for T in Ts], -1) * 252
    if vol:
        return r_variance ** 0.5
    return r_variance