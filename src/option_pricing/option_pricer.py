from typing import *
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import brentq

from src.data_source import PriceData
from src.option_pricing.black_scholes import price_BS
from src.option_pricing.smile import Smile
from src.utils import Uniform


def implied_vol(price: float, K: float, T: float, S0: float, r: float, ignore_warning=False) -> float:
    """Given a price, compute the vol sigma in BS formula that matches this price."""
    a, b = 1e-6, 10.0
    func = lambda s: price_BS(K, T, s, S0, r) - price

    if func(a) > 0 or func(b) < 0:
        if not ignore_warning:
            print("WARNING. Implied_vol solver could not find a solution.")
        return a if func(a) > 0 else b
    return brentq(func, a, b)


class HMCPricer:
    def __init__(self,
                 M: int,
                 ave: Optional = None,
                 detrend: Optional[bool] = False,
                 K_bounds: Optional[list] = None,
                 basis_func_method: str = 'piecewise_quadratic'):
        self.ave = ave or Uniform()
        self.M = M
        self.detrend = detrend
        self.K_bounds = K_bounds
        self.basis_func_method = basis_func_method

    def reset_Ks(self, dt: int) -> None:
        if self.K_bounds is not None:
            Ka, Kb = self.K_bounds
            self.Ks = np.linspace(Ka, Kb, self.M) + 10
        else:
            M_max = 8.0
            Ms = np.linspace(-M_max, M_max, self.M)
            self.Ks = Smile.from_M_to_K(0.2, dt, Ms)

    def get_price(self, x: np.ndarray) -> np.ndarray:
        """ Get prices C(xs). """
        if self.basis_func_method != 'piecewise_quadratic':
            raise ValueError("Unkwnown basis function method in HMC")

        return np.stack([(x > k) * (x - k) ** 2 for k in self.Ks])

    def get_hedge(self, x: np.ndarray) -> np.ndarray:
        """ Get hedges phi(xs). """
        if self.basis_func_method != 'piecewise_quadratic':
            raise ValueError("Unkwnown basis function method in HMC")

        return np.stack([2 * (x > k) * (x - k) for k in self.Ks])

    def perform_iteration(self,
                          it: int,
                          x_curr: np.ndarray,
                          x_prev: np.ndarray,
                          discount: float,
                          param_curr: Optional[np.ndarray],
                          price_curr: Optional[np.ndarray]
                          ) -> np.ndarray:
        """ From param_curr get param_prev. They are obtained by linear regression of price_curr
        against previous price + previous hedge. """
        if param_curr is None:
            Y = price_curr  # C_T(x)
        elif price_curr is None:
            Y = param_curr @ self.get_price(x_curr)  # C_{k+1}(x_{k+1})
        else:
            raise ValueError("Step in HMC requires price_curr or param_curr")

        self.reset_Ks(it)

        # C_k + phi_k (x_{k+1} - x_k)
        X = discount * self.get_price(x_prev) + self.get_hedge(x_prev) * (discount * x_curr - x_prev)[None, :]

        regr = LinearRegression(fit_intercept=False)
        regr.fit(X.T, Y)
        param_prev = regr.coef_

        # param_test = scipy.linalg.pinv(X @ X.T) @ X @ Y

        return param_prev

    def price(self,
              x: np.ndarray,
              strike: float
              ) -> Tuple[float, Callable, Callable]:
        """ Price a European Call option given several price paths X.

        Keyword arguments:
        X -- array of shape B x (N+1)
        """
        N = x.shape[-1] - 1

        y = x
        if self.detrend:
            dlny = np.diff(np.log(x), axis=-1)
            dlny -= dlny.mean(0, keepdims=True)
            y = PriceData(dlnx=dlny, x_init=100.0).x

        # iterate backward to determine current params in Longstaff-Schwartz
        price_curr = (y[:, -1] - strike) * (y[:, -1] > strike)
        param_curr = None
        it = 1
        for n in np.arange(1, N + 1)[::-1]:
            #             discount = np.exp(-r * T / N)  # TODO. Verify the discount, T / N or N / T ?
            discount = 1.0
            param_curr = self.perform_iteration(it, y[:, n], y[:, n - 1], discount, param_curr, price_curr if n == N else None)
            it += 1

        # go back to the price
        prices = param_curr @ self.get_price(y[:, 0])

        return self.ave.avg(prices, axis=-1), lambda x: param_curr @ self.get_price(x), lambda x: param_curr @ self.get_hedge(x)
