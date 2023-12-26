from typing import Dict, List, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression

from shadowing.utils import windows


def kernel_pl(taus: np.ndarray, delta: float, alpha: float):
    return (taus + delta) ** (-alpha)


def kernel_exp(taus: np.ndarray, lam: float):
    return lam * np.exp(-lam * taus)


def get_RV(x: np.ndarray, from_dln: bool = False):
    """ Given prices x computes realized volatility. """
    if from_dln:
        annualizer = x.shape[-1] / 252
        return ((x ** 2).sum(-1) / annualizer) ** 0.5
    annualizer = (x.shape[-1] - 1) / 252
    return ((np.diff(np.log(x)) ** 2).sum(-1) / annualizer) ** 0.5


DEFAULT1 = {
    'power-law': {'delta': 0.044, 'alpha': 2.82},
    # 'exp': {'lam0': 55.0, 'lam1': 10.0, 'theta': 0.25}
    'exp': {'lam0': 64.5, 'lam1': 3.83, 'theta': 0.67}
}
DEFAULT2 = {
    'power-law': {'delta': 0.025, 'alpha': 1.86},
    # 'exp': {'lam0': 20.0, 'lam1': 3.0, 'theta': 0.5}
    'exp': {'lam0': 37.6, 'lam1': 1.2, 'theta': 0.2}
}


class AutoregressiveVolModel:

    def __init__(
        self,
        T: int,
        w: int,
        s: int,
        dt: int,
        ktype: str,
        k1_dict: Dict | None = None,
        k2_dict: Dict | None = None
    ):
        self.T = T  # number of days
        self.w = w  # number of dts in the past
        self.s = s  # training stride periods
        self.dt = dt  # time interval
        if k1_dict == None:
            k1_dict = DEFAULT1[ktype]
        if k2_dict == None:
            k2_dict = DEFAULT2[ktype]
        if ktype == "power-law":
            self.k1 = self.init_pl_kernel(w=w, dt=dt, **k1_dict)
            self.k2 = self.init_pl_kernel(w=w, dt=dt, **k2_dict)
        else:
            self.k1 = self.init_exp_kernel_2_factors(w=w, dt=dt, **k1_dict)
            self.k2 = self.init_exp_kernel_2_factors(w=w, dt=dt, **k2_dict)
        self.linreg = LinearRegression(fit_intercept=False)

    @staticmethod
    def init_exp_kernel_2_factors(
        w: int,
        dt: int,
        lam0: float,
        lam1: float,
        theta: float
    ) -> np.ndarray:
        taus = np.arange(w)[::-1] * dt

        k0 = kernel_exp(taus, lam=lam0)
        k1 = kernel_exp(taus, lam=lam1)
        k0 = k0 / k0.sum() / dt
        k1 = k1 / k1.sum() / dt
        kernel = (1 - theta) * k0 + theta * k1

        return kernel

    @staticmethod
    def init_pl_kernel(
        w: int,
        dt: float,
        delta: float,
        alpha: float
    ) -> np.ndarray:
        """ Given offset and exponent of a power law, return the kernel. """
        taus = np.arange(w)[::-1] * dt
        kernel = kernel_pl(taus, delta=delta, alpha=alpha)
        return kernel * 252 / kernel.sum()

    def separate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Window time-series x. """
        assert x.ndim == 1
        w_params = {'w': self.w + 1 + self.T + 1, 's': self.s, 'offset': 0}
        indices = windows(np.arange(x.size), **w_params)
        idx_x, idx_y = indices[:, :-self.T - 1], indices[:, -self.T - 1:]
        x_w = windows(x, **w_params)
        x_train = np.diff(np.log(x_w[:, :-self.T - 1]))
        y_train = get_RV(x_w[:, -self.T - 1:])
        return idx_x, idx_y, x_train, y_train

    @staticmethod
    def apply_kernel(
        dlnx: np.ndarray, 
        k1: np.ndarray, 
        k2: np.ndarray
    ) -> np.ndarray:
        assert dlnx.shape[-1] == k1.size == k2.size
        R1t = (dlnx * k1).sum(-1)
        R2t = ((dlnx ** 2) * k2).sum(-1) ** 0.5

        return np.stack([np.ones_like(R1t), R1t, R2t], axis=-1)

    def train(self, x: np.ndarray) -> None:
        """ x should be of granularity dt """
        # preparing input
        _, _, dlnx, y = self.separate(x)
        X = self.apply_kernel(dlnx, self.k1, self.k2)

        # linear regression
        self.linreg.fit(X, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        # preparing input
        if x.ndim == 1:
            _, _, dlnx, _ = self.separate(x)
        else:
            dlnx = np.diff(np.log(x[..., -self.w - 1:]))
        X = self.apply_kernel(dlnx, self.k1, self.k2)
        return self.linreg.predict(X)


class PDVModel:
    def __init__(
        self,
        lams1: List[float],
        lams2: List[float],
        thetas: List[float],
        betas: List[float]
    ):
        """
        lams1: parameters of kernel on dx
        lams2: parameters of kernel on |dx|^2
        """
        self.lams1 = np.array(lams1)  # scales for kernel 1 on returns
        self.lams2 = np.array(lams2)  # scales for kernel 2 on vols
        self.thetas = np.array(thetas)  # kernel mixing parameter
        self.betas = np.array(betas)  # regressive vol weight

    def gen_dw(self, size: Tuple, m: float, s: float) -> np.ndarray:
        """ White noise source of randomness. """
        return m + s * np.random.randn(size)

    def mixing(self, theta: float, X: np.ndarray) -> np.ndarray:
        """ Convex combination of X. """
        return (1 - theta) * X[0] + theta * X[1]

    def sigma(self, R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
        """ Regressive model of sigma along factor R1 on past returns and R2 on past vols. """
        r1 = self.mixing(self.thetas[0], R1)
        r2 = self.mixing(self.thetas[1], R2)
        sigma = self.betas[0] + self.betas[1] * r1 + self.betas[2] * r2 ** 0.5
        # return np.clip(sigma, -1e16, 1.5)
        return np.clip(sigma, 0.0, 1.5)

    def actualize_factors(
        self,
        R1: np.ndarray,
        R2: np.ndarray,
        dt: float,
        dwt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Apply exponential kernel on factors. """
        sigma_curr = self.sigma(R1, R2)
        dR1 = (sigma_curr * dwt - R1 * dt) * self.lams1
        dR2 = (sigma_curr ** 2 - R2) * dt * self.lams2
        return R1 + dR1, R2 + dR2

    def gen(
        self,
        T: int,
        dt: float,
        S0: float,
        R10: np.ndarray,
        R20: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Generate an instanteneous path sigma_t and price path S_t. """
        n_steps = int(T / dt)
        S = np.ones(n_steps) * S0
        sigma = np.zeros(n_steps)

        dW = self.gen_dw(m=0.0, s=np.sqrt(dt), size=n_steps-1)

        R1_curr, R2_curr = np.array(R10), np.array(R20)
        sigma[0] = self.sigma(R1_curr, R2_curr)

        # perform n_steps - 1
        for t in range(1, n_steps):
            dwt = dW[t-1]
            sigma[t] = self.sigma(R1_curr, R2_curr)
            S[t] = S[t-1] * (1 + sigma[t] * dwt)
            R1_curr, R2_curr = self.actualize_factors(R1_curr, R2_curr, dt, dwt)
        return sigma, S
    

def compute_factor(
    x_past: np.ndarray,
    pdv_model: PDVModel,
    w: int,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """ Given past trajectory, compute initial factors. """
    dlnx = np.diff(np.log(x_past))

    # retrieve kernels
    taus = np.arange(w)[::-1][1:] * dt
    k10 = kernel_exp(taus, lam=pdv_model.lams1[0])
    k11 = kernel_exp(taus, lam=pdv_model.lams1[1])
    k20 = kernel_exp(taus, lam=pdv_model.lams2[0])
    k21 = kernel_exp(taus, lam=pdv_model.lams2[1])
    k10 = k10 / k10.sum() / dt
    k11 = k11 / k11.sum() / dt
    k20 = k20 / k20.sum() / dt
    k21 = k21 / k21.sum() / dt

    # compute R10, R20
    R100, R200 = AutoregressiveVolModel.apply_kernel(dlnx, k10, k20)[0, 1:]
    R110, R210 = AutoregressiveVolModel.apply_kernel(dlnx, k11, k21)[0, 1:]
    R10 = np.array([R100, R110])
    R20 = np.array([R200, R210]) ** 2.0  # IMPORTANT

    return R10, R20


def future_pdv_model(
    x_past: np.ndarray,
    pdv_model: PDVModel,
    w: int,
    S0: float,
    S: int,
    T: int,
    dt: float
) -> np.ndarray:
    """ Given past trajectory, compute initial factors at present and return trajectories. """
    R10, R20 = compute_factor(x_past, pdv_model, w, dt)

    # generate
    x_gen = []
    for s in range(S):
        _, new_gen = pdv_model.gen(T=T, dt=dt, S0=S0, R10=R10, R20=R20)
        x_gen.append(new_gen)

    return np.stack(x_gen)
