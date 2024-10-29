""" Path-Dependent Volatility model Guyon, Lekeufack, 2024
https://www.tandfonline.com/doi/abs/10.1080/14697688.2023.2221281 """
from typing import Dict, List, Tuple, Literal
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import t

from scatspectra import PriceData, windows


def kernel_pl(taus: np.ndarray, delta: float, alpha: float):
    """ Power-law kernel with a lag delta to avoid. """
    return (taus + delta) ** (-alpha)


def kernel_exp(taus: np.ndarray, lam: float):
    """ Exponential kernel. """
    return lam * np.exp(-lam * taus)


def get_RV(x: np.ndarray, from_dln: bool = False):
    """ Given prices x computes realized volatility. """
    if from_dln:
        annualizer = x.shape[-1] / 252
        return ((x ** 2).sum(-1) / annualizer) ** 0.5
    annualizer = (x.shape[-1] - 1) / 252
    return ((np.diff(np.log(x)) ** 2).sum(-1) / annualizer) ** 0.5


# default values from Guyon, Lekeufack, 2024
DEFAULT1 = {
    'power-law': {'delta': 0.044, 'alpha': 2.82},
    'exp': {'lam0': 64.5, 'lam1': 3.83, 'theta': 0.67}
}
DEFAULT2 = {
    'power-law': {'delta': 0.025, 'alpha': 1.86},
    'exp': {'lam0': 37.6, 'lam1': 1.2, 'theta': 0.2}
}


class AutoregressiveLinearPredictor:
    """ Autoregressive Volatility Model. Regresses future volatility on past 
    returns and past square returns, see Guyon, Lekeufack, 2024
    https://www.tandfonline.com/doi/abs/10.1080/14697688.2023.2221281
    """
    def __init__(
        self,
        T: int,
        w: int,
        s: int,
        dt: float,
        ktype: Literal["exp", "power-law"],
        k1_dict: Dict | None = None,
        k2_dict: Dict | None = None,
        extra_term: bool = False
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
        self.extra_term = extra_term

    @staticmethod
    def init_exp_kernel_2_factors(
        w: int,
        dt: float,
        lam0: float,
        lam1: float,
        theta: float
    ) -> np.ndarray:
        """ Return a linear combination of two exponential kernels."""
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
        """ Separate the timeseries x into chunks of past returns x and future returns y. """
        assert x.ndim == 1
        w_params = {'w': self.w + 1 + self.T, 's': self.s, 'offset': 0}
        indices = windows(np.arange(x.size), **w_params)
        idx_x, idx_y = indices[:, :-self.T-1], indices[:, -self.T-1:]
        x_w = windows(x, **w_params)
        # the x and y have an overlap of 1 day: they share 1 price in common
        # but their increments are disjoint
        x_train = np.diff(np.log(x_w[:, :self.w+1]))
        y_train = get_RV(x_w[:, self.w:])
        return idx_x, idx_y, x_train, y_train

    @staticmethod
    def embedding(
        dlnx: np.ndarray, 
        k1: np.ndarray, 
        k2: np.ndarray,
        extra_term: bool = False
    ) -> np.ndarray:
        """ Compute the embedding of the past through kernels on returns 
        and square returns. 
        
        :param dlnx: past returns, shape (B, T)
        :param k1: kernel on returns, shape (T,)
        :param k2: kernel on square returns, shape (T,)
        :param extra_term: whether to add the extra term in Guyon, Lekeufack's paper 
        """
        assert dlnx.shape[-1] == k1.size == k2.size
        R1t = (dlnx * k1).sum(-1)
        R2t = ((dlnx ** 2) * k2).sum(-1) ** 0.5
        emb = [np.ones_like(R1t), R1t, R2t]
        if extra_term:
            R12t = (0.5*np.abs(R1t) + 0.5*R1t) ** 2
            emb += [R12t]
        return np.stack(emb, axis=-1)

    def train(self, x: np.ndarray) -> None:
        """ Train the linear model. x should be of granularity dt """
        # preparing input: separate into regressor and regressed
        _, _, dlnx, y = self.separate(x)
        X = self.embedding(dlnx, self.k1, self.k2, self.extra_term)

        # linear regression
        self.linreg.fit(X, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """ Given an input x, predict the future volatility from the trained
        linear model. 
        
        :param x: the input timeseries (typically log-returns), 
            shape (B, T), B = nb of dates to do prediction on
        """
        # preparing input
        X = self.embedding(x, self.k1, self.k2, self.extra_term)

        # linear prediction
        y = self.linreg.predict(X)

        return y


class PDVModel:
    """ Path-Dependent Volatility model Guyon, Lekeufack, 2024
    https://www.tandfonline.com/doi/abs/10.1080/14697688.2023.2221281 """

    def __init__(
        self,
        lams1: List[float],
        lams2: List[float],
        thetas: List[float],
        betas: List[float],
        snp: PriceData | None = None,
        nu: float | None = None
    ):
        """
        lams1: parameters of kernel on dx
        lams2: parameters of kernel on |dx|^2
        """
        self.lams1 = np.array(lams1)  # scales for kernel 1 on returns
        self.lams2 = np.array(lams2)  # scales for kernel 2 on vols
        self.thetas = np.array(thetas)  # kernel mixing parameter
        self.betas = np.array(betas)  # regressive vol weight
        self.snp = snp  # in case the returns are fitted on the snp marginal distribution
        self.nu = nu
        self.fit_params = None
        self.dlnx_dist = None
        if snp is not None:
            self.calibrate_log_returns(self.snp)
        if nu is not None:
            self.define_dlnx_dist(nu)
    
    def define_dlnx_dist(self, nu: float):
        self.dlnx_dist = t(loc=0.0, scale=1.0, df=nu)

    def calibrate_log_returns(self, snp: PriceData):
        # rescale snp distribution with the PDV vol
        data = snp.dlnx.ravel().copy()
        self.fit_params = t.fit(data)
        self.dlnx_dist = t(*self.fit_params)

    def gen_dw(self, s: float, size: Tuple) -> np.ndarray:
        """ Source of randomness. """
        if self.snp is not None or self.nu is not None:
            dw = self.dlnx_dist.rvs(size=size)
        else:
            dw = np.random.randn(*size)
        dw -= dw.mean()
        dw /= dw.std()
        dw *= s
        return dw

    def mixing(self, theta: float, X: np.ndarray) -> np.ndarray:
        """ Convex combination of X. """
        return (1 - theta) * X[0] + theta * X[1]

    def sigma(self, R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
        """ Regressive model of sigma along factor R1 on past returns and R2 on past vols. """
        r1 = self.mixing(self.thetas[0], R1)
        r2 = self.mixing(self.thetas[1], R2) 
        sigma = self.betas[0] + self.betas[1] * r1 + self.betas[2] * r2 ** 0.5
        if len(self.betas) > 3:
            sigma += self.betas[3] * (0.5*np.abs(r1) + 0.5*r1) ** 2
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

        dW = self.gen_dw(s=np.sqrt(dt), size=(n_steps-1,))

        R1_curr, R2_curr = np.array(R10), np.array(R20)
        sigma[0] = self.sigma(R1_curr, R2_curr)

        # perform n_steps - 1
        for t in range(1, n_steps):
            dwt = dW[t-1]
            sigma[t] = self.sigma(R1_curr, R2_curr)
            S[t] = S[t-1] * (1 + sigma[t] * dwt)
            R1_curr, R2_curr = self.actualize_factors(R1_curr, R2_curr, dt, dwt)
            
        return sigma, S


class PDVModelDiscrete:
    """ Our own discrete version of the PDV model, largely inspired from 
    Guyon, Lekeufack, 2024
    https://www.tandfonline.com/doi/abs/10.1080/14697688.2023.2221281 """

    def __init__(
        self,
        lams1: List[float],
        lams2: List[float],
        thetas: List[float],
        betas: List[float],
        snp: PriceData | None = None,
        nu: float | None = None
    ):
        """
        lams1: parameters of kernel on dx
        lams2: parameters of kernel on |dx|^2
        """
        self.lams1 = np.array(lams1)  # scales for kernel 1 on returns
        self.lams2 = np.array(lams2)  # scales for kernel 2 on vols
        self.thetas = np.array(thetas)  # kernel mixing parameter
        self.betas = np.array(betas)  # regressive vol weight
        self.snp = snp  # in case the returns are fitted on the snp marginal distribution
        self.nu = nu
        self.fit_params = None
        self.dlnx_dist = None
        if snp is not None:
            self.calibrate_log_returns(self.snp)
        if nu is not None:
            self.define_dlnx_dist(nu)
    
    def define_dlnx_dist(self, nu: float):
        self.dlnx_dist = t(loc=0.0, scale=1.0, df=nu)

    def calibrate_log_returns(self, snp: PriceData):
        # rescale snp distribution with the PDV vol
        data = snp.dlnx.ravel().copy()
        self.fit_params = t.fit(data)
        self.dlnx_dist = t(*self.fit_params)

    def gen_dw(self, s: float, size: Tuple) -> np.ndarray:
        """ Source of randomness. """
        if self.snp is not None or self.nu is not None:
            dw = self.dlnx_dist.rvs(size=size)
        else:
            dw = np.random.randn(*size)
        dw -= dw.mean(-1, keepdims=True)
        dw /= dw.std(-1, keepdims=True)
        dw *= s
        return dw

    def mixing(self, theta: float, X: np.ndarray) -> np.ndarray:
        """ Convex combination of X. """
        return (1 - theta) * X[:,0] + theta * X[:,1]

    def sigma(self, R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
        """ Regressive model of sigma along factor R1 on past returns and R2 on past vols. """
        r1 = self.mixing(self.thetas[0], R1)
        r2 = self.mixing(self.thetas[1], R2) 
        sigma = self.betas[0] + self.betas[1] * r1 + self.betas[2] * r2 ** 0.5
        if len(self.betas) > 3:
            sigma += self.betas[3] * (0.5*np.abs(r1) + 0.5*r1) ** 2
        return np.clip(sigma, 0.0, 1.5)

    def actualize_factors(
        self,
        R1: np.ndarray,
        R2: np.ndarray,
        dwt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Apply exponential kernel on factors. 
        Here dwt is the return from t to t+1: it includes the prediction of the volatility."""
        # sigma_curr = self.sigma(R1, R2)
        R1_next = np.exp(-self.lams1[None,:]/252) * R1 + self.lams1[None,:] * dwt[:,None]
        # dR1 = (sigma_curr * dwt - R1 * dt) * self.lams1
        R2_next = np.exp(-self.lams2[None,:]/252) * R2 + self.lams2[None,:] * dwt[:,None] ** 2
        # dR2 = (sigma_curr ** 2 - R2) * dt * self.lams2
        return R1_next, R2_next

    def gen(
        self,
        T: int,
        dt: float,
        S0: float,
        S: int,
        R10: np.ndarray,
        R20: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Generate an instanteneous path sigma_t and price path S_t. """

        if np.abs(dt-1/252) > 1e-6:
            raise ValueError("dt should be 1.0 in the discrete model")
        n_steps = int(T / dt)
        St = np.ones((S, n_steps)) * S0
        sigma = np.zeros((S, n_steps))

        dW = self.gen_dw(s=np.sqrt(dt), size=(S,n_steps))

        R1_curr, R2_curr = np.array(R10), np.array(R20)
        R1_curr = np.repeat(R1_curr[None,:], S, axis=0)
        R2_curr = np.repeat(R2_curr[None,:], S, axis=0)
        sigma[:,0] = self.sigma(R1_curr, R2_curr)

        # perform n_steps - 1
        for t in range(1, n_steps):
            dwt = dW[:,t]
            sigma[:,t] = self.sigma(R1_curr, R2_curr)
            rt = sigma[:,t] * dwt
            rt = np.maximum(rt, -0.999999)  # rt cannot be negative
            St[:,t] = St[:,t-1] * (1 + rt)
            R1_curr, R2_curr = self.actualize_factors(R1_curr, R2_curr, rt)
            
        return sigma, St
  

def compute_factor(
    x_past: np.ndarray,
    pdv_model: PDVModel | PDVModelDiscrete,
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
    R100, R200, _ = AutoregressiveLinearPredictor.embedding(dlnx, k10, k20, extra_term=len(pdv_model.betas) > 3)[0, 1:]
    R110, R210, _ = AutoregressiveLinearPredictor.embedding(dlnx, k11, k21, extra_term=len(pdv_model.betas) > 3)[0, 1:]
    R10 = np.array([R100, R110])
    R20 = np.array([R200, R210]) ** 2.0

    return R10, R20


def future_pdv_model(
    x_past: np.ndarray,
    pdv_model: PDVModel | PDVModelDiscrete,
    w: int,
    S0: float,
    S: int,
    T: int,
    dt: float
) -> np.ndarray:
    """ Given past trajectory, compute initial factors at present and return trajectories. """
    R10, R20 = compute_factor(x_past, pdv_model, w, dt)

    # generate
    _, x_gen = pdv_model.gen(T=T, dt=dt, S0=S0, S=S, R10=R10, R20=R20)

    return x_gen
