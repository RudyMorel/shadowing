import numpy as np
from scipy.stats import rv_continuous, gennorm
from scipy.special import gamma, beta, softmax


class DiscreteProba:
    def __init__(self, w):
        self.w = w

    def avg(self, x, axis, keepdims=False):
        return np.average(x, axis, self.w, keepdims=keepdims)

    def variance(self, x, axis):
        xmean = self.avg(x, axis, keepdims=True)
        return self.avg((x - xmean) ** 2, axis)

    def std(self, x, axis):
        return self.variance(x, axis) ** 0.5


class Softmax(DiscreteProba):
    def __init__(self, l2s, eta):
        weights = softmax(-l2s ** 2 / 2 / eta ** 2)
        super(Softmax, self).__init__(weights)


class Uniform(DiscreteProba):
    def __init__(self):
        super(Uniform, self).__init__(None)

    def avg(self, x, axis, keepdims=False):
        return x.mean(axis, keepdims=keepdims)


class GeneralizedNormal(rv_continuous):
    def __init__(self, alpha):
        super(GeneralizedNormal, self).__init__()
        if isinstance(alpha, np.ndarray):
            alpha = alpha.item()
        self.alpha = alpha
        self.X = gennorm(alpha)
        self.lam = np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))  # lam such that lam X has E|X|^2 = 1

    def _pdf(self, x):
        return self.X.pdf(x / self.lam) / self.lam

    def _cdf(self, x):
        return self.X.cdf(x / self.lam)

    def sparsity(self):
        return gamma(2 / self.alpha) ** 2 / (gamma(1 / self.alpha) * gamma(3 / self.alpha))


class GeneralizedNormalRedefined(rv_continuous):
    def __init__(self, alpha):
        super(GeneralizedNormalRedefined, self).__init__()
        self.alpha = alpha
        self.beta = (gamma(3 / alpha) / gamma(1 / alpha)) ** (alpha / 2)  # ensuring E[X]^2 = 1
        self.Z = 0.5 * alpha * gamma(3 / alpha) ** 0.5 / gamma(1 / alpha) ** 1.5  # ensuring proba distribution

    def _pdf(self, x):
        return self.Z * np.exp(-self.beta * x ** self.alpha)


class GeneralizedRayleigh(rv_continuous):
    def __init__(self, alpha):
        if isinstance(alpha, np.ndarray):
            alpha = alpha.item()
        if alpha < 0.2:
            print("WARNING: normalization factor Z is high")
        super(GeneralizedRayleigh, self).__init__(a=0.0)
        self.alpha = alpha
        self.beta = (gamma(4 / alpha) / gamma(2 / alpha)) ** (alpha / 2)  # ensuring E[X]^2 = 1
        self.Z = alpha * gamma(4 / alpha) / gamma(2 / alpha) ** 2  # ensuring proba distribution

    def _pdf(self, x):
        return (1.0 * (x > 0)) * self.Z * x * np.exp(-self.beta * x ** self.alpha)

    def sparsity(self):
        return (self.beta ** (-1 / self.alpha) * gamma(3 / self.alpha) / gamma(2 / self.alpha)) ** 2


class SkewedGT(rv_continuous):  # from Theodossiou 1998
    def __init__(self, k, n, lam):
        super(SkewedGT, self).__init__(a=-5.0, b=5.0)
        self.k, self.n, self.lam = k, n, lam
        slam = (1 + 3 * lam ** 2 - 4 * lam ** 2 * beta(2 / k, (n - 1) / k) ** 2 * beta(1 / k, n / k) ** (-1.0) * beta(
            3 / k, (n - 2) / k) ** (-1.0)) ** 0.5
        self.C = 0.5 * k * beta(1 / k, n / k) ** (-1.5) * beta(3 / k, (n - 2) / k) ** 0.5 * slam
        self.theta = (k / (n - 2)) ** (1 / k) * beta(1 / k, n / k) ** 0.5 * beta(3 / k, (n - 2) / k) ** (
            -0.5) * slam ** (-1)

    def _pdf(self, x):
        signx = 1.0 * (x >= 0.0) - 1.0 * (x < 0)
        return self.C * (1 + (self.k / (self.n - 2)) * self.theta ** (-self.k) * (1 + signx * self.lam) ** (
            -self.k) * np.abs(x) ** self.k) \
               ** (-(self.n + 1) / self.k)