""" Gaussian models. """
import warnings
import numpy as np
from numpy.random import normal as nd
from scipy.linalg import sqrtm as sq

from src.utils import standardize


def gaussian_cme(cov, R, T):
    """ Create S synthesis of a gaussian process of length T with the specified
    autocovariance through circulant matrix embedding (see C. R. Dietrich AND G. N. Newsam).

    :param cov: T array, function r such that Cov[Y(x)Y(y)] = r(|x-y|)
    :param: S: int, number of synthesis
    :param T: int, number of samples
    """

    # Circulant matrix embedding: fft of periodized autocovariance:
    cov = np.concatenate((cov, np.flip(cov[1:-1])), axis=0)
    L = np.fft.fft(cov)[None, :]
    if np.any(L.real < 0):
        warnings.warn('Found FFT of covariance < 0. Embedding matrix is not non-negative definite.')

    # Random noise in Fourier domain
    z = np.random.randn(R, 2 * T - 2) + 1j * np.random.randn(R, 2 * T - 2)

    # Impose covariance and invert
    # Use fft to ignore normalization, because only real part is needed.
    x = np.fft.fft(z * np.sqrt(L / (2 * T - 2)), axis=-1).real

    # First N samples have autocovariance cov:
    x = x[:, :T]

    return x


def get_base_gaussian(shape, mean_x, mean_dx, std_dx, fixed_range, x_target):
    """ Obtains B(t) a brownian calibrated on x.

    :param shape: N x T
    :return: N x T array
    """

    if fixed_range is not None:
        T = x_target.shape[-1]
        dx_target = np.diff(x_target, axis=-1)
        fixed = fixed_range[:-1]
        free = np.array([i for i in range(T - 1) if i not in fixed])

        # complete with present values
        dx = np.random.randn(1, T - 1)
        dx[:, fixed] = dx_target[:, fixed]
        dx[:, free] = standardize(dx[:, free], axis=-1)

        mean_dx_free = (mean_dx * (T - 1) - dx[:, fixed].sum(-1)) / free.size
        std_dx_free = (((T - 1) * std_dx ** 2 -
                        fixed.size * np.std(dx[:, fixed], axis=-1)[0] ** 2) / free.size) ** 0.5

        dx[:, free] *= std_dx_free
        dx[:, free] += mean_dx_free

        if fixed[0] > 0:
            dx = np.concatenate([dx[:, fixed], dx[:, free]], axis=-1)

        x = np.concatenate([np.zeros_like(dx[:, :1]), np.cumsum(dx, axis=-1)], axis=-1)
        x += x_target[:, fixed[0]] - x[:, 0]

        return x
    else:
        white_noise = np.random.normal(loc=0.0, scale=1.0, size=(shape[0], shape[1] - 1))
        white_noise -= white_noise.mean(-1, keepdims=True)
        white_noise /= (white_noise ** 2).mean(-1, keepdims=True) ** 0.5

        white_noise *= std_dx
        white_noise += mean_dx

        brownian = np.concatenate([np.zeros_like(white_noise[:, :1]), np.cumsum(white_noise, axis=-1)], axis=-1)
        brownian -= brownian.mean(-1, keepdims=True)

        return brownian + mean_x


def simulateBrownians(N, T, mean=None, cov=None):
    # Needs to take into account the week ends ? meaning, the time between two days ?
    # Useless step ? since the power spectrum should rescale it ?
    if mean is None:
        mean = np.zeros((N,))
    if cov is None:
        cov = np.identity(N)

    increments = np.random.multivariate_normal(mean, cov, size=(T,)).T
    increments /= 252

    return increments


def powerSpectrumNormalization(x, K):
    """ A function that enables the TS x to match the power spectrum K by convolution

    :param x: The vector to normalize
    :param K: The targeted power spectrum
    :return: The vector x\circ kappa with same spectrum as K
    """
    Kfft = np.fft.fft(K)
    xfft = np.fft.fft(x)

    discrepancy = np.divide(np.abs(Kfft), np.abs(xfft))
    x_normalized_fft = np.multiply(xfft, discrepancy)

    return np.fft.ifft(x_normalized_fft).real


def fbm(R, T, H, sigma=1, dt=None):
    """ Create a realization of fractional Brownian motion using circulant
    matrix embedding.

    Inputs:
      - shape: if scalar, it is the  number of samples. If tuple it is (N, R), the
               number of samples and realizations, respectively.
      - H (scalar): Hurst exponent.
      - sigma (scalar): variance of processr

    Outputs:
      - fbm: synthesized fbm realizations. If 'shape' is scalar, fbm is of shape (N,).
             Otherwise, it is of shape (N, R).
    """
    if not 0 <= H <= 1:
        raise ValueError('H must satisfy 0 <= H <=1')

    if not dt:
        dt = 1 / T

    # Create covariance of fGn
    n = np.arange(T)
    r = dt ** (2 * H) * sigma ** 2 / 2 * (
                np.abs(n + 1) ** (2 * H) + np.abs(n - 1) ** (2 * H) - 2 * np.abs(n) ** (2 * H))

    fbm = np.cumsum(gaussian_cme(r, R, T), axis=1)

    return fbm


def geom_brownian(R: int, T: float, nb_sample: int, S0: float, mu: float, sigma: float):
    """ Simulate a geometric brownian also called Black-Scholes trajectory of trend mu and vol sigma. """
    B = np.cumsum(np.random.randn(R, nb_sample), -1) / np.sqrt(nb_sample)
    x = S0 * np.exp((mu - 0.5 * sigma ** 2) * np.linspace(0, T, nb_sample) + sigma * np.sqrt(T) * (B - B[:, 0:1]))

    return x
