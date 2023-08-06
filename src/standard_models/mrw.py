""" Multifrctal random walks. """
import numpy as np
from numpy.fft import fft, ifft

from src.standard_models import gaussian_cme, fbm


def gaussian_w(R, T, L, lam, dt=1):
    """ Auxiliar function to create gaussian process w. """
    kmax = int(np.floor(L / dt - 1))
    k = np.arange(kmax)
    rho = np.ones(T)
    rho[:kmax] = L / (k + 1) / dt
    cov = (lam ** 2) * np.log(rho)
    w = gaussian_cme(cov, R, T)
    return w


def mrw(R, T, L, H, lam, sigma=1):
    """
    Create a realization of multifractal random walk

    WRONG : Create a realization of fractional Brownian motion using circulant
    matrix embedding.

    Inputs:
      - shape: if scalar, it is the  number of samples. If tuple it is (N, R), the
               number of samples and realizations, respectively.
      - H (scalar): Hurst exponent.
      - lambda (scalar): intermittency parameter.
      - L (scalar): integral scale.
      - sigma (scalar): variance of process.

    Outputs:
      - mrw: synthesized mrw realizations. If 'shape' is scalar, fbm is of shape (N,).
             Otherwise, it is of shape (N, R).

    References:
      - Bacry, Delour, Muzy, "Multifractal Random Walk", Physical Review E, 2001
    """
    if not 0 <= H <= 1:
        raise ValueError('H must satisfy 0 <= H <= 1')

    if L > T:
        raise ValueError('Integral scale L is larger than data length T')

    # 1) Gaussian process w
    w = gaussian_w(R, T, L, lam)

    # Adjust mean to ensure convergence of variance
    # r = 1 / 2  # see Bacry, Delour & Muzy, Phys Rev E, 2001, page 4
    r = 1.0  #
    w -= np.mean(w, axis=-1, keepdims=True) + r * lam ** 2 * np.log(L)

    # 2) fGn e
    fgn = np.diff(fbm(R, T + 1, H, sigma), axis=-1)

    # 3) mrw
    mrw = np.cumsum(fgn * np.exp(w), axis=-1)

    return mrw


def getBaseMRW(D, T, H, lam):
    x_mrw_list = []
    for d in range(D):
        x_mrw_list.append(mrw(1, T, H, lam, T))
    return np.transpose(np.concatenate(x_mrw_list, axis=1))


def skewed_mrw(R, T, L, H, lam, gamma, K0=1, alpha=1, sigma=1, dt=1, beta=1, do_mirror=False):
    """ Skewed mrw as in Pochart & Bouchaud. Assumes dt = 1, so no parameter beta is needed. """
    if not 0 <= H <= 1:
        raise ValueError('H must satisfy 0 <= H <= 1')

    if L / dt > T:
        raise ValueError('Integral scale L/dt is larger than data length N')

    # Tp = int(np.floor(T / dt - 1))

    # 1) Gaussian process w
    w = gaussian_w(R, T, L, lam, dt)

    # Adjust mean to ensure convergence of variance
    # r = 1 / 2  # see Bacry, Delour & Muzy, Phys Rev E, 2001, page 4
    r = 1.0  # see Bacry, Delour & Muzy, Phys Rev E, 2001, page 4
    w -= np.mean(w, axis=-1, keepdims=True) + r * lam ** 2 * np.log(L / dt)

    # 2) fGn e
    fgn = np.diff(fbm(R, 2 * T + 1, H, sigma, dt), axis=-1)

    # 3) Correlate components
    past = skewness_convolution(fgn, K0, alpha, gamma, beta, dt)
    wtilde = w - past

    # 4) skewed mrw
    smrw = np.cumsum(fgn[:, T:] * np.exp(wtilde), axis=-1)

    if do_mirror:
        past_mirror = skewness_convolution(-fgn, K0, alpha, gamma, beta, dt)
        wtilde_mirror = w - past_mirror
        smrw_mirror = np.cumsum(-fgn[:, T:] * np.exp(wtilde_mirror), axis=0)
        return smrw, smrw_mirror

    return smrw


def smrw_exp(R, T, L, H, lam, gamma, K0=1, alpha=1, sigma=1, dt=1, beta=1, do_mirror=False):
    """ Skewed mrw as in Pochart & Bouchaud. Assumes dt = 1, so no parameter beta is needed. """
    if not 0 <= H <= 1:
        raise ValueError('H must satisfy 0 <= H <= 1')

    if L / dt > T:
        raise ValueError('Integral scale L/dt is larger than data length T')

    # Tp = int(np.floor(T / dt - 1))

    # 1) Gaussian process w
    w = gaussian_w(R, T, L, lam, dt)

    # Adjust mean to ensure convergence of variance
    # r = 1 / 2  # see Bacry, Delour & Muzy, Phys Rev E, 2001, page 4
    r = 1.0  # see Bacry, Delour & Muzy, Phys Rev E, 2001, page 4
    w -= np.mean(w, axis=-1, keepdims=True) + r * lam ** 2 * np.log(L / dt)

    # 2) fGn e
    fgn = np.diff(fbm(R, 2 * T + 1, H, sigma, dt), axis=-1)

    # 3) Correlate components
    past = skewness_convolution(fgn, K0, alpha, gamma, beta, dt)
    wtilde = w - past

    # 4) skewed mrw
    smrw = np.cumsum(fgn[:, T:] * np.exp(wtilde), axis=-1)

    if do_mirror:
        past_mirror = skewness_convolution(-fgn, K0, alpha, gamma, beta, dt)
        wtilde_mirror = w - past_mirror
        smrw_mirror = np.cumsum(-fgn[:, T:] * np.exp(wtilde_mirror), axis=0)
        return smrw, smrw_mirror

    return smrw


def skewness_convolution(e, K0, alpha, gamma, beta=1, dt=1):
    """
    Noise e should be of length 2*N, with "N false past variables" at the beginning
    to avoid spurious correlations due to cutoffs in convolution.
    """
    S, T = e.shape
    T = T // 2

    tau = np.arange(1, T+1)
    Kbar = np.zeros((2*T))
    Kbar[1:T+1] = K0 * np.exp(-gamma * tau) / (tau**alpha) / (dt**beta)
    # Kbar[1:T+1] = K0 / (tau**alpha) / (dt**beta)
    skew_conv = np.real(ifft(fft(Kbar[None, :], axis=-1) * fft(e, axis=-1), axis=-1))
    return skew_conv[:, T:]


def skewness_conv_exp(e, K0, alpha):
    """
    Noise e should be of length 2*N, with "N false past variables" at the beginning
    to avoid spurious correlations due to cutoffs in convolution.
    """
    S, T = e.shape
    T = T // 2

    tau = np.arange(1, T+1)
    Kbar = np.zeros((2*T))
    # Kbar[1:T+1] = K0 * np.exp(-gamma * tau) / (tau**alpha) / (dt**beta)
    # Kbar[1:T+1] = K0 / (tau**alpha) / (dt**beta)
    Kbar[1:T+1] = K0 * np.exp(-alpha*tau)
    skew_conv = np.real(ifft(fft(Kbar[None, :], axis=-1) * fft(e, axis=-1), axis=-1))
    return skew_conv[:, T:]


def skewness_convolution_dumb(e, K0, alpha, beta=1, dt=1):
    '''
    Direct and inefficient calculation for testing purposes.
    Receives "true" input noise of size N.
    '''
    N, R = e.shape

    def K(i,j):
        return K0 / (j-i)**alpha / dt**beta

    scorr = np.zeros((N, R))
    for k in range(N):
        for i in range(k):
            scorr[k, :] += K(i, k) * e[i, :]
    return scorr
