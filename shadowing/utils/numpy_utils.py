import numpy as np
from scipy.signal import convolve as sci_convolve


def windows(x, w, s, offset=0, cover_end=False):
    """ Extracts windows of size w with stride s from x. Discard any residual

    :param x: array of shape (..., T)
    :param w: the window size
    :param s: the stride
    :param offset: the starting offset
    :param cover_end: if True, adjust the offset so that the last window covers the end of x
    :return: array of shape (..., (T - w) // s + 1, w)
    """
    if offset > 0 and cover_end:
        raise ValueError("No offset should be provided if cover_end is True.")
    if offset > 0:
        return windows(x[...,offset:], w, s, 0, cover_end)
    if cover_end:
        offset = x.shape[-1] % w
        return windows(x, w, s, offset, cover_end=False)
    nrows = 1 + (x.shape[-1] - w) // s
    n = x.strides[-1]
    return np.lib.stride_tricks.as_strided(x, shape=x.shape[:-1]+(nrows,w), strides=x.strides[:-1]+(s*n,n))


def shifted_product_aux(x, y):
    """
    Computes E[x(t-tau)y(t)] for values of tau between -T+1 and T-1

    :param x: (B) x T array
    :param y: (B) x T array
    :return: tuple of two arrays
        - first array : tau = 0 to tau = T-1 (x in advance on y)
        - snd array : tau = 0 to tau = -T+1 (y in advance on x)
    """
    assert x.shape == y.shape

    # perform convolution
    y = np.flip(y, axis=-1)
    shape = x.shape[:-1]
    x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
    conv = np.stack([sci_convolve(x1d, y1d, method='fft') for (x1d, y1d) in zip(x, y)], axis=0)  # batched

    # normalization factors to get averages
    T = x.shape[-1]
    norms = np.array([T - np.abs(tau) for tau in range(-T + 1, T)])

    # the result for every tau
    sft_prod = conv / norms[None, :]
    sft_prod = sft_prod.reshape(shape + (sft_prod.shape[-1],))

    return np.flip(sft_prod[..., :T], axis=-1), sft_prod[..., T - 1:]


def apply_exp_kernel(I, beta):
    """

    :param I: N x T array
    :return: N x T array
    """
    if beta is None:
        return I

    I_shape = I.shape
    I = I.reshape((-1, I_shape[-1]))

    #TODO: make sure the convolution is causal
    print("WARNING. Is the convolution causal?")

    kernel = np.exp(-np.array([beta * s for s in range(I.shape[-1])]))
    Tb = int(np.log(10) / beta)
    if Tb > 1e6:
        raise ValueError("beta is too small, padding would add too many zeros")
    kernel[Tb + 1:] = 0.0
    kernel /= kernel.sum()

    # padding
    kernel = np.pad(kernel, (0, Tb + 1), 'constant')
    I = np.pad(I, ((0, 0), (Tb + 1, 0)), 'constant')

    # convolution
    x_hat = np.fft.fft(I)
    kernel_hat = np.fft.fft(kernel[None, :])
    x_avg = np.fft.ifft(x_hat * kernel_hat).real
    return x_avg[:, Tb + 1:].reshape(I_shape)


def shifted_product(x, y, beta):
    """
    Computes E[\tilde{x}(t-tau)y(t)] where \tilde{x} is an exponentially average of x in the past if tau => 0,
    in the future otherwise.

    :param x: (B) x T array
    :param y: (B) x T array
    :param beta: the exponential average parameter
    :return:
    """
    x_pos = apply_exp_kernel(x, beta)  # x is smoothed in the past
    x_neg = np.flip(apply_exp_kernel(np.flip(x, axis=-1), beta), axis=-1)  # x is smoothed in the future

    pos, _ = shifted_product_aux(x_pos, y)
    _, neg = shifted_product_aux(x_neg, y)

    tau_pos = np.arange(x.shape[-1])

    return tau_pos, -tau_pos, pos, neg


def time_corr_increments(x, y, w, s, offset, taumax, rmm, beta):
    if y is None:
        y = x

    # windows in order to obtain increments periods of length w
    xw = windows(x, w, s, offset).sum(-1)  # (C) x nb_w
    yw = windows(y, w, s, offset).sum(-1)  # (C) x nb_w

    # remove mean
    if rmm:
        xw -= xw.mean(-1, keepdims=True)

    # compute correlations
    tau_pos, tau_neg, pos, neg = shifted_product(xw, yw, beta)
    taumax = min(pos.shape[-1], taumax)

    return tau_pos[:taumax], tau_neg[:taumax], pos[..., :taumax], neg[..., :taumax]


def leverage(x, w, s, offset, p, taumax, beta):
    """
    Computes E[(x(t-tau) - E[x(t-tau)]) |x(t)|^p]

    :param x: a (C) x T array
    :param w: the averaging window
    :param p: the power
    :param taumax: the power
    :param beta: smoothing parameter
    :return:
    """
    # windows
    size = (x.shape[-1] - w - offset) // s + 1
    Xw = np.stack([x[..., k * s + offset: w + k * s + offset] for k in range(size)], axis=-2)  # (C) x nb_window x w
    Xw = Xw.sum(-1)  # (C) x nb_window
    Xw -= Xw.mean(axis=-1, keepdims=True)

    _, _, pos, neg = shifted_product(x=Xw, y=np.abs(Xw)**p, beta=beta)
    return np.concatenate([np.flip(neg[..., 1:taumax], axis=-1), pos[..., :taumax]], axis=-1)


def windowed_mean(x, w, s, offset=0):
    """

    :param x: numpy array (B) x T
    :param w: window
    :param s: stride
    :return:
    """
    cms = np.cumsum(x, axis=-1)
    cms = np.concatenate([np.zeros(x.shape[:-1] + (1,)), cms], axis=-1)
    size = (x.shape[-1] - w - offset) // s + 1
    res = np.zeros_like(x, shape=x.shape[:-1] + (size,))
    for k in range(size):
        res[..., k] = cms[..., w + k * s + offset] - cms[..., k * s + offset]
    return res / w


def r2_score(y_true, y_pred, axis=-1):
    """ Identical to scipy r2_score but specifying axis. """
    def MSE(a, b):
        return ((a - b) ** 2).mean(axis)
    return 1 - MSE(y_true, y_pred) / MSE(y_true, y_true.mean(axis, keepdims=True))


def array_equal(x, y, precision=1e-6):
    if x.shape != y.shape:
        return False
    return np.abs(x - y).mean() / np.abs(x).mean() < precision
