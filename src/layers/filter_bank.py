""" Implements Morlet, Battle-Lemarie, Bump steerable and Meyer wavelets used in convolution layers. """
import numpy as np
from scipy.fftpack import ifft


###############################################
# Morlet Wavelets
###############################################


def morlet_1d(N, xi, sigma, normalize='l1', P_max=5, eps=1e-7):
    """ Computes the Fourier transform of a Morlet filter.

    A Morlet filter is the sum of a Gabor filter and a low-pass filter
    to ensure that the sum has exactly zero mean in the temporal domain.
    It is defined by the following formula in time:
    psi(t) = g_{sigma}(t) (e^{i xi t} - beta)
    where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
    the cancelling parameter.

    Parameters
    ----------
    N : int
        size of the temporal support
    xi : float
        central frequency (in [0, 1])
    sigma : float
        bandwidth parameter
    normalize : string, optional
        normalization types for the filters. Defaults to 'l1'.
        Supported normalizations are 'l1' and 'l2' (understood in time domain).
    P_max: int, optional
        integer controlling the maximal number of periods to use to ensure
        the periodicity of the Fourier transform. (At most 2*P_max - 1 periods
        are used, to ensure an equal distribution around 0.5). Defaults to 5
        Should be >= 1
    eps : float
        required machine precision (to choose the adequate P)

    Returns
    -------
    morlet_f : array_like
        numpy array of size (N,) containing the Fourier transform of the Morlet
        filter at the frequencies given by np.fft.fftfreq(N).
    """
    if type(P_max) != int:
        raise ValueError('P_max should be an int, got {}'.format(type(P_max)))
    if P_max < 1:
        raise ValueError('P_max should be non-negative, got {}'.format(P_max))
    # Find the adequate value of P
    P = min(adaptive_choice_P(sigma, eps=eps), P_max)
    assert P >= 1
    # Define the frequencies over [1-P, P[
    freqs = np.arange((1 - P) * N, P * N, dtype=float) / float(N)
    if P == 1:
        # in this case, make sure that there is continuity around 0
        # by using the interval [-0.5, 0.5]
        freqs_low = np.fft.fftfreq(N)
    elif P > 1:
        freqs_low = freqs
    else:
        raise ValueError("Invalid P value in morlet_1d.")
    # define the gabor at freq xi and the low-pass, both of width sigma
    gabor_f = np.exp(-(freqs - xi)**2 / (2 * sigma**2))
    low_pass_f = np.exp(-(freqs_low**2) / (2 * sigma**2))
    # discretize in signal <=> periodize in Fourier
    gabor_f = periodize_filter_fourier(gabor_f, nperiods=2 * P - 1)
    low_pass_f = periodize_filter_fourier(low_pass_f, nperiods=2 * P - 1)
    # find the summation factor to ensure that morlet_f[0] = 0.
    kappa = gabor_f[0] / low_pass_f[0]
    morlet_f = gabor_f - kappa * low_pass_f
    # normalize the Morlet if necessary
    morlet_f *= get_normalizing_factor(morlet_f, normalize=normalize)
    return morlet_f


def adaptive_choice_P(sigma, eps=1e-7):
    """ Adaptive choice of the value of the number of periods in the frequency
    domain used to compute the Fourier transform of a Morlet wavelet.

    This function considers a Morlet wavelet defined as the sum
    of
    * a Gabor term hat psi(omega) = hat g_{sigma}(omega - xi)
    where 0 < xi < 1 is some frequency and g_{sigma} is
    the Gaussian window defined in Fourier by
    hat g_{sigma}(omega) = e^{-omega^2/(2 sigma^2)}
    * a low pass term \\hat \\phi which is proportional to \\hat g_{\\sigma}.

    If \\sigma is too large, then these formula will lead to discontinuities
    in the frequency interval [0, 1] (which is the interval used by numpy.fft).
    We therefore choose a larger integer P >= 1 such that at the boundaries
    of the Fourier transform of both filters on the interval [1-P, P], the
    magnitude of the entries is below the required machine precision.
    Mathematically, this means we would need P to satisfy the relations:

    |\\hat \\psi(P)| <= eps and |\\hat \\phi(1-P)| <= eps

    Since 0 <= xi <= 1, the latter implies the former. Hence the formula which
    is easily derived using the explicit formula for g_{\\sigma} in Fourier.

    Parameters
    ----------
    sigma: float
        Positive number controlling the bandwidth of the filters
    eps : float, optional
        Positive number containing required precision. Defaults to 1e-7

    Returns
    -------
    P : int
        integer controlling the number of periods used to ensure the
        periodicity of the final Morlet filter in the frequency interval
        [0, 1[. The value of P will lead to the use of the frequency
        interval [1-P, P[, so that there are 2*P - 1 periods.
    """
    val = np.sqrt(-2 * (sigma**2) * np.log(eps))
    P = int(np.ceil(val + 1))
    return P


def periodize_filter_fourier(h_f, nperiods=1):
    """ Computes a periodization of a filter provided in the Fourier domain.

    Parameters
    ----------
    h_f : array_like
        complex numpy array of shape (N*n_periods,)
    nperiods: int, optional
        Number of periods which should be used to periodize

    Returns
    -------
    v_f : array_like
        complex numpy array of size (N,), which is a periodization of
        h_f as described in the formula:
        v_f[k] = sum_{i=0}^{n_periods - 1} h_f[i * N + k]
    """
    N = h_f.shape[0] // nperiods
    v_f = h_f.reshape(nperiods, N).mean(axis=0)
    return v_f


def get_normalizing_factor(h_f, normalize='l1'):
    """ Computes the desired normalization factor for a filter defined in Fourier.

    Parameters
    ----------
    h_f : array_like
        numpy vector containing the Fourier transform of a filter
    normalize : string, optional
        desired normalization type, either 'l1' or 'l2'. Defaults to 'l1'.

    Returns
    -------
    norm_factor : float
        such that h_f * norm_factor is the adequately normalized vector.
    """
    h_real = ifft(h_f)
    if np.abs(h_real).sum() < 1e-7:
        raise ValueError('Zero division error is very likely to occur, ' +
                         'aborting computations now.')
    if normalize == 'l1':
        norm_factor = 1. / (np.abs(h_real).sum())
    elif normalize == 'l2':
        norm_factor = 1. / np.sqrt((np.abs(h_real)**2).sum())
    else:
        raise ValueError("Supported normalizations only include 'l1' and 'l2'")
    return norm_factor


def gauss_1d(N, sigma, normalize='l1', P_max=5, eps=1e-7):
    """ Computes the Fourier transform of a low pass gaussian window.

    \\hat g_{\\sigma}(\\omega) = e^{-\\omega^2 / 2 \\sigma^2}

    Parameters
    ----------
    N : int
        size of the temporal support
    sigma : float
        bandwidth parameter
    normalize : string, optional
        normalization types for the filters. Defaults to 'l1'
        Supported normalizations are 'l1' and 'l2' (understood in time domain).
    P_max : int, optional
        integer controlling the maximal number of periods to use to ensure
        the periodicity of the Fourier transform. (At most 2*P_max - 1 periods
        are used, to ensure an equal distribution around 0.5). Defaults to 5
        Should be >= 1
    eps : float, optional
        required machine precision (to choose the adequate P)

    Returns
    -------
    g_f : array_like
        numpy array of size (N,) containing the Fourier transform of the
        filter (with the frequencies in the np.fft.fftfreq convention).
    """
    # Find the adequate value of P
    if type(P_max) != int:
        raise ValueError('P_max should be an int, got {}'.format(type(P_max)))
    if P_max < 1:
        raise ValueError('P_max should be non-negative, got {}'.format(P_max))
    P = min(adaptive_choice_P(sigma, eps=eps), P_max)
    assert P >= 1
    # switch cases
    if P == 1:
        freqs_low = np.fft.fftfreq(N)
    elif P > 1:
        freqs_low = np.arange((1 - P) * N, P * N, dtype=float) / float(N)
    else:
        raise ValueError("Invalid P value in gauss_1d.")
    # define the low pass
    g_f = np.exp(-freqs_low**2 / (2 * sigma**2))
    # periodize it
    g_f = periodize_filter_fourier(g_f, nperiods=2 * P - 1)
    # normalize the signal
    g_f *= get_normalizing_factor(g_f, normalize=normalize)
    # return the Fourier transform
    return g_f


def compute_sigma_psi(xi, Q, r=np.sqrt(0.5)):
    """ Computes the frequential width sigma for a Morlet filter of frequency xi
    belonging to a family with Q wavelets.

    The frequential width is adapted so that the intersection of the
    frequency responses of the next filter occurs at a r-bandwidth specified
    by r, to ensure a correct coverage of the whole frequency axis.
    """
    factor = 1. / (2. ** (1. / Q))
    term1 = (1 - factor) / (1 + factor)
    term2 = 1. / np.sqrt(2 * np.log(1. / r))
    return xi * term1 * term2


def compute_morlet_parameters(J, Q, max_frequency):
    """ Compute central frequencies and bandwidth of morlet dictionary. """
    scaling = np.logspace(0, -J, num=J * Q, endpoint=False, base=2)
    mu_freq = max_frequency * scaling
    sigma_freq = np.array([compute_sigma_psi(m, Q) for m in mu_freq])
    return mu_freq, sigma_freq


def compute_morlet_low_pass_parameters(J, Q, max_frequency):
    """ Compute central frequency and bandwidth of low-pass filter.
    This function uses the oracle sigma_{low} = sigma_J * (2.372 Q + 1.109) where sigma_{low}
    is the bandwidth of the low pass filter, sigma_J is the bandwidth of a morlet wavelet centered in max_frequency
    2^{-J} and :math:Q is the number of wavelet per scales. This function was empiricaly found to minimise the
    dictionary's Littlewood-Paley inequality constant.
    """
    mu_J = max_frequency * 2 ** -J
    sigma_J = compute_sigma_psi(mu_J, Q)
    sigma_low = sigma_J * (2.372 * Q + 1.109)
    return sigma_low


###############################################
# Battle-Lemarie Wavelets
###############################################

# BL_XI0 = 0.7593990773014584
BL_XI0 = 0.75 * 1.012470304985129


def compute_battle_lemarie_parameters(J, Q, high_freq=0.5):
    factor = 1. / 2. ** (1 / Q)
    xi, sigma = [high_freq * factor ** j for j in range(J * Q)], []
    return xi, sigma


def b_function(freqs):
    cos2 = np.cos(freqs * np.pi) ** 2
    sin2 = np.sin(freqs * np.pi) ** 2

    num = 5 + 30 * cos2 + 30 * sin2 * cos2 + 70 * cos2 ** 2 + 2 * sin2 ** 2 * cos2 + 2 * sin2 ** 3 / 3
    num /= 105 * 2 ** 8
    sin8 = sin2 ** 4

    return num, sin8


def battle_lemarie_psi(T, Q, xi, normalize):
    xi0 = BL_XI0  # mother wavelet center

    # frequencies for mother wavelet with 1 wavelet per octave
    abs_freqs = np.linspace(0, 1, T + 1)[:-1]
    # frequencies for wavelet centered in xi with 1 wavelet1 per octave
    freqs = abs_freqs * xi0 / xi
    # frequencies for wavelet centered in xi with Q wavelets per octave
    # freqs = xi0 + (xi_freqs - xi0) * Q

    num, den = b_function(freqs)
    num2, den2 = b_function(freqs / 2)
    numpi, denpi = b_function(freqs / 2 + 0.5)

    stable_den = np.empty_like(freqs)
    stable_den[freqs != 0] = np.sqrt(den[freqs != 0]) / (2 * np.pi * freqs[freqs != 0]) ** 4
    # protection in omega = 0
    stable_den[freqs == 0] = 2 ** (-4)

    mask = np.mod(freqs, 2) != 1
    stable_den[mask] *= np.sqrt(den2[mask] / denpi[mask])
    mask = np.mod(freqs, 2) == 1
    # protection in omega = 2pi [4pi]
    stable_den[mask] = np.sqrt(den2[mask]) / (np.pi * freqs[mask]) ** 4

    psi_hat = np.sqrt(numpi / (num * num2)) * stable_den
    psi_hat[freqs < 0] = 0

    # remove small bumps after the main bumps in order to improve the scaling of high frequency wavelets
    idx = 0
    while True:
        idx += 1
        if idx == psi_hat.size - 1 or (psi_hat[idx - 1] > psi_hat[idx] and psi_hat[idx] < psi_hat[idx + 1]):
            break

    psi_hat[idx + 1:] = 0.0

    if normalize == 'l1':
        pass
    if normalize == 'l2':
        psi_hat /= np.sqrt((np.abs(psi_hat) ** 2).sum())

    return psi_hat


def battle_lemarie_phi(N, Q, xi_min):
    if Q != 1:
        raise NotImplementedError("Scaling BL wavelets to multiple wavelets per octave not implemented yet.")

    xi0 = BL_XI0  # mother wavelet center

    abs_freqs = np.fft.fftfreq(N)
    freqs = abs_freqs * xi0 / xi_min
    # freqs = xi_freqs * Q

    num, den = b_function(freqs)

    stable_den = np.empty_like(freqs)
    stable_den[freqs != 0] = np.sqrt(den[freqs != 0]) / (2 * np.pi * freqs[freqs != 0]) ** 4
    stable_den[freqs == 0] = 2 ** (-4)

    phi_hat = stable_den / np.sqrt(num)
    return phi_hat


###############################################
# Bump steerable Wavelets
###############################################


def compute_bump_steerable_parameters(J, Q, high_freq=0.5):
    return compute_battle_lemarie_parameters(J, Q, high_freq=high_freq)


def low_pass_constants(Q):
    """ Function computing the ideal amplitude and variance for the low-pass of a bump
    wavelet dictionary, given the number of wavelets per scale Q.
    The amplitude and variance are computed by minimizing the frame error eta:
        1 - eta <= sum psi_la ** 2 <= 1 + eta
    Simple models are then fitted to compute those values quickly.
    The computation was done using gamma = 1.
    """
    ampl = -0.04809858889110362 + 1.3371665071917382 * np.sqrt(Q)
    xi2sigma = np.exp(-0.35365794431968484 - 0.3808886546835562 / Q)
    return ampl, xi2sigma


def hwin(freqs, gamma1):
    psi_hat = np.zeros_like(freqs)
    idx = np.abs(freqs) < gamma1

    psi_hat[idx] = np.exp(1. / (freqs[idx] ** 2 - gamma1 ** 2))
    psi_hat *= np.exp(1 / gamma1 ** 2)

    return psi_hat


def bump_steerable_psi(N, xi):
    abs_freqs = np.linspace(0, 1, N + 1)[:-1]
    psi = hwin((abs_freqs - xi) / xi, 1.)

    return psi


def bump_steerable_phi(N, Q, xi_min):
    ampl, xi2sigma = low_pass_constants(Q)
    sigma = xi_min * xi2sigma

    abs_freqs = np.abs(np.fft.fftfreq(N))
    phi = ampl * np.exp(- (abs_freqs / (2 * sigma)) ** 2)

    return phi


###############################################
# Meyer steerable Wavelets
###############################################


def compute_meyer_parameters(J, Q, high_freq):
    return compute_battle_lemarie_parameters(J, Q, high_freq=high_freq)


def compute_shannon_parameters(J, Q, high_freq):  # bad code
    xi = [high_freq * 2 ** (-j / Q) for j in range(J * Q)]
    sigma = [om * (2 ** (1 / Q) - 1) / (2 ** (1 / Q) + 1) for om in xi]
    return xi, sigma


def shannon_psi(N, xi, sigma):
    freqs = np.linspace(0.0, 1.0, N, endpoint=True)
    psi = 1.0 * ((freqs < xi + sigma) & (xi - sigma <= freqs))
    return psi


def shannon_phi(N, sigma):
    freqs = np.linspace(0.0, 1.0, N, endpoint=True)
    phi = 1.0 * ((freqs <= sigma) | (1.0 - freqs <= sigma))
    return phi


def meyer_psi(N, Q, xi):
    if Q != 1:
        raise NotImplementedError("Scaling Meyer wavelets to multiple wavelets per octave not implemented yet.")

    # frequencies for mother wavelet with 1 wavelet per octave
    abs_freqs = np.linspace(-0.5, 0.5, N + 1)[:-1]
    psi = meyer_mother_psi(8 / 3 * np.pi * abs_freqs / xi)
    return np.fft.fftshift(psi)


def meyer_phi(N, xi):
    abs_freqs = np.linspace(-0.5, 0.5, N + 1)[:-1]
    phi = meyer_mother_phi(8 / 3 * np.pi * abs_freqs / xi)
    return np.fft.fftshift(phi)


def nu(x):
    out = np.zeros(x.shape)
    idx = np.logical_and(0 < x, x < 1)
    out[idx] = x[idx] ** 4 * (35 - 84 * x[idx] + 70 * x[idx] ** 2 - 20 * x[idx] ** 3)
    return out


def meyer_mother_psi(w):
    psi = np.zeros(w.shape) + 1j * np.zeros(w.shape)
    idx = np.logical_and(2 * np.pi / 3 < w, w < 4 * np.pi / 3)
    psi[idx] = np.sin(np.pi / 2 * nu(3 * np.abs(w[idx]) / 2 / np.pi - 1)) / np.sqrt(2 * np.pi)  # * np.exp(1j*w[idx]/2)

    idx = np.logical_and(4 * np.pi / 3 < w, w < 8 * np.pi / 3)
    psi[idx] = np.cos(np.pi / 2 * nu(3 * np.abs(w[idx]) / 4 / np.pi - 1)) / np.sqrt(2 * np.pi)  # * np.exp(1j*w[idx]/2)

    return 2 * psi


def meyer_mother_phi(w):
    phi = np.zeros(w.shape) + 1j * np.zeros(w.shape)
    idx = np.abs(w) < 2 * np.pi / 3
    phi[idx] = 1 / np.sqrt(2 * np.pi)
    idx = np.logical_and(2 * np.pi / 3 < np.abs(w), np.abs(w) < 4 * np.pi / 3)
    phi[idx] = np.cos(np.pi / 2 * nu(3 * np.abs(w[idx]) / 2 / np.pi - 1)) / np.sqrt(2 * np.pi)
    return phi * 2


###############################################
# initialize wavelets
###############################################


def init_wavelet_param(wav_type, J, Q, high_freq):
    """ Init central frequencies and frequency width of the band-pass filters. """
    if wav_type == 'morlet':
        xi, sigma = compute_morlet_parameters(J, Q, high_freq)
    elif wav_type == 'battle_lemarie':
        xi, sigma = compute_battle_lemarie_parameters(J, Q, high_freq)
    elif wav_type == 'bump_steerable':
        if Q != 1:
            print("\nWarning: width of Bump-Steerable wavelets not adaptative with Q.\n")
        xi, sigma = compute_bump_steerable_parameters(J, Q, high_freq)
    elif wav_type == 'meyer':
        # if Q != 1:
        #    print("\nWarning: width of Meyer wavelets not adaptative with Q in the current implementation.\n")
        xi, sigma = compute_meyer_parameters(J, Q, high_freq)
    elif wav_type == 'shannon':
        xi, sigma = compute_shannon_parameters(J, Q, high_freq)
    else:
        raise ValueError("Unkown wavelet type: {}".format(wav_type))

    return np.array(xi), np.array(sigma)


def init_band_pass(wav_type, T, J, Q, high_freq, wav_norm):
    """ Compute the band-pass Fourier transforms. """
    xis, sigmas = init_wavelet_param(wav_type, J, Q, high_freq)

    if wav_type == "morlet":
        psi_hat = [morlet_1d(T, xi, sigma, wav_norm) for xi, sigma in zip(xis, sigmas)]
    elif wav_type == "battle_lemarie":
        psi_hat = [battle_lemarie_psi(T, Q, xi, wav_norm) for xi in xis]
    elif wav_type == "bump_steerable":
        psi_hat = [bump_steerable_psi(T, xi) / np.sqrt(Q) for xi in xis]
    elif wav_type == 'meyer':
        psi_hat = [meyer_psi(T, 1, xi) for xi in xis]
    elif wav_type == 'shannon':
        psi_hat = [shannon_psi(T, xi, sigma) for xi, sigma in zip(xis, sigmas)]
    else:
        raise ValueError("Unkown wavelet type: {}".format(wav_type))
    psi_hat = np.stack(psi_hat, axis=0)

    # some high frequency wavelets have strange behavior at negative low frequencies
    psi_hat[:, -T // 8:] = 0.0

    return psi_hat


def init_low_pass(wav_type, T, J, Q, high_freq):
    """ Compute the low-pass Fourier transforms assuming it has the same variance
    as the lowest-frequency wavelet. """
    xis, sigmas = init_wavelet_param(wav_type, J, Q, high_freq)
    xis = np.append(xis, 0.0)

    if wav_type == "morlet":
        sigma_low = sigmas[-1]
        np.append(sigmas, compute_morlet_low_pass_parameters(J, Q, high_freq))
        phi_hat = gauss_1d(T, sigma_low)
    elif wav_type == "battle_lemarie":
        xi_low = xis[-2]  # Because 0 was appended for Morlet
        phi_hat = battle_lemarie_phi(T, 1, xi_low)
    elif wav_type == "bump_steerable":
        xi_low = xis[-2]  # Because 0 was appended for Morlet
        phi_hat = bump_steerable_phi(T, 1, xi_low)
    elif wav_type == 'meyer':
        xi_low = xis[-2]
        phi_hat = meyer_phi(T, xi_low)
    elif wav_type == 'shannon':
        sigma_low = xis[-2] - sigmas[-1]
        phi_hat = shannon_phi(T, sigma_low)
    else:
        raise ValueError("Unkown wavelet type: {}".format(wav_type))

    return phi_hat
