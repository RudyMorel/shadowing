from time import time
from heapq import heapreplace
import numpy as np
import scipy
import torch

import src.utils.complex_utils as cplx


def from_values_to_absreturns(values):
    """

    :param values: (B) x N x T
    :return: (B) x N x (T-1)
    """
    return np.diff(values, axis=-1)


def from_absreturns_to_values(returns):
    """
    Returned array starts at 0.

    :param returns: (B) x N x (T-1)
    :return: (B) x N x T
    """
    values = np.cumsum(returns, axis=-1)
    init_values = np.zeros_like(returns[..., 0])[..., None]
    return np.concatenate([init_values, values], axis=-1)


def from_values_to_logreturns(values):
    """

    :param values: (B) x N x T
    :return: (B) x N x (T-1)
    """
    return np.diff(np.log(values), axis=-1)


def from_logreturns_to_values(returns):
    """
    Returned array starts at 1

    :param returns: (B) x N x (T-1)
    :return: (B) x N x T
    """
    values = np.cumprod(np.exp(returns), axis=-1)
    init_values = np.ones_like(returns[..., 0])[..., None]
    return np.concatenate([init_values, values], axis=-1)


def abs_translation(x, c):
    return x + c - x[..., 0][..., None]


def log_translation(x, c):
    return x * c / x[..., 0][..., None]


class TimeSeriesDerivation:  # could be extended to general F sucha that get_returns(x) = D[Fx]
    def __init__(self):
        # forward : x -> Dx, backward : Dx -> x
        self.admissible_types = ['abs', 'log']
        self.forward_dict = {'abs': from_values_to_absreturns, 'log': from_values_to_logreturns}
        self.backward_dict = {'abs': from_absreturns_to_values, 'log': from_logreturns_to_values}
        self.translations = {'abs': abs_translation, 'log': log_translation}

    def _checks(self, type):
        if not type in self.admissible_types:
            raise ValueError('TimeSeriesDerivation wrong type derivation')

    def get_returns(self, values, type='abs'):
        self._checks(type)
        return self.forward_dict[type](values)

    def get_values(self, returns, type='abs'):
        self._checks(type)
        return self.backward_dict[type](returns)

    def invariant_rescale(self, values, init_value, type='abs'):
        self._checks(type)
        return self.translations[type](x=values, c=init_value)


def rm_trend(x, output_trend=False):
    """
    Obtain the array y such that Dy = Dx + constant and E[Dy] = 0.

    :param x: (C) x T array
    :param output_trend: (C) array of the trends
    :return:
    """
    tendancy = x[..., -1] - x[..., 0]
    remover = np.linspace(np.zeros_like(tendancy), tendancy, x.shape[-1])
    remover = np.moveaxis(remover, 0, -1)
    remover -= remover.mean(axis=-1, keepdims=True)  # so that it does not change the mean of x
    if output_trend:
        return x - remover, remover
    return x - remover


def remove_tendancy(values, returnstype='abs', output_tendancy=False):
    """
    Obtain the series y(t) = x(t) - a t such that E[DFy] = 0, where F = id or ln, and y(0)=x(0).

    :param values:
    :param output_tendancy:
    :return:
    """
    deriv = TimeSeriesDerivation()

    # remove tendancy
    returns = deriv.get_returns(values=values, type=returnstype)
    tendancy = returns.mean(axis=-1, keepdims=True)
    values_corrected = deriv.get_values(returns=returns - tendancy,
                                        type=returnstype)  # values with zero abs or log derivative

    # right initialization
    values_corrected = deriv.invariant_rescale(values=values_corrected, init_value=values[..., 0][..., None],
                                               type=returnstype)

    return (tendancy, values_corrected) if output_tendancy else values_corrected


def add_tendancy(values, tendancy, returnstype='abs'):
    deriv = TimeSeriesDerivation()

    # add tendancy
    returns = deriv.get_returns(values=values, type=returnstype)
    values_corrected = deriv.get_values(returns + tendancy)

    # right initialization
    values_corrected = deriv.invariant_rescale(values=values_corrected, init_value=values[..., 0][..., None],
                                               type=returnstype)

    return values_corrected


def remove_analytic(x, mean_high, output_stat=False, backend='numpy'):
    N = x.shape[0]

    projector = mean_high.T @ mean_high
    projector_orthog = np.identity(N) - projector
    if backend == 'torch':
        projector = torch.FloatTensor(projector)
        projector_orthog = torch.FloatTensor(projector_orthog)
    projection = projector_orthog @ x

    if output_stat:
        return projection, projector @ x
    else:
        return projection


def remove_l2(x, axis, output_stat=False, backend='numpy'):
    if backend == 'torch':
        return remove_l2_torch(x, axis, output_stat)
    elif backend == 'numpy':
        return remove_l2_np(x, axis, output_stat)
    else:
        raise ValueError("Wrong backend")


def remove_l2_np(x, axis, output_stat=False):
    """
    Rescale by the l2 norm (defined as a mean)

    :param x: numpy array
    :param axis: the axis of normalization
    :param output_stat: whether to output the l2 norms
    :return:
    """
    if axis is None:
        return x
    l2_norm = ((x ** 2).mean(axis=axis, keepdims=True)) ** 0.5
    if output_stat:
        return np.divide(x, l2_norm), l2_norm
    else:
        return np.divide(x, l2_norm)


def remove_l2_torch(x, dim, output_stat=False):
    """
    Rescale by the l2 norm (defined as a mean)

    :param x: torch tensor
    :param dim: dim of normalization
    :param output_stat: whether to output the l2 norms
    :return:
    """
    if dim is None:
        return x
    l2_norm = ((x ** 2).mean(dim=dim, keepdim=True)) ** 0.5
    if output_stat:
        return torch.div(x, l2_norm), l2_norm
    else:
        return torch.div(x, l2_norm)


def remove_mean_np(x, axis, output_stat=False):
    """
    Remove the mean of a numpy array along certain axis

    :param x: a numpy array
    :param axis: the spatial dimensions to be normalized
    :param output_stat: wheter to output the normalized vector only or withs the means and stds
    :return:
    """
    if axis is None:
        return x
    m = x.mean(axis=axis, keepdims=True)
    if output_stat:
        return x - m, m
    else:
        return x - m


def normalize(x, axis, p=2.0, backend='numpy'):
    if backend == 'numpy':
        return normalize_np(x=x, axis=axis, p=p)
    elif backend == 'torch':
        return normalize_torch(x=x, dim=axis, p=p)
    raise ValueError('Backend not recognized in normalize')


def normalize_torch(x, dim, p):
    return x / ((torch.abs(x) ** p).mean(dim=dim, keepdim=True)) ** (1 / p)


def normalize_np(x, axis, p):
    return x / ((np.abs(x) ** p).mean(axis=axis, keepdims=True)) ** (1 / p)


def standardize(x, axis, p=2.0, backend='numpy'):
    if backend == 'numpy':
        return standardize_np(x=x, axis=axis, p=p)
    elif backend == 'torch':
        return standardize_torch(x=x, dim=axis, p=p)
    raise ValueError('Backend not recognized in standardize')


def standardize_torch(x, dim, p):
    res = x - x.mean(dim=dim, keepdim=True)
    return res / ((torch.abs(res) ** p).mean(dim=dim, keepdim=True)) ** (1 / p)


def standardize_np(x, axis, p):
    res = x - x.mean(axis=axis, keepdims=True)
    return res / ((np.abs(res) ** p).mean(axis=axis, keepdims=True)) ** (1 / p)


def standardize_dx_numpy(x, axis, rm_std=True, keep_init=False):
    dx = np.diff(x, axis=axis)
    if rm_std:
        new_dx = standardize(dx, axis=axis, backend='numpy')
    else:
        new_dx = dx - dx.mean(axis=axis, keepdims=True)
    new_x = np.zeros_like(x)
    new_x[..., 1:] = np.cumsum(new_dx, axis=axis)
    if keep_init:
        new_x += np.take(x, indices=[0], axis=axis)
    return new_x


def sort_by_frequencies(x, output_order=False):
    """
    Return an array with rearranged rows
    :param x: N x T numpy array
    :return: N x T numpy array from lowest to highest frequency
    """
    N = x.shape[0]
    T = x.shape[1]
    times = np.linspace(0, 1, T // 2)

    def get_central_index(f):
        l1_norm = f.sum(0)
        f /= l1_norm
        gravity = f.dot(times)
        return gravity

    def get_central_frequency(row):
        spectrum = np.abs(np.fft.fft(row))
        return get_central_index(spectrum[:T // 2])

    order = np.argsort([get_central_frequency(x[i, :]) for i in range(N)])
    if output_order:
        return x[order, :], order
    else:
        return x[order, :]


def get_antisym_vector(A, indices):
    return A[indices[0], indices[1]]


def get_antisym_mat(v, indices, num_atoms):
    if indices is None:
        indices = torch.tril_indices(row=num_atoms, col=num_atoms, offset=-1)
    res = torch.zeros((num_atoms, num_atoms), dtype=torch.float32, device='cuda')
    res[indices[0], indices[1]] = v
    return res - res.t()


def svd(A, output_checks=False, dimensions=None):
    u, sp, vh = torch.svd(A)

    if dimensions is not None:
        u = u[:, :dimensions]
        vh = vh[:, :dimensions]
        sp = sp[:dimensions]

    sp = torch.diag(sp)

    if output_checks:
        check0 = ((A - u @ sp @ vh.t()) ** 2).sum() / (A ** 2).sum()
        return [check0], u, sp, vh
    return u, sp, vh


def svd_np(A, dim=None, output_checks=False):
    u, sp, vh = np.linalg.svd(A, full_matrices=False)
    vh = vh.T

    if not dim is None:
        u = u[:, :dim]
        vh = vh[:, :dim]
        sp = sp[:dim]

    sp = np.diag(sp)

    if output_checks:
        check0 = ((A - u @ sp @ vh.T) ** 2).sum() / (A ** 2).sum()
        return [check0], u, sp, vh
    return u, sp, vh


def schur_np(A, output_checks=False):
    # careful, when T is diagonal, its diagonal is not necessarily ordered
    T, Z = scipy.linalg.schur(a=A, output='complex')

    if output_checks:
        Z_star = np.transpose(Z.conjugate())
        checks0 = np.abs(A - Z @ T @ Z_star).sum()
        checks1 = np.abs(A - Z @ np.diag(np.diag(T)) @ Z_star).sum()  # May be false for non-orthogonal matrix A

    T = np.stack([T.real, T.imag], axis=-1)
    Z = np.stack([Z.real, Z.imag], axis=-1)

    if output_checks:
        return [checks0, checks1], T, Z
    return T, Z


def modulo(tup, N):
    res = (0,) * len(tup)
    for i in range(len(tup)):
        res[i] = tup[i] % N
    return res


def complex_diagonalization(A, output_checks=False):
    """
    Complex diagonalization of a real matrix

    :param A: a real matrix as torch tensor of size N x N
    :return: complex matrices U and D verifying U D U* = M and U U* = I,
        U: N x N x 2 containing eigenvectors as columns
        D: N x N x 2 diagonal matrix
    """
    N = A.shape[0]

    eigva, eigve = torch.eig(input=A, eigenvectors=True)
    eigveC_list = []

    is_complex_eigva = eigva[:, 1].nonzero()[:, 0]

    i_eigva = 0
    while i_eigva < N:  # TODO: this could be done matricially by multiplying blocks

        # Complex conjugated eigenvalue
        if i_eigva in is_complex_eigva:
            this_eigve = cplx.to_c(eigve[:, i_eigva], eigve[:, i_eigva + 1])
            eigveC_list.append(this_eigve)
            eigveC_list.append(cplx.conjugate(this_eigve))
            i_eigva += 2

        # Real eigenvalue
        else:
            eigveC_list.append(cplx.from_real(eigve[:, i_eigva]))
            i_eigva += 1

    eigveC = torch.stack(eigveC_list, dim=1)
    eigva = torch.stack([torch.diag(cplx.real(eigva)), torch.diag(cplx.imag(eigva))], dim=-1)

    if output_checks:
        test_decompo1 = cplx.modulus(cplx.mm(eigveC, eigva) - cplx.mm(cplx.from_real(A), eigveC)).sum()
        test_decompo2 = cplx.modulus(
            cplx.from_real(A) - cplx.mm(cplx.mm(eigveC, eigva), cplx.conjugate(eigveC).transpose(0, 1))).sum()
        return [test_decompo1, test_decompo2], eigva, eigveC
    return eigva, eigveC


def complex_diagonalization_np(A, output_checks=False):
    """
    Complex diagonalization of a real matrix

    :param A: a real matrix as numpy array of size N x N
    :return: complex matrices U and D verifying U D U* = M and U U* = I,
        U: N x N x 2 containing eigenvectors as columns
        D: N x N x 2 diagonal matrix
    """
    eigva, eigve = np.linalg.eig(A)
    eigva = np.diag(eigva)

    if output_checks:
        test_decompo1 = np.abs(eigve @ eigva - A @ eigve).sum()
        test_decompo2 = np.abs(A - eigve @ eigva @ (eigve.conjugate()).T).sum()

    eigva = np.stack([eigva.real, eigva.imag], axis=-1)
    eigve = np.stack([eigve.real, eigve.imag], axis=-1)

    if output_checks:
        return [test_decompo1, test_decompo2], eigva, eigve
    else:
        return eigva, eigve


def nullity_score(M, comparator):
    return M.abs().max_gap() / (((comparator ** 2).mean()) ** 0.5)


def nullity_score_numpy(M, comparator):
    return np.abs(M).max_gap() / (((comparator ** 2).mean()) ** 0.5)


def sparsify_antisymmetric(A):
    N = A.shape[0]
    A_sparse = A.copy()

    # Determine the two rows to be zero
    indices = [i for i in range(N)]
    rows_norm = (A_sparse ** 2).mean(axis=1)
    i0, i1 = np.argsort(rows_norm)[:2]
    indices.remove(i0)
    indices.remove(i1)

    A_sparse[i0, :] = A_sparse[:, i0] = 0.0
    A_sparse[i1, :] = A_sparse[:, i1] = 0.0

    # Put ones and minus ones at the right places
    js = []
    while len(indices) > 0:
        i = indices[0]
        j_max = np.argmax(np.abs(A_sparse[i, :]))
        sign = 1.0 if A_sparse[i, j_max] > 0 else -1.0
        if j_max in js:
            print('Problem in sparsification of antisymmetric matrix, two ones on the same row')
        A_sparse[i, :] = A_sparse[:, i] = A_sparse[j_max, :] = A_sparse[:, j_max] = 0.0
        A_sparse[i, j_max] = sign
        A_sparse[j_max, i] = -sign
        indices.remove(i)
        indices.remove(j_max)
        js.append(j_max)

    return A_sparse


def sparsify_antisymmetric_hard(A):
    N = A.shape[0]
    A_sparse = A.copy()

    # Determine the two rows to be zero
    indices = [i for i in range(N)]
    rows_norm = (A_sparse ** 2).mean(axis=1)
    i0, i1 = np.argsort(rows_norm)[:2]
    indices.remove(i0)
    indices.remove(i1)

    A_sparse[i0, :] = A_sparse[:, i0] = 0.0
    A_sparse[i1, :] = A_sparse[:, i1] = 0.0

    # Put ones and minus ones at the right places
    while len(indices) > 0:
        i = indices[0]
        # i + 1 is necessarily in indices
        sign = 1.0 if A[i, i + 1] > 0 else -1.0
        A_sparse[i, :] = A_sparse[:, i] = A_sparse[i + 1, :] = A_sparse[:, i + 1] = 0.0
        A_sparse[i, i + 1] = sign
        A_sparse[i + 1, i] = -sign
        indices.remove(i)
        indices.remove(i + 1)
    return A_sparse


def eigh(C):
    """
    Return eigva and eigve such that C = eigve^* @ np.diag(eigva) @ eigve
    :param C: (B) x N x N array
    :return: (B) x N eigen values in decreasing order, (B) x N x N eigen vectors (rows)
    """
    eigva, eigve = np.linalg.eigh(C)

    if np.isnan(eigva).any() or np.isnan(eigve).any():
        print('PCA calculation had to go through scipy')
        eigva, eigve = scipy.linalg.eigh(C)
    if np.isnan(eigva).any() or np.isnan(eigve).any():
        raise ValueError('PCA did not converge')
    eigva = np.flip(eigva, axis=-1)
    eigve = np.flip(eigve, axis=-1).conj().swapaxes(-1, -2)

    return eigva, eigve


def pca(x, rmm=True, rms=False, axis=(-2, -1)):
    """
    Compute PCA of data x.

    :param x: .. x N .. x T
    :param rmm: remove mean of x
    :param rms: divide x by its l2 norm
    :param axis: the axis index of channels and time
    :return:
    """
    y = x.swapaxes(axis[0], -2).swapaxes(axis[1], -1)
    if rmm:
        y -= y.mean(axis=-1, keepdims=True)
    if rms:
        y /= ((y ** 2).mean(axis=-1, keepdims=True)) ** 0.5
    C = y @ (y.conj().swapaxes(-1, -2)) / (y.shape[-1] ** 2)
    return eigh(C)  # lambda is obtained with the average l2 norm


def shift(x, y, tau):
    """
    Return shifted vectors : if tau > 0 x is in advance of tau on y otherwise y is in advance of tau on x
    :param x: (B) x T
    :param y: (B') x T
    :param tau:
    :return:
    """
    if tau == 0:
        return x, y
    elif tau > 0:
        return x[..., :-tau], y[..., tau:]
    return shift(y, x, -tau)[::-1]


def autocovariance_estimator(x):
    """
    Naive estimation of the function r(tau) = E[X(x)X(x+tau)] of a stationary process X.

    :param x: N x T
    :return:
    """
    r = np.zeros(x.shape[-1])
    mean = x.mean(axis=-1, keepdims=True)
    for t in range(x.shape[-1]):
        left, right = shift(x=x, y=x, tau=t)
        r[t] = ((left - mean) * (right - mean)).sum(axis=-1)
    r /= x.shape[-1]

    return r


def circular(v):
    N = v.size
    return np.array([[v[i - j] for j in range(N)] for i in range(N)])


def partition_dict(d, n):
    """Return list of list of l where l is a list in d such that the sum of the sublists are approx equal."""

    # solve the problem approximately
    def sublist_creator(x, n, sort=True):
        bins = [[0] for _ in range(n)]
        if sort:
            x = sorted(x)[::-1]
        for i, key in x:
            least = bins[0]
            least[0] += i
            least.append((i, key))
            heapreplace(bins, least)
        return [x[1:] for x in bins]

    # create a new dict containing the key in order to keep track of the partition
    valueskey = [(len(l), key) for (key, l) in d.items()]

    # find the best partition on several iterations
    tic = time()
    max_diff = 11
    best_partition, best_max_diff = None, 1e30
    while time() - tic < 2 and max_diff > 5:
        partition = sublist_creator(valueskey, n)
        partition_no_key = [[nb for (nb, _) in l] for l in partition]
        max_diff = max([sum(l) for l in partition_no_key]) - min([sum(l) for l in partition_no_key])
        if max_diff < best_max_diff or best_partition is None:
            best_max_diff, best_partition = max_diff, partition

    # transform back to list of list of cov_idx
    best_partition = [[d[key] for (_, key) in sublist] for sublist in best_partition]

    return best_partition


def get_path_basis(N, J, A_path, coeff_str, pca0, pca1):
    A1 = 1 if A_path[0] is None else A_path[0]
    A2 = None if len(A_path) <= 1 else 1 if A_path[1] is None else A_path[1]
    res = []
    if coeff_str[0] == 'i':
        I = cplx.from_real(torch.eye(N, dtype=torch.float64)).unsqueeze(0)  # 1 x N x N x 2
        res.append(I)
    elif coeff_str[0] == 'p':
        res.append(pca0)
    else:
        raise ValueError('Path name not recognized')
    if len(coeff_str) == 1:
        return res

    order1_channel = res[-1].shape[-3]
    if coeff_str[1] == 'i':
        I2 = cplx.from_real(torch.eye(order1_channel, dtype=torch.float64))
        res.append(I2.repeat(J + 1, 4, 1, 1, 1))
    elif coeff_str[1] == 'p':
        if pca1.shape[1] != A1:
            raise ValueError('pca1 has the wrong dimensions')
        res.append(pca1)
    else:
        raise ValueError('Path name not recognized')
    if len(coeff_str) == 2:
        return res

    order2_channel = res[-1].shape[-3]
    JJ = (J * J + J) // 2
    if coeff_str[2] == 'i':
        I3 = cplx.from_real(torch.eye(order2_channel, dtype=torch.float64))
        res.append(I3.repeat(JJ, A1, A2, 1, 1, 1))
        return res
    raise ValueError('Path name not recognized')


def smooth_junction(t, t1, t2, x0, x1):
    """Smooth non-decreasing function f(t) that satisfies:
        - f(t) = x0 for t < t1
        - f(t) = x1 for t > t2
    """
    assert t1 <= t2

    def h(t):
        x = np.zeros_like(t)
        x[t > 0.0] = np.exp(-1 / t[t > 0.0])
        return x

    def g(t):
        return h(t) / (h(t) + h(1 - t))

    if t1 == t2:
        x = np.ones_like(t)
        x[t <= t1] = 0.0
        return x

    return g((t - t1) / (t2 - t1)) * (x1 - x0) + x0


def cumsum_zero(dx):
    """ Cumsum of a vector preserving dimension through zero-pading. """
    res = np.cumsum(dx, axis=-1)
    res = np.concatenate([np.zeros_like(res[..., 0:1]), res], axis=-1)
    return res
