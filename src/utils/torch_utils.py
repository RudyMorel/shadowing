""" Utils function for manipulating tensors. """
from typing import *
import numpy as np
import torch


def is_long_tensor(z):
    return isinstance(z, torch.LongTensor) or isinstance(z, torch.cuda.LongTensor)


def is_double_tensor(z):
    return isinstance(z, torch.DoubleTensor) or isinstance(z, torch.cuda.DoubleTensor)


def to_numpy(tensor):
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    return tensor.detach().numpy()


def unsqueeze(tsr, n, dim):
    """ Add n singleton dimensions at dim. """
    return tsr.view(tsr.shape[:dim] + (1,)*n + tsr.shape[dim:])


def multid_where(a: Iterable, b: Iterable) -> List:
    """ Find the position in b of each element of a.

    :param a: (K) x n array
    :param b: C x n array
    :return: (K) array
    """
    # todo: check that converting to tuple is ok. tobyte was giving different results for same arrays
    d = {tuple(item): idx for idx, item in enumerate(b)}
    occurences = [d.get(tuple(item)) for item in a]

    return occurences


def multid_where_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Find the position in b of each element of a.

    :param a: (K) x n array
    :param b: C x n array
    :return: (K) array
    """
    # todo: check that converting to tuple is ok. tobyte was giving different results for same arrays
    d = {tuple(item): idx for idx, item in enumerate(b)}
    a_it = a.reshape(-1, a.shape[-1])
    occurences = np.array([d.get(tuple(item)) for item in a_it])

    occurences = occurences.reshape(a.shape[:-1])

    return occurences


def multid_row_isin(a, b):
    """ Returns if the row k in A is fully present in B.

    :param a: K x J x n array
    :param b: C x n array
    :return: K array
    """
    d = {item.tobytes(): 0 for item in b}
    a_isin = np.array([x.tobytes() in d.keys() for x in a.reshape(-1, a.shape[-1])])
    a_isin = a_isin.reshape(a.shape[:-1])
    a_isin = a_isin.prod(-1)

    return a_isin.astype(bool)
