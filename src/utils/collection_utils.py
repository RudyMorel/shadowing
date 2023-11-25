""" Utils function for manipulating collections. """
from typing import *
from functools import reduce
from collections import OrderedDict
import itertools
import numpy as np
import pandas as pd


def transpose(li: Iterable[Iterable]) -> List[List]:
    """Transpose [[a11, ..., a1n], ... [am1, ..., amn]] and return  [[a11, ..., am1], ... [a1n, ..., amn]]."""
    return list(map(list, zip(*li)))


def compose(*functions: Callable) -> Callable:
    """Given functions f1, ..., fn, return f1 o ... o fn."""
    def inner(arg):
        for f in reversed(functions):
            arg = f(arg)
        return arg
    return inner


def select_rs(x, r, s):
    if r is not None:
        x = x[min(r, x.shape[0] - 1):min(r, x.shape[0] - 1) + 1, :, :]
    if s is not None:
        x = x[:, min(s, x.shape[1] - 1):min(s, x.shape[1] - 1) + 1, :]
    return x


def dfs_edges(g, source=None, depth_limit=None):
    if source is None:
        # edges for all components
        nodes = g
    else:
        # edges for components with source
        nodes = source
    visited = set()
    if depth_limit is None:
        depth_limit = len(g)
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start, depth_limit, iter(g[start]))]
        while stack:
            parent, depth_now, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    yield parent, child
                    visited.add(child)
                    if depth_now > 1:
                        stack.append((child, depth_now - 1, iter(g[child])))
            except StopIteration:
                stack.pop()


def concat_list(a: List[List]) -> List:
    return list(itertools.chain.from_iterable(a))


def reverse_permutation(s: Sequence) -> List:
    """Giben permutation s: i -> j return the inverse permutation j -> i."""
    sinv = [0] * len(s)
    for i in range(len(s)):
        sinv[s[i]] = i
    return sinv


def get_permutation(a: Sequence, b: Sequence):
    """Return the permutation s such that a[s[i]] = b[i]"""
    assert set(a) == set(b)

    d = {val: key for key, val in enumerate(a)}
    s = [d[val] for val in b]

    return s


def split_equal_sum(li: List[int], r: int) -> Tuple[List[List[int]], List[List[int]]]:
    sublists = OrderedDict({s: [] for s in range(r)})
    subindices = OrderedDict({s: [] for s in range(r)})
    sums = np.zeros((r,))

    li = np.array(li)
    order = np.argsort(li)[::-1]

    for i, x in enumerate(li[order]):
        i0 = np.argmin(sums)
        sublists[i0].append(x)
        subindices[i0].append(order[i])
        sums[i0] += x

    return list(subindices.values()), list(sublists.values())


def df_product(*dfs: pd.DataFrame) -> pd.DataFrame:
    for df in dfs:
        df['key'] = 1
    return reduce(lambda l, r: pd.merge(l, r, on='key'), dfs).drop(columns='key')


def df_product_channel_single(df: pd.DataFrame, N: int, method: str) -> pd.DataFrame:
    """ Pandas cartesian product {(0,0), ..., (0, Nl-1))} x df """
    if method == "same":
        df_n = pd.DataFrame(np.stack([np.arange(N), np.arange(N)], 1), columns=['nl', 'nr'])
    elif method == "zero_left":
        df_n = pd.DataFrame(np.stack([np.zeros(N, dtype=np.int32), np.arange(N)], 1), columns=['nl', 'nr'])
    elif method == "zero_right":
        df_n = pd.DataFrame(np.stack([np.arange(N), np.zeros(N, dtype=np.int32)], 1), columns=['nl', 'nr'])
    else:
        raise ValueError("Unrecognized channel product method.")

    return df_product(df_n, df)
