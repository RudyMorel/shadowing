""" A class that implements the scale paths used in the scattering transform. """
from typing import *
from itertools import product, chain
from collections import OrderedDict
import numpy as np
import torch
import pandas as pd


""" Notations
- sc_path or path: a scale path, tuple (j1, j2, ... jr)
- sc_idx or idx: paths are numbered by their scale path
- J: number of octaves
- Q1: number of wavelets per octave on first wavelet layer
- Q2: number of wavelets per octave on second wavelet layer
"""


class ScatteringShape:
    def __init__(self,
                 N: int,
                 n_scales: int,
                 A: int,
                 T: int) -> None:
        self.N = N
        self.n_scales = n_scales
        self.A = A
        self.T = T


class ScaleIndexer:
    """ Implements the scale paths used in the scattering transform. """
    def __init__(self,
                 r: int,
                 J: List[int],
                 Q: List[int],
                 strictly_increasing: Optional[bool] == True) -> None:
        self.r, self.J, self.Q = r, J, Q
        self.strictly_increasing = strictly_increasing

        self.sc_paths = self.create_sc_paths()  # list[order] array
        self.p_coding, self.p_decoding = self.construct_path_coding_dicts()
        self.sc_idces = self.create_sc_idces()  # # list[order] array

        self.low_pass_mask = self.compute_low_pass_mask()  # list[order] array

    def JQ(self, r: int) -> int:
        """ Return the number of wavelet at a certain order. """
        return self.J[r-1] * self.Q[r-1]

    def condition(self, path: List[int]) -> bool:
        """ Tells if path j1, j2 ... j{r-1} jr is admissible. """
        def compare(i, j):
            return i < j if self.strictly_increasing else i <= j
        depth_ok = len(path) <= self.r
        scales_ok = all(compare(i // self.Q[o], j // self.Q[o+1]) for o, (i, j) in enumerate(zip(path[:-1], path[1:])))
        low_pass_ok = sum([j == self.JQ(o+1) for o, j in enumerate(path)]) <= 1
        return depth_ok and scales_ok and low_pass_ok

    def create_sc_paths(self) -> List[np.ndarray]:
        """ The tables j1, j2 ... j{r-1} jr for every order r. """
        sc_paths_l = []
        for r in range(1, self.r + 1):
            sc_paths_r = np.array([p for p in product(*[range(self.JQ(o+1)+1) for o in range(r)]) if self.condition(p)])
            sc_paths_l.append(sc_paths_r)
        return sc_paths_l

    def create_sc_idces(self) -> List[np.ndarray]:
        """ The scale idces numerating scale paths. """
        sc_idces_l = []
        for r in range(1, self.r + 1):
            sc_idces_r = np.array([self.path_to_idx(p) for p in self.sc_paths[r-1]])
            sc_idces_l.append(sc_idces_r)
        return sc_idces_l

    def construct_path_coding_dicts(self) -> Tuple[Dict[Tuple, int], Dict[int, Tuple]]:
        """ Construct the enumeration idx -> path. """
        coding = OrderedDict({(): 0})
        coding = OrderedDict(coding, **{tuple(path): i for i, path in enumerate(list(chain.from_iterable(self.sc_paths)))})
        decoding = OrderedDict({v: k for k, v in coding.items()})

        return coding, decoding

    def get_all_paths(self) -> List[Tuple]:
        return list(self.p_coding.keys())

    def get_all_idces(self) -> List[int]:
        return list(self.p_decoding.keys())

    def path_to_idx(self, path: Union[Tuple, List, np.ndarray]) -> int:
        """ Return scale index i corresponding to path. """
        path = np.array(path)
        if len(path) > 0 and path[-1] == -1:
            i0 = np.argmax(path == -1)
            path = path[:i0]
        return self.p_coding[tuple(path)]

    def idx_to_path(self, idx: int, squeeze: Optional[bool] = True) -> Tuple[int]:
        """ Return scale path j1, j2 ... j{r-1} jr corresponding to scale index i. """
        if idx == -1:
            return tuple()
        if squeeze:
            return self.p_decoding[idx]
        return self.p_decoding[idx] + (pd.NA, ) * (self.r - len(self.p_decoding[idx]))

    def is_low_pass(self, path: Union[Tuple, List, np.ndarray]) -> bool:
        """ Determines if the path indexed by idx is ending with a low-pass. """
        if isinstance(path, (int, np.integer)):
            path = self.idx_to_path(path)
        return path[-1] >= self.JQ(self.order(path))

    def order(self, path: Union[Tuple, List, np.ndarray]) -> int:
        """ The scattering order of the path indexed by idx. """
        if isinstance(path, (int, np.integer)):
            path = self.idx_to_path(path)
        return len(path)

    def compute_low_pass_mask(self) -> List[torch.Tensor]:
        """ Compute the low pass mask telling at each order which are the paths ending with a low pass filter. """
        return [torch.LongTensor(paths[:, -1]) == self.JQ(order+1)
                for order, paths in enumerate(self.sc_paths[:3])]
