""" Moments to be used on top of a scattering transform. """
from typing import *
from itertools import product
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.layers.scale_indexer import ScaleIndexer
from src.layers.described_tensor import Description


class Estimator(nn.Module):
    """ Estimator used on scattering. """
    def forward(self, x: torch.tensor, **kwargs) -> torch.tensor:
        pass


class TimeAverage(Estimator):
    """ Averaging operator to estimate probabilistic expectations or correlations. """
    def __init__(self, window: Optional[Iterable] = None) -> None:
        super(TimeAverage, self).__init__()
        self.w = torch.tensor(window, dtype=torch.int64) if window is not None else None

    def forward(self, x: torch.tensor, **kwargs) -> torch.tensor:
        if self.w is not None:
            x = x[..., self.w]
        return x.mean(-1, keepdim=True)


class WindowSelector(Estimator):
    """ Selecting operator. """
    def __init__(self, window: Iterable) -> None:
        super(WindowSelector, self).__init__()
        self.w = torch.tensor(window, dtype=torch.int64)

    def forward(self, x: torch.tensor, **kwargs) -> torch.tensor:
        return x[..., self.w]


class Order1Moments(nn.Module):
    """ Average low passes at a given scattering order. """
    def __init__(self, ave: Optional[Estimator] = None):
        super(Order1Moments, self).__init__()
        self.ave = ave or TimeAverage()

    def forward(self, Wx: torch.tensor) -> torch.tensor:
        """ Computes E{Wx} and E{|Wx|}.

        :param Wx: B x N x js x A x T tensor
        :return: B x N x K x T' tensor
        """
        y_mod = self.ave(torch.abs(Wx[:, :, :-1, :, :]))
        y_low = self.ave(Wx[:, :, -1:, :, :])

        y = torch.cat([y_mod, y_low], dim=-3)

        return y.reshape(y.shape[0], y.shape[1], -1, y.shape[-1])


class Exp(nn.Module):
    """ Average second order scattering layer. """
    def __init__(self, r: int, sc_idxer: ScaleIndexer, ave: Optional[Estimator] = None):
        super(Exp, self).__init__()
        self.ave = ave or TimeAverage()

        self.lp_mask = sc_idxer.low_pass_mask[r-1]

    def forward(self, WmWx: torch.tensor) -> torch.tensor:
        """ Computes E{WmWx}.

        :param WmWx: B x N x js x A x T tensor
        :return: B x N x K x T' tensor
        """
        y = self.ave(WmWx[:, :, self.lp_mask, :, :])

        return self.ave(WmWx[:, :, self.lp_mask, :, :])


class ScatCoefficients(nn.Module):
    """ Compute per channel (marginal) order q moments. """
    def __init__(self, qs: List[float], ave: Optional[Estimator] = None):
        super(ScatCoefficients, self).__init__()
        self.c_types = ['marginal']
        self.ave = ave or TimeAverage()

        self.register_buffer('qs', torch.tensor(qs))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """ Computes E[|Sx|^q].

        :param x: B x N x js x A x T tensor
        :return: B x N x js x A x len(qs) x T' tensor
        """
        return self.ave(torch.abs(x).unsqueeze(-2) ** self.qs[:, None])


class Cov(nn.Module):
    """ Diagonal model along scales. """
    def __init__(self, rl: int, rr: int, sc_idxer: ScaleIndexer, nchunks: int, ave: Optional[Estimator] = None):
        super(Cov, self).__init__()
        self.sc_idxer = sc_idxer
        self.nchunks = nchunks

        self.df_scale = self.create_scale_description(sc_idxer.sc_idces[rl-1], sc_idxer.sc_idces[rr-1], sc_idxer)

        self.idx_l, self.idx_r = self.df_scale[['scl', 'scr']].values.T
        if rl == 2:
            self.idx_l -= sc_idxer.JQ(1) + 1
        if rr == 2:
            self.idx_r -= sc_idxer.JQ(1) + 1

        self.ave = ave or TimeAverage()

    @staticmethod
    def create_scale_description(scls: np.ndarray, scrs: np.ndarray, sc_idxer: ScaleIndexer) -> pd.DataFrame:
        """ Return the dataframe that describes the scale association in the output of forward. """
        info_l = []
        for (scl, scr) in product(scls, scrs):
            rl, rr = sc_idxer.order(scl), sc_idxer.order(scr)
            ql, qr = sc_idxer.Q[rl-1], sc_idxer.Q[rr-1]
            if rl > rr:
                continue

            pathl, pathr = sc_idxer.idx_to_path(scl), sc_idxer.idx_to_path(scr)
            jl, jr = sc_idxer.idx_to_path(scl, squeeze=False), sc_idxer.idx_to_path(scr, squeeze=False)

            if rl == rr == 2 and pathl < pathr:
                continue

            # remove scale paths that has the redundancy |j1|j1 for capturing envelope correlations
            if rl == rr == 2 and np.any((np.diff(pathl) == 0) | (np.diff(pathr) == 0)):
                continue

            # correlate low pass with low pass or band pass with band pass and nothing else
            if (sc_idxer.is_low_pass(scl) and not sc_idxer.is_low_pass(scr)) or \
                    (not sc_idxer.is_low_pass(scl) and sc_idxer.is_low_pass(scr)):
                continue

            # only consider wavelets with non-negligibale overlapping support in Fourier
            # weak condition: last wavelets must be closer than one octave
            # if abs(pathl[-1] / ql - pathr[-1] / qr) >= 1:
            #     continue
            # strong condition: last wavelets must be equal
            if abs(pathl[-1] / ql - pathr[-1] / qr) > 0:
                continue

            low = sc_idxer.is_low_pass(scl)

            info_l.append(('ps' if rl * rr == 1 else 'phaseenv' if rl * rr == 2 else 'envelope',
                           2, rl, rr, scl, scr, *jl, *jr, 0, 0, low or scl == scr, low))

        out_columns = ['c_type', 'q', 'rl', 'rr', 'scl', 'scr'] + \
                      [f'jl{r}' for r in range(1, sc_idxer.r + 1)] + \
                      [f'jr{r}' for r in range(1, sc_idxer.r + 1)] + \
                      ['al', 'ar', 'real', 'low']
        df_scale = pd.DataFrame(info_l, columns=out_columns)

        # now do a diagonal or cartesian product along channels
        df_scale = (
            df_scale
            .drop('jl2', axis=1)
            .rename(columns={'jr2': 'j2'})
        )

        return df_scale

    @staticmethod
    def get_channel_idx(Nl, Nr, diago_n):
        if diago_n:
            nl = nr = torch.arange(Nl)
        else:
            nl, nr = torch.tensor(list(product(range(Nl), range(Nr)))).T
        return nl, nr

    def forward(self, sxl: torch.tensor, sxr: Optional[torch.tensor] = None,
                diago_n: Optional[bool] = True) -> torch.tensor:
        """ Extract diagonal covariances j2=j'2.

        :param sxl: B x Nl x jl x Al x T tensor
        :param sxr: B x Nr x jr x Ar x T tensor
        :return: B x channels x K x T' tensor
        """
        if sxr is None:
            sxr = sxl

        # select communicating scales
        scl, scr = self.idx_l, self.idx_r
        xl, xr = sxl[:, :, scl, 0, :], sxr[:, :, scr, 0, :]

        # select communicating channels
        nl, nr = self.get_channel_idx(sxl.shape[1], sxr.shape[1], diago_n)
        xl, xr = xl[:, nl, ...], xr[:, nr, ...]

        y = self.ave(xl * xr.conj())

        return y


class CovScaleInvariant(nn.Module):
    """ Reduced representation by making covariances invariant to scaling. """
    def __init__(self, sc_idxer: ScaleIndexer, df_scale_input: pd.DataFrame):
        super(CovScaleInvariant, self).__init__()
        self.df_scale_input = df_scale_input
        self.register_buffer('P', self._construct_invariant_projector(sc_idxer.JQ(1)))

    @staticmethod
    def create_scale_description(sc_idxer) -> pd.DataFrame:
        """ Return the dataframe that describes the output of forward. """
        J = sc_idxer.JQ(1)

        data = []

        # phase-envelope coefficients
        for a in range(1, J):
            data.append((2, 1, 2, a, pd.NA, 0, 0, False, False, 'phaseenv'))

        # scattering coefficients
        for (a, b) in product(range(J-1), range(-J+1, 0)):
            if a - b >= J:
                continue
            data.append((2, 2, 2, a, b, 0, 0, a == 0, False, 'envelope'))

        df_output = pd.DataFrame(data, columns=['q', 'rl', 'rr', 'a', 'b', 'al', 'ar', 'real', 'low', 'c_type'])

        return df_output

    def _construct_invariant_projector(self, J) -> torch.tensor:
        """ The projector P that takes a scattering covariance matrix C and computes PC the invariant projection. """
        df = Description(self.df_scale_input)

        P_l = []

        # phase-envelope coefficients
        for a in range(1, J):
            P_row = torch.zeros(self.df_scale_input.shape[0], dtype=torch.complex64)
            for j in range(a, J):
                mask = df.where(jl1=j, jr1=j-a, c_type='phaseenv')
                assert mask.sum() == 1
                P_row[mask] = 1.0
            P_l.append(P_row)

        # scattering coefficients
        for (a, b) in product(range(J-1), range(-J+1, 0)):
            if a - b >= J:
                continue

            P_row = torch.zeros(self.df_scale_input.shape[0], dtype=torch.complex64)
            for j in range(a, J+b):
                mask = df.where(jl1=j, jr1=j-a, j2=j-b)
                assert mask.sum() == 1
                P_row[mask] = 1.0
            P_l.append(P_row)

        P = torch.stack(P_l)

        # to get average along j instead of sum
        P /= P.sum(-1, keepdim=True)

        return P

    def forward(self, cov: torch.tensor) -> torch.tensor:
        """
        Keeps the scale invariant part of a Scattering Covariance. It is obtained by projection.

        :param cov: B x Nl x Nr x K tensor
        :return:
        """
        return self.P @ cov
