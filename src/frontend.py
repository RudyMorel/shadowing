""" Frontend functions for analysis and generation. """
import os
from pathlib import Path
from itertools import product
from time import time

import numpy as np
from numpy.random import default_rng
import scipy
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.utils import to_numpy, df_product, df_product_channel_single
from src.data_source import ProcessDataLoader, FBmLoader, PoissonLoader, MRWLoader, SMRWLoader
from src.layers import (Description, DescribedTensor, ChunkedModule,
                        NormalizationLayer, Wavelet, Modulus, PhaseOperator, LinearLayer,
                        Exp, Order1Moments, ScatCoefficients, Cov, CovScaleInvariant, MSELossScat,
                        ScaleIndexer, format_np, format_torch,
                        Solver, CheckConvCriterion, SmallEnoughException, MaxIteration)

""" Notations

Dimension sizes:
- B: batch size
- R: number of realizations (used to estimate scattering covariance)
- N: number of data channels (N>1 : multivariate process)
- T: number of data samples (time samples)
- J: number of scales (octaves)
- Q: number of wavelets per octave
- r: number of conv layers in a scattering model

Tensor shapes:
- x: input, of shape  (B, N, T)
- Rx: output (DescribedTensor), 
    - y: tensor of shape (B, K, T) with K the number of coefficients in the representation
    - descri: pandas DataFrame of shape K x nb_attributes used, description of the output tensor y
"""


##################
# DATA LOADING
##################

def load_data(process_name, R, T, cache_dir=None, **data_param):
    """ Time series data loading function.

    :param process_name: fbm, poisson, mrw, smrw, hawkes, turbulence or snp
    :param R: number of realizations
    :param T: number of time samples
    :param cache_dir: the directory used to cache trajectories
    :param data_param: the model
    :return: dataloader
    """
    loader = {
        'fbm': FBmLoader,
        'poisson': PoissonLoader,
        'mrw': MRWLoader,
        'smrw': SMRWLoader,
    }

    if process_name == 'snp':
        raise ValueError("S&P data is private, please provide your own data.")
    if process_name == 'heliumjet':
        raise ValueError("Helium jet data is private, please provide your own data.")
    if process_name == 'hawkes':
        raise ValueError("Hawkes data is not yet supported.")
    if process_name not in loader.keys():
        raise ValueError("Unrecognized model name.")

    if cache_dir is None:
        cache_dir = Path(__file__).parents[0] / '_cached_dir'

    dtld = loader[process_name](cache_dir)
    x = dtld.load(R=R, n_files=R, T=T, **data_param).x

    return x


##################
# ANALYSIS
##################


class Model(nn.Module):
    """ Model class for analysis and generation. """
    def __init__(self, model_type, qs,
                 T, r, J, Q, wav_type, high_freq, wav_norm,
                 A,
                 rpad,
                 channel_transforms, N, Ns,
                 sigma2, norm_on_the_fly, standardize,
                 estim_operator,
                 c_types,
                 cov_chunk,
                 dtype,
                 histogram_moments,
                 skew_redundance):
        super(Model, self).__init__()
        self.model_type = model_type
        self.sc_idxer = ScaleIndexer(r=r, J=J, Q=Q, strictly_increasing=not skew_redundance)
        self.r = r

        # time layers
        self.Ws = nn.ModuleList([Wavelet(T, J[o], Q[o], wav_type[o], wav_norm[o], high_freq[o], rpad, o+1, self.sc_idxer)
                                 for o in range(r)])

        # phase transform
        self.A = A
        self.phase_operator = Modulus() if self.A is None else PhaseOperator(A)

        # normalization layer
        self.standardize = standardize
        if norm_on_the_fly:
            self.norm_layer_scale = NormalizationLayer(2, None, True)
        elif model_type == 'covreduced' or sigma2 is not None:
            self.norm_layer_scale = NormalizationLayer(2, sigma2.pow(0.5), False)
        else:
            self.norm_layer_scale = nn.Identity()

        # channel transforms
        if channel_transforms is None:
            self.L1 = nn.Identity()
            self.L2 = nn.Identity()
            self.Lphix = nn.Identity()
        else:
            B1, B2, B3 = channel_transforms
            self.L1 = LinearLayer(B1)
            self.L2 = LinearLayer(B2)
            self.Lphix = LinearLayer(B3)
        self.N = N
        self.Ns = Ns[:-1]
        self.norm_L1 = nn.Identity()
        self.norm_L2 = nn.Identity()

        # marginal moments
        self.module_scat = ScatCoefficients(qs or [1.0, 2.0], estim_operator)
        self.module_scat_q1 = ScatCoefficients([1.0], estim_operator)
        self.module_q1 = Order1Moments(estim_operator)
        self.module_exp_r1 = Exp(1, self.sc_idxer, estim_operator)
        self.histogram_moments = histogram_moments

        if r == 2:
            self.module_exp_r2 = Exp(2, self.sc_idxer, estim_operator)
            # correlation moments
            self.module_cov_w = Cov(1, 1, self.sc_idxer, 1, estim_operator)
            self.module_cov_wmw = Cov(1, 2, self.sc_idxer, 1, estim_operator)
            self.module_cov_mw = Cov(2, 2, self.sc_idxer, cov_chunk, estim_operator)
            self.df_cov = Description(self.build_description_correlation([1, 1], self.sc_idxer))
            self.module_covinv = CovScaleInvariant(self.sc_idxer, self.df_cov) if model_type == "covreduced" else None

        self.description = self.build_description()

        # self.all_c_types = None if "c_type" not in self.description.columns else self.description.c_type.unique().tolist()

        self.c_types = c_types or self.description.c_type.unique().tolist()

        # cast model to the right precision
        if dtype == torch.float64:
            self.double()

    def double(self):
        """ Change model parameters and buffers to double precision (float64 and complex128). """
        def cast(t):
            if t.is_floating_point():
                return t.double()
            if t.is_complex():
                return t.to(torch.complex128)
            return t
        return self._apply(cast)

    def build_descri_scattering_network(self, Ns):
        """ Assemble the description of output of Sx = (Wx, W|Wx|, ..., W|...|Wx||). """
        r_max = self.sc_idxer.r

        df_l = []
        for (r, N, sc_idces, sc_paths) in \
                zip(range(1, r_max + 1), Ns, self.sc_idxer.sc_idces, self.sc_idxer.sc_paths):
            # assemble description at a given scattering layer r
            ns = pd.DataFrame(np.arange(N), columns=['n'])
            scs = pd.DataFrame([sc for sc in sc_idces], columns=['sc'])
            js = pd.DataFrame(np.array([self.sc_idxer.idx_to_path(sc, squeeze=False) for sc in scs.sc.values]),
                              columns=[f'j{r}' for r in range(1, r_max + 1)])
            scs_js = pd.concat([scs, js], axis=1)
            a_s = pd.DataFrame(np.arange(1), columns=['a'])
            df_l.append(df_product(ns, scs_js, a_s))
        df = pd.concat(df_l)
        df['low'] = [self.sc_idxer.is_low_pass(sc) for sc in df['sc'].values]
        df['r'] = [self.sc_idxer.order(sc) for sc in df['sc'].values]
        df = df.reindex(columns=['r', 'n', 'sc', *[f'j{r}' for r in range(1, r_max + 1)], 'a', 'low'])

        return df

    @staticmethod
    def make_description_compatible(df):
        """ Convert marginal description to correlation description. """
        df = df.rename(columns={'r': 'rl', 'n': 'nl', 'sc': 'scl', 'j1': 'jl1', 'a': 'al'})
        df['real'] = True
        df['nr'] = df['nl']
        df['rr'] = df['scr'] = df['ar'] = df['jr1'] = pd.NA
        df['c_type'] = ['mean' if low else 'spars' for low in df.low.values]
        df = df.reindex(columns=['c_type', 'nl', 'nr', 'q', 'rl', 'rr',
                                 'scl', 'scr', 'jl1', 'jr1', 'j2', 'al', 'ar', 'real', 'low'])

        return df

    def build_description_q1_moments(self, N):
        """ Assemble the description of averages E{Wx} and E{|Wx|}. """
        df = self.build_descri_scattering_network([N])
        df = df.query("r==1")

        # compatibility with covariance description
        df = Model.make_description_compatible(df)
        df['q'] = 1

        return df

    @staticmethod
    def build_description_correlation(Ns, sc_idxer):
        """ Assemble the description the phase modulus correlation E{Sx, Sx}. """
        scs_r1, scs_r2 = sc_idxer.sc_idces[:2]

        df_ww = Cov.create_scale_description(scs_r1, scs_r1, sc_idxer)
        df_wmw = Cov.create_scale_description(scs_r1, scs_r2, sc_idxer)
        df_mw = Cov.create_scale_description(scs_r2, scs_r2, sc_idxer)

        def channel_expand(df, N1, N2):
            return df_product_channel_single(df, N1, "same")

        df_cov = pd.concat([
            channel_expand(df_ww, Ns[0], Ns[0]),
            channel_expand(df_wmw, Ns[0], Ns[1]),
            channel_expand(df_mw, Ns[0], Ns[1])
        ])

        return df_cov

    def build_description(self):
        """ Assemble the description of output of forward. """

        if self.model_type is None:

            df = self.build_descri_scattering_network(self.Ns)

        elif self.model_type == 'scat':

            df = self.build_descri_scattering_network(self.Ns)
            df['c_type'] = 'scat'
            df['real'] = True
            qs = pd.DataFrame(self.module_scat.qs.detach().cpu().numpy(), columns=['q'])
            df = df_product(df, qs)

        elif self.model_type == 'cov':

            df_r1 = self.build_description_q1_moments(self.N)
            df_r2 = self.build_description_correlation([self.N, self.N], self.sc_idxer)

            df = pd.concat([df_r1, df_r2])

        elif self.model_type == 'covreduced':

            df_r1 = self.build_description_q1_moments(self.N)
            df_r2 = self.build_description_correlation([self.N, self.N], self.sc_idxer)

            # ps and low pass of phaseenv and envelope
            df_cov_non_invariant = df_r2[df_r2['low'] | (df_r2['c_type'] == "ps")]
            df_non_invariant = pd.concat([df_r1, df_cov_non_invariant])

            # phaseenv and envelope that are invariant
            df_inv = CovScaleInvariant.create_scale_description(self.sc_idxer)
            df_inv = df_product_channel_single(df_inv, self.N, method="same")

            # make non-invariant / invariant descriptions compatible
            df_inv['scr'] = df_inv['scl'] = df_inv['jl1'] = df_inv['jr1'] = df_inv['j2'] = pd.NA
            df_non_invariant['a'] = df_non_invariant['b'] = pd.NA

            df = pd.concat([df_non_invariant, df_inv])

        elif self.model_type == 'scat+cov':

            df_exp = self.build_description_q1_moments(self.N)

            df_scat = self.build_descri_scattering_network(self.Ns)
            df_scat = df_scat[df_scat['r'] == 2]
            df_scat = self.make_description_compatible(df_scat)
            df_scat['q'] = 1
            df_scat['c_type'] = 'scat'

            df_cov = self.build_description_correlation(self.Ns, self.sc_idxer)

            df = pd.concat([df_exp, df_scat, df_cov])

        else:

            raise ValueError("Unrecognized model type.")

        if self.histogram_moments:
            df_hist = df.query("c_type=='spars'").copy()
            df_hist.loc[:, 'c_type'] = 'hist_skewness'
            df = pd.concat([df, df_hist])

        return Description(df)

    def compute_scattering(self, x):
        """ Compute the Wx, W|Wx|, ..., W|...|Wx||. """
        Sx_l = []
        for order, W in enumerate(self.Ws):
            x = W(x)
            if order == 0:
                x = self.norm_layer_scale(x)
            Sx_l.append(x)
            x = torch.abs(x)

        return Sx_l

    def compute_spars(self, Wx, reshape=True):
        """ Compute E{Wx} and E{|Wx|}. """
        exp = self.module_q1(Wx)
        # exp = self.module_q1(Wx).pow(2.0)  # TODO. Under testing.
        if reshape:
            return exp.view(exp.shape[0], -1, exp.shape[-1])
        return exp

    def compute_phase_mod_correlation(self, Wx, WmWx, reshape=True):
        """ Compute phase-modulus correlation matrix E{rho Wx (rho Wx)^ *}. """
        cov1 = self.module_cov_w(Wx, Wx)
        cov2 = self.module_cov_wmw(Wx, WmWx)
        cov3 = self.module_cov_mw(WmWx, WmWx)

        def reshaper(y):
            if reshape:
                return y.view(y.shape[0], -1, y.shape[-1])
            return y

        return torch.cat([reshaper(cov) for cov in [cov1, cov2, cov3]], dim=-2)

    def count_coefficients(self, **kwargs) -> int:
        """ Returns the number of moments satisfying kwargs. """
        descri = self.description
        if self.c_types is not None:
            descri = descri.reduce(c_type=self.c_types)
        return descri.where(**kwargs).sum()

    def forward(self, x):

        if self.standardize:
            x = x / x.std(-1, keepdim=True)

        # scattering layer
        Sx = self.compute_scattering(x)

        if self.model_type is None:

            y = torch.cat([out.view(x.shape[0], -1, x.shape[-1]) for out in Sx], dim=1)

        elif self.model_type == 'scat':

            Sx = torch.cat([out.view(x.shape[0], -1, x.shape[-1]) for out in Sx], dim=1)
            y = self.module_scat(Sx)
            y = y.view(y.shape[0], -1, y.shape[-1])

        elif self.model_type == 'cov':

            exp2 = self.module_exp_r2(Sx[1])
            exp2 = exp2.view(exp2.shape[0], -1, exp2.shape[-1])

            exp1 = self.module_exp_r1(Sx[0])
            exp1 = exp1.view(exp1.shape[0], -1, exp1.shape[-1])

            cov = self.compute_phase_mod_correlation(*Sx)

            y = torch.cat([exp2, exp1, cov], dim=1)

        elif self.model_type == 'covreduced':

            exp = self.compute_spars(Sx[0])

            noninv_mask = self.df_cov.where(c_type="ps") | self.df_cov.where(low=True)
            cov_full = self.compute_phase_mod_correlation(*Sx, reshape=False)
            cov_noninv = cov_full[..., noninv_mask, :]
            cov_inv = self.module_covinv(cov_full)  # invariant to scaling

            cov = torch.cat([c.view(c.shape[0], -1, c.shape[-1]) for c in [cov_noninv, cov_inv]], dim=-2)

            y = torch.cat([exp, cov], dim=-2)

        elif self.model_type == 'scat+cov':
            Wx, WmWx = Sx[:2]

            exp1 = self.compute_spars(Wx)
            exp2 = self.module_scat_q1(WmWx)
            exp = torch.cat([exp1, exp2.view(exp2.shape[0], -1, exp2.shape[-1])], dim=1)

            cov = self.compute_phase_mod_correlation(Wx, WmWx)

            y = torch.cat([exp, cov], 1)

        else:

            raise ValueError("Unrecognized model type.")

        if not y.is_complex():
            y = torch.complex(y, torch.zeros_like(y))

        if self.histogram_moments:
            # get multi-scale increments (can be seen as a particular wavelet transform)
            filters = torch.tensor([[1] * (2 ** j) + [0] * (x.shape[-1] - 2 ** j) for j in range(self.sc_idxer.J[0])],
                                   dtype=x.dtype, device=x.device)
            def multiscale_dx(x):
                return torch.fft.ifft(torch.fft.fft(filters[:,None,:]) * torch.fft.fft(x)).real

            dx = multiscale_dx(x)
            dx_norm = dx.pow(2.0).mean(-1, keepdim=True).pow(0.5)
            # def energy(x):
            #     return x.pow(2.0).mean((0,1,3,4)) / target_norm.pow(2.0).mean((0,1,3,4))
            def skewness(dx):
                # x_norm = x / target_norm
                # x2_signed = nn.ReLU()(x_norm).pow(2.0) - nn.ReLU()(-x_norm).pow(2.0)
                return torch.sigmoid(dx / dx_norm).mean(-1, keepdim=True)
            # def kurtosis(x):
            #     return torch.abs(x).mean((0,1,3,4)).pow(2.0) / target_norm.pow(2.0).mean((0,1,3,4))
            # histogram_statistics = [skewness]

            # histo_stats = torch.stack([stat(dxw_target) for stat in histogram_statistics])
            histo_stats = skewness(dx)
            histo_stats = np.sqrt(0.5) * histo_stats[:,0,:,0,:]

            y = torch.cat([y, histo_stats], dim=1)

        if y.shape[1] != self.description.shape[0]:
            raise ValueError("Mismatch between tensor coefficient and its description.")

        Rx = DescribedTensor(x=x, y=y, descri=self.description)

        if self.c_types is not None:
            return Rx.reduce(c_type=self.c_types)

        return Rx


def init_model(model_type, shape,
               r, J, Q, wav_type, high_freq, wav_norm,
               A,
               rpad,
               qs,
               sigma2, norm_on_the_fly, standardize,
               c_types,
               estim_operator,
               histogram_moments,
               skew_redundance,
               nchunks, dtype):  # TODO. J and Q should be accepted as ints at this stage, make this function useable
    """ Initialize a scattering covariance model.

    :param model_type: moments to compute on scattering
    :param shape: input shape
    :param r: number of wavelet layers
    :param J: number of octaves for each waveelt layer
    :param Q: number of scales per octave for each wavelet layer
    :param wav_type: wavelet types for each wavelet layer
    :param high_freq: central frequency of mother wavelet for each waveelt layer, 0.5 gives important aliasing
    :param wav_norm: wavelet normalization for each waveelt layer e.g. l1, l2
        None: compute Sx = W|Wx|(t,j1,j2) and keep time axis t
        "scat": compute marginal moments on Sx: E{|Sx|^q} by time average
        "cov": compute covariance on Sx: Cov{Sx, Sx} as well as E{|Wx|} and E{|Wx|^2} by time average
        "covreduced": same as "cov" but compute reduced covariance: P Cov{Sx, Sx}, where P is the self-similar projection
        "scat+cov": both "scat" and "cov"
    :param rpad: if true, uses a reflection pad to account for edge effects
    :param qs: if model_type == 'scat' the exponents of the scattering marginal moments
    :param sigma2: a tensor of size B x N x J, wavelet power spectrum to normalize the representation with
    :param norm_on_the_fly: normalize first wavelet layer on the fly
    :param c_types: coefficient types used (None will use all of them)
    :param estim_operator: estimation operator to use
    :param nchunks: the number of chunks
    :param dtype: data precision, either float32 or float64

    :return: a torch module
    """
    B, N, T = shape

    if nchunks < B:
        batch_chunk = nchunks
        cov_chunk = 1
    else:
        batch_chunk = B
        cov_chunk = nchunks // B

    # SCATTERING MODULE
    channel_transforms = None
    Ns = [N] * (r+1)

    model = Model(model_type, qs,
                  T, r, J, Q, wav_type, high_freq, wav_norm,
                  A,
                  rpad,
                  channel_transforms, N, Ns,
                  sigma2, norm_on_the_fly, standardize,
                  estim_operator,
                  c_types,
                  cov_chunk,
                  dtype,
                  histogram_moments,
                  skew_redundance)
    model = ChunkedModule(model, batch_chunk)

    return model


def compute_sigma2(x, J, Q, wav_type, high_freq, wav_norm, rpad, nchunks, cuda):
    """ Computes power specturm sigma(j)^2 used to normalize scattering coefficients. """
    marginal_model = init_model(model_type='scat', shape=[x.shape[i] for i in [0,1,4]],
                                r=1, J=J, Q=Q,
                                wav_type=wav_type, high_freq=high_freq, wav_norm=wav_norm,
                                A=None,
                                rpad=rpad,
                                qs=[2.0],
                                sigma2=None,
                                norm_on_the_fly=False, standardize=False,
                                c_types=None,
                                estim_operator=None,
                                nchunks=nchunks, dtype=x.dtype,
                                histogram_moments=False, skew_redundance=False)
    if cuda:
        x = x.cuda()
        marginal_model = marginal_model.cuda()

    sigma2 = marginal_model(x).y.real.reshape(x.shape[0], x.shape[1], -1)  # B x N x J

    return sigma2


def analyze(x, model_type='cov',
            r=2, J=None, Q=1, wav_type='battle_lemarie', wav_norm='l1', high_freq=0.425,
            A=None,
            rpad=True,
            qs=None,
            normalize=None, keep_ps=False, sigma2=None,
            estim_operator=None,
            histogram_moments=False,
            skew_redundance=False,
            nchunks=1, cuda=False):
    """ Compute scattering based model.

    :param x: an array of shape (T, ) or (B, T) or (B, N, T)
    :param r: number of wavelet layers
    :param J: number of octaves for each wavelet layer
    :param Q: number of scales per octave for each wavelet layer
    :param wav_type: wavelet types for each wavelet layer
    :param wav_norm: wavelet normalization i.e. l1, l2 for each layer
    :param high_freq: central frequency of mother wavelet for each layer, 0.5 gives important aliasing
    :param model_type: moments to compute on scattering
        None: compute Sx = W|Wx|(t,j1,j2) and keep time axis t
        "scat": compute marginal moments on Sx: E{|Sx|^q} by time average
        "cov": compute covariance on Sx: Cov{Sx, Sx} as well as E{|Wx|} and E{|Wx|^2} by time average
        "covreduced": same as "cov" but compute reduced covariance: P Cov{Sx, Sx}, where P is the self-similar projection
        "scat+cov": both "scat" and "cov"
    :param normalize:
        None: no normalization,
        "each_ps": normalize Rx.y[b,:,:] by its power spectrum
        "batch_ps": normalize RX.y[b,:,:] by the average power spectrum over all trajectories b in the batch
    :param qs: exponent to use in a marginal model
    :param keep_ps: keep the power spectrum even after normalization
    :param estim_operator: AveragingOperator by default, but can be overwritten
    :param nchunks: nb of chunks, increase it to reduce memory usage
    :param cuda: does calculation on gpu

    :return: a DescribedTensor result
    """
    if model_type not in [None, "scat", "cov", "covreduced", "scat+cov"]:
        raise ValueError("Unrecognized model type.")
    if normalize not in [None, "each_ps", "batch_ps"]:
        raise ValueError("Unrecognized normalization.")
    if model_type == "covreduced" and normalize is None:
        raise ValueError("For covreduced model, user should provide a normalize argument.")
    if r > 2 and model_type not in [None, 'scat']:
        raise ValueError("Moments with covariance are not implemented for more than 3 convolution layers.")

    x = format_np(x)

    B, N, T = x.shape
    x_torch = format_torch(x)

    if x_torch.dtype not in [torch.float32, torch.float64]:
        x_torch = x_torch.type(torch.float32)
        print("WARNING. Casting data to float 32.")
    dtype = x_torch.dtype

    if J is None:
        J = int(np.log2(T)) - 3
    if isinstance(J, int):
        J = [J] * r
    if isinstance(Q, int):
        Q = [Q] * r
    if isinstance(wav_type, str):
        wav_type = [wav_type] * r
    if isinstance(wav_norm, str):
        wav_norm = [wav_norm] * r
    if isinstance(high_freq, float):
        high_freq = [high_freq] * r
    if qs is None:
        qs = [1.0, 2.0]

    # covreduced needs a spectrum normalization
    if normalize is not None and sigma2 is None:
        sigma2 = compute_sigma2(x_torch, J, Q, wav_type, high_freq, wav_norm, rpad, nchunks, cuda)
        if normalize == "batch_ps":
            sigma2 = sigma2.mean(0, keepdim=True)
    if sigma2 is not None and sigma2.is_complex():
        raise ValueError("Normalization should be real!.")

    # initialize model
    model = init_model(model_type=model_type, shape=x.shape,
                       r=r, J=J, Q=Q, wav_type=wav_type, high_freq=high_freq, wav_norm=wav_norm,
                       A=A,
                       rpad=rpad,
                       qs=qs,
                       sigma2=sigma2,
                       norm_on_the_fly=normalize=="each_ps", standardize=False,  # TODO. Replace norm_on_fly by False
                       estim_operator=estim_operator,
                       c_types=None,
                       nchunks=nchunks, dtype=dtype,
                       histogram_moments=histogram_moments, skew_redundance=skew_redundance)

    # compute
    if cuda:
        x_torch = x_torch.cuda()
        model = model.cuda()

    Rx = model(x_torch)

    if keep_ps and normalize is not None and model_type in ["cov", "covreduced", "scat+cov"] and estim_operator is None:
        # retrieve the power spectrum that was normalized
        for n in range(N):
            mask_ps = Rx.descri.where(c_type='ps', nl=n, nr=n)
            if mask_ps.sum() != 0:
                Rx.y[:, mask_ps, :] = Rx.y[:, mask_ps, :] * sigma2[:, n, :].reshape(sigma2.shape[0], -1, 1)

    return Rx.cpu()


def format_to_real(Rx):
    """ Transforms a complex described tensor z into a real tensor (Re z, Im z). """
    if "real" not in Rx.descri:
        raise ValueError("Described tensor should have a column indicating which coefficients are real.")
    Rx_real = Rx.reduce(real=True)
    Rx_complex = Rx.reduce(real=False)

    descri_complex_real = Rx_complex.descri.clone()
    descri_complex_imag = Rx_complex.descri.clone()
    descri_complex_real["real"] = True
    descri = Description(pd.concat([Rx_real.descri, descri_complex_real, descri_complex_imag]))

    y = torch.cat([Rx_real.y.real, Rx_complex.y.real, Rx_complex.y.imag], dim=1)

    return DescribedTensor(None, y, descri)


##################
# GENERATION
##################

class GenDataLoader(ProcessDataLoader):
    """ A data loader for generation. Caches the generated trajectories. """
    def __init__(self, *args):
        super(GenDataLoader, self).__init__(*args)
        self.default_kwargs = {}

    def dirpath(self, **kwargs):
        """ The directory path in which the generated trajectories will be stored. """
        B, N, T = kwargs['model_params']['shape']
        model_params = kwargs['model_params']
        r, J, Q, wav_type, rpad, model_type = (model_params[key] for key in ['r', 'J', 'Q', 'wav_type', 'rpad', 'model_type'])
        path_str = f"{self.model_name}_{wav_type[0]}_B{B}_N{N}_T{T}_J{J[0]}_Q1_{Q[0]}_Q2_{Q[1]}" \
                   f"_hf{kwargs['model_params']['high_freq'][0]}_rmax{r}_model_{model_type}_rpad{int(rpad)}" \
                   f"_tol{kwargs['optim_params']['tol_optim']:.2e}" \
                   f"_it{kwargs['optim_params']['it']}" \
                   + (f"_hist" if kwargs['model_params']['histogram_moments'] else "") \
                   + (f"_sk_redund" if kwargs['model_params']['skew_redundance'] else "")
        return self.dir_name / path_str.replace('.', '_').replace('-', '_')

    def generate_trajectory(self, seed, x, Rx, x0, model_params, optim_params, gpu, dirpath):
        """ Performs cached generation. """
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        np.random.seed(seed)
        filename = dirpath / str(np.random.randint(1e7, 1e8))
        if filename.is_file():
            raise OSError("File for saving this trajectory already exists.")

        x_torch = None
        if x is not None:
            x_torch = format_torch(x)

        # B, N, T = model_params['shape']
        shape_generated = torch.Size(model_params['shape'])
        # shape_generated = B, N, (T if model_params['x_past'] is None else T - model_params['x_past'].shape[-1])
        # shape_generated = torch.Size(shape_generated)

        if x_torch is not None:
            sigma2_target = compute_sigma2(x_torch, model_params['J'], model_params['Q'],
                                           model_params['wav_type'], model_params['high_freq'], model_params['wav_norm'],
                                           model_params['rpad'],
                                           model_params['nchunks'], optim_params['cuda'])
            # print("Global normalization average")
            # sigma2_target = sigma2_target.mean(1, keepdims=True)

        if model_params['sigma2'] is None:
            model_params['sigma2'] = sigma2_target.mean(0, keepdim=True)  # do a "batch_ps" normalization
        if optim_params['cuda']:
            model_params['sigma2'] = model_params['sigma2'].cuda()

        if model_params['sigma2'].is_complex():
            raise ValueError("Normalization sigma2 should be real!.")

        # initialize model
        print("Initialize model")
        model = init_model(**model_params)
        if optim_params['cuda'] and gpu is not None:
            if x_torch is not None:
                x_torch = x_torch.cuda()
            model = model.cuda()

        # prepare target representation
        if Rx is None:
            print("Preparing target representation")
            model_unit_norm = init_model(sigma2=sigma2_target, **{key:val for (key,val) in model_params.items() if key !='sigma2'})
            if optim_params['cuda'] and gpu is not None:
                model_unit_norm.cuda()
            Rx = model_unit_norm(x_torch).cpu()

        # prepare initial gaussian process
        if x0 is None:
            # infer correct x0 variance
            if x is not None:
                x0_mean, x0_var = x.mean(-1), np.var(x, axis=-1)
            else:
                x0_mean, x0_var = 0.0, 1.0

            def gen_wn(mean, std, shape, dtype):
                wn = np.random.randn(*shape)
                if dtype == torch.float32:
                    wn = np.float32(wn)
                wn -= wn.mean(axis=-1, keepdims=True)
                wn /= np.std(wn, axis=-1, keepdims=True)

                return mean + std * wn

            x0 = gen_wn(x0_mean[:,:,None], x0_var[:,:,None]**0.5, shape_generated, model_params['dtype'])

        # init loss, solver and convergence criterium
        loss = MSELossScat(J=model_params['J'][0], histogram_moments=False, wrap_avg=model_params['model_type']=='cov')
        solver_fn = Solver(shape=shape_generated, model=model, loss=loss, Rxf=Rx, x0=x0,
                           cuda=optim_params['cuda'])

        check_conv_criterion = CheckConvCriterion(solver=solver_fn, tol=optim_params['tol_optim'])

        print('Embedding: uses {} coefficients {}'.format(
            model.module.count_coefficients(),
            ' '.join(
                ['{}={}'.format(c_type, model.module.count_coefficients(c_type=c_type))
                 for c_type in model.module.description.c_type.unique()])
        ))

        method, maxfun, jac = optim_params['method'], optim_params['maxfun'], optim_params['jac']
        relative_optim, maxiter = optim_params['relative_optim'], optim_params['it']

        tic = time()
        # Decide if the function provides gradient or not
        func = solver_fn.joint if jac else solver_fn.function
        try:
            res = scipy.optimize.minimize(
                func, x0.ravel(), method=method, jac=jac, callback=check_conv_criterion,
                options={'ftol': 1e-24, 'gtol': 1e-24, 'maxiter': maxiter, 'maxfun': maxfun}
            )
            loss_tmp, x_opt, it, msg = res['fun'], res['x'], res['nit'], res['message']
            if not res.success and it == maxiter:
                raise MaxIteration()
        except MaxIteration:
            print("MAX ITERATIONS REACHED. Optimization failed.")
            return
        except SmallEnoughException:  # raised by check_conv_criterion
            print('SmallEnoughException')
            x_opt = check_conv_criterion.result
            it = check_conv_criterion.counter
            msg = "SmallEnoughException"

        toc = time()

        flo, fgr = solver_fn.joint(x_opt)
        flo, fgr = flo, np.max(np.abs(fgr))
        x_synt = x_opt.reshape(x0.shape)

        if not isinstance(msg, str):
            msg = msg.decode("ASCII")

        print('Optimization Exit Message : ' + msg)
        print(f"found parameters in {toc - tic:0.2f}s, {it} iterations -- {it / (toc - tic):0.2f}it/s")
        print(f"    abs sqrt error {flo ** 0.5:.2E}")
        print(f"    relative gradient error {fgr:.2E}")
        print(f"    loss0 {np.sqrt(solver_fn.loss0):.2E}")

        if model_params['dtype'] == torch.float32:
            x_synt = np.float32(x_synt)

        return x_synt  # S x N x T

    def worker(self, i, **kwargs):
        cuda = kwargs['optim_params']['cuda']
        gpus = kwargs['optim_params']['gpus']
        seed = kwargs['optim_params']['seed'][i]
        x0 = kwargs['optim_params']['x0']
        if x0 is not None:
            x0 = kwargs['optim_params']['x0'][:,i,:,:]
        kwargs['seed'] = seed
        kwargs['x0'] = x0
        if cuda and gpus is None:
            kwargs['gpu'] = '0'
        elif cuda and gpus is not None:
            kwargs['gpu'] = str(gpus[i % len(gpus)])
        else:
            kwargs['gpu'] = None
        try:
            x = self.generate_trajectory(**kwargs)
            if x is None:
                return
            fname = f"{np.random.randint(1e7, 1e8)}.npy"
            np.save(str(kwargs['dirpath'] / fname), x)
            print(f"Saved: {kwargs['dirpath'].name}/{fname}")
        except ValueError as e:
            print(e)
            return


def generate(x=None, Rx=None, x0=None, S=1,
             sigma2=None, standardize=False, norm_on_the_fly=False,
             shape=None,
             model_type='cov', r=2, J=None, Q=1, wav_type='battle_lemarie', wav_norm='l1', high_freq=0.425,
             A=None,
             rpad=True,
             qs=None,
             c_types=None,
             nchunks=1, it=10000,
             tol_optim=5e-4,
             seed=None,
             histogram_moments=False,
             skew_redundance=False,
             generated_dir=None, exp_name=None,
             cuda=False, gpus=None, num_workers=1):
    """ Generate new realizations of x from a scattering covariance model.
    We first compute the scattering covariance representation of x and then sample it using gradient descent.

    :param x: an array of shape (T, ) or (B, T) or (B, N, T)
    :param Rx: overwrite target representation
    :param x0: array of shape (B, N, T), overwrite initial signal in gradient descent
    :param S: number of syntheses
    :param r: number of wavelet layers
    :param J: number of octaves for each wavelet layer
    :param Q: number of scales per octave for each wavelet layer
    :param wav_type: wavelet type
    :param wav_norm: wavelet normalization i.e. l1, l2
    :param high_freq: central frequency of mother wavelet, 0.5 gives important aliasing
    :param model_type: moments to compute on scattering, ex: None, 'scat', 'cov', 'covreduced'
    :param qs: if model_type == 'marginal' the exponents of the scattering marginal moments
    :param c_types: coefficient types used (None will use all of them)
    :param nchunks: nb of chunks, increase it to reduce memory usage
    :param it: maximum number of gradient descent iteration
    :param tol_optim: error below which gradient descent stops
    :param seed: None, int or list, random seed of original white noise in gradient descent for each synthesis
    :param generated_dir: the directory in which the generated dir will be located
    :param exp_name: experience name
    :param cuda: does calculation on gpu
    :param gpus: a list of gpus to use
    :param num_workers: number of generation workers

    :return: a DescribedTensor result
    """
    if x is None and shape is None:
        raise ValueError("Should provide the shape of data to generate.")
    if x0 is not None and x0.ndim != 4:
        raise ValueError("If provided, x0 should be of shape (B,nruns,N,T).")
    if x is None and sigma2 is None:
        raise ValueError("Without x, a normalization sigma2 should be provided.")

    x = format_np(x)

    # infer shape to use
    if shape is None:
        if x is not None:
            shape = x.shape
        # elif x0 is not None and x_past is not None:
        #     shape = x0[:,0,:,:].shape + x_past.shape
        else:
            raise ValueError("Should provide the shape of data to generate.")

    B, N, T = shape

    if shape is not None and len(shape) != 3:
        raise ValueError("Shape to generate should be of type (B,N,T).")
    if x is None and Rx is None:
        raise ValueError("Should provide data or a representation to generate data.")
    if x0 is not None and x is not None and x0[:,0,:,:].shape != x.shape:
        raise ValueError(f"If specified, x0 should be of shape {x.shape}")
    # if x_past is not None and x0 is not None:
    #     assert x_past.shape[-1] + x0.shape[-1] == T

    if J is None:
        J = int(np.log2(T)) - 5
    def make_list(x, type, r): return [x] * r if x is None or isinstance(x, type) else x
    J = make_list(J, int, r)
    Q = make_list(Q, int, r)
    wav_type = make_list(wav_type, str, r)
    wav_norm = make_list(wav_norm, str, r)
    high_freq = make_list(high_freq, float, r)
    seed = make_list(seed, int, S)  # TODO. Use seed for description of dirpath, otherwise no possibility to retrieve same seeds
    if qs is None:
        qs = [1.0, 2.0]
    if generated_dir is None:
        generated_dir = Path(__file__).parents[1] / '_cached_dir'

    # use a GenDataLoader to cache trajectories
    dtld = GenDataLoader(exp_name or 'gen_scat_cov', generated_dir, num_workers)

    # MODEL params
    model_params = {
        'shape': shape,
        'r': r, 'J': J, 'Q': Q,
        'wav_type': wav_type,  # 'battle_lemarie' 'morlet' 'shannon'
        'high_freq': high_freq,  # 0.323645 or 0.425
        'wav_norm': wav_norm,
        'model_type': model_type, 'qs': qs,
        'A': A,
        'rpad': rpad,
        'sigma2': sigma2, 'norm_on_the_fly': norm_on_the_fly,
        'standardize': standardize,
        'c_types': c_types,
        'nchunks': nchunks,
        'estim_operator': None,
        'histogram_moments': histogram_moments,
        'skew_redundance': skew_redundance,
        'dtype': torch.float64
    }

    # OPTIM params
    optim_params = {
        'it': it,
        'cuda': cuda,
        'gpus': gpus,
        'relative_optim': False,
        'maxfun': 2e6,
        'method': 'L-BFGS-B',
        'jac': True,  # origin of gradient, True: provided by solver, else estimated
        'tol_optim': tol_optim,
        'seed': seed,
        'x0': x0,
    }

    # multi-processed generation
    x_gen = dtld.load(R=S, n_files=int(np.ceil(S/B)), x=x, Rx=Rx,
                      model_params=model_params,
                      optim_params=optim_params).x

    return x_gen


##################
# VIZUALIZE
##################

COLORS = ['skyblue', 'coral', 'lightgreen', 'darkgoldenrod', 'mediumpurple', 'red', 'purple', 'black',
          'paleturquoise'] + ['orchid'] * 20


def bootstrap_variance_complex(x, n_points, n_samples):
    """ Estimate variance of tensor x along last axis using bootstrap method. """
    # sample data uniformly
    sampling_idx = np.random.randint(low=0, high=x.shape[-1], size=(n_samples, n_points))
    sampled_data = x[..., sampling_idx]

    # computes mean
    mean = sampled_data.mean(-1).mean(-1)

    # computes bootstrap variance
    var = (torch.abs(sampled_data.mean(-1) - mean[..., None]).pow(2.0)).sum(-1) / (n_samples - 1)

    return mean, var


def error_arg(z_mod, z_err, eps=1e-12):
    """ Transform an error on |z| into an error on Arg(z). """
    z_mod = np.maximum(z_mod, eps)
    return np.arctan(z_err / z_mod / np.sqrt(np.clip(1 - z_err ** 2 / z_mod ** 2, 1e-6, 1)))


def get_variance(z):
    """ Compute complex variance of a sequence of complex numbers z1, z2, ... """
    B = z.shape[0]
    return torch.abs(z - z.mean(0, keepdim=True)).pow(2.0).sum(0).div(B-1).div(B)


def plot_raw(Rx, ax, legend=False):
    """ Raw plot of the coefficients contained in Rx. """
    if 'j' in Rx.descri.columns or 'j2' not in Rx.descri.columns:
        raise ValueError("Raw plot of DescribedTensor only implemented for ouput of cov model.")
    descri = Description(Rx.descri.reindex(columns=['c_type', 'real', 'low',
                                                    'nl', 'nr', 'q', 'rl', 'rr', 'scl', 'scr',
                                                    'jl1', 'jr1', 'j2', 'al', 'ar']))
    Rx = DescribedTensor(x=None, descri=descri, y=Rx.y).mean_batch()
    if Rx.y.is_complex():
        Rx = format_to_real(Rx)
    Rx = Rx.sort()
    ctypes = Rx.descri['c_type'].unique()

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, ctype in enumerate(ctypes):
        mask_real = np.where(Rx.descri.where(c_type=ctype, real=True))[0]
        mask_imag = np.where(Rx.descri.where(c_type=ctype, real=False))[0]
        ax.axvspan(mask_real.min(), mask_real.max(), color=cycle[i], label=ctype if legend else None, alpha=0.7)
        if mask_imag.size > 0:
            ax.axvspan(mask_imag.min(), mask_imag.max(), color=cycle[i], alpha=0.4)
        ax.axhline(0.0, color='black', linewidth=0.02)
    ax.plot(Rx.y[0, :, 0], linewidth=0.7)
    if legend:
        ax.legend()
    return Rx.descri


def plot_marginal_moments(Rxs, estim_bar=False,
                          axes=None, labels=None,
                          colors=None, linewidth=3.0, fontsize=30):
    """ Plot the marginal moments
        - (wavelet power spectrum) sigma^2(j)
        - (sparsity factors) s^2(j)

    :param Rxs: DescribedTensor or list of DescribedTensor
    :param estim_bar: display estimation error due to estimation on several realizations
    :param axes: custom axes: array of size 2
    :param labels: list of labels for each model output
    :param linewidth: curve linewidth
    :param fontsize: labels fontsize
    :return:
    """
    if isinstance(Rxs, DescribedTensor):
        Rxs = [Rxs]
    if labels is not None and len(Rxs) != len(labels):
        raise ValueError("Invalid number of labels")
    if axes is not None and axes.size != 2:
        raise ValueError("The axes provided to plot_marginal_moments should be an array of size 2.")
    colors = colors or COLORS

    labels = labels or [''] * len(Rxs)
    axes = None if axes is None else axes.ravel()

    def plot_exponent(js, ax, label, color, y, y_err):
        plt.sca(ax)
        plt.plot(-js, y, label=label, linewidth=linewidth, color=color)
        if not estim_bar:
            plt.scatter(-js, y, marker='+', s=200, linewidth=linewidth, color=color)
        else:
            eb = plt.errorbar(-js, y, yerr=y_err, capsize=4, color=color, fmt=' ')
            eb[-1][0].set_linestyle('--')
        plt.yscale('log', base=2)
        plt.xlabel(r'$-j$', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(-js, [fr'$-{j + 1}$' for j in js], fontsize=fontsize)

    if axes is None:
        ax1 = plt.subplot2grid((1, 2), (0, 0))
        ax2 = plt.subplot2grid((1, 2), (0, 1))
        axes = [ax1, ax2]

    for i_lb, (lb, Rx) in enumerate(zip(labels, Rxs)):
        if 'c_type' not in Rx.descri.columns:
            raise ValueError("The model output does not have the moments.")
        js = np.unique(Rx.descri.reduce(low=False).jl1.dropna())

        has_power_spectrum = 2.0 in Rx.descri.q.values
        has_sparsity = 1.0 in Rx.descri.q.values

        # averaging on the logs may have strange behaviors because of the strict convexity of the log
        if has_power_spectrum:
            Wx2_nj = Rx.select(rl=1, c_type=['ps', 'scat', 'spars', 'marginal'], q=2.0, low=False)[:, :, 0]
            if Wx2_nj.is_complex():
                Wx2_nj = Wx2_nj.real
            logWx2_n = torch.log2(Wx2_nj)

            # little subtlety here, we plot the log on the mean but the variance is the variance on the log
            logWx2_err = get_variance(logWx2_n) ** 0.5
            logWx2 = torch.log2(Wx2_nj.mean(0))
            # logWx2 -= logWx2[0].item()
            ps_norm_rectifier = 2 * 0.5 * np.arange(logWx2.shape[-1])
            plot_exponent(js, axes[0], lb, colors[i_lb], 2.0 ** (logWx2 + ps_norm_rectifier),
                          np.log(2) * logWx2_err * 2.0 ** logWx2)
            a, b = axes[0].get_ylim()
            if i_lb == len(labels):
                axes[0].set_ylim(min(a, 2 ** (-2)), max(b, 2 ** 2))
            if i_lb == len(labels) - 1 and any([lb != '' for lb in labels]):
                plt.legend(prop={'size': 15})
            plt.title(r'Power Spectrum $\Phi_2$', fontsize=fontsize)

            if has_sparsity:
                Wx1_nj = Rx.select(rl=1, c_type=['ps', 'scat', 'spars', 'marginal'], q=1.0, low=False)[:, :, 0]
                if Wx1_nj.is_complex():
                    Wx1_nj = Wx1_nj.real
                logWx1_nj = torch.log2(Wx1_nj)

                logWxs_n = 2 * logWx1_nj - logWx2_n.mean(0, keepdims=True)
                logWxs_err = get_variance(logWxs_n) ** 0.5
                logWxs = torch.log2(Wx1_nj.mean(0).pow(2.0) / Wx2_nj.mean(0))
                plot_exponent(js, axes[1], lb, colors[i_lb], 2.0 ** logWxs, np.log(2) * logWxs_err * 2.0 ** logWxs)
                a, b = axes[1].get_ylim()
                if i_lb == len(labels) - 1:
                    axes[1].set_ylim(min(2 ** (-2), a), 1.0)
                if i_lb == len(labels) - 1 and any([lb != '' for lb in labels]):
                    plt.legend(prop={'size': 15})
                plt.title(r'Sparsity factors $\Phi_1$', fontsize=fontsize)

    for ax in axes.ravel():
        ax.grid(True)


def plot_phase_envelope_spectrum(Rxs, estim_bar=False, self_simi_bar=False, theta_threshold=0.005,
                                 sigma2=None,
                                 axes=None, labels=None, colors=None, fontsize=30, single_plot=False, ylim=0.1):
    """ Plot the phase-envelope cross-spectrum C_{W|W|}(a) as two graphs : |C_{W|W|}| and Arg(C_{W|W|}).

    :param Rxs: DescribedTensor or list of DescribedTensor
    :param estim_bar: display estimation error due to estimation on several realizations
    :param self_simi_bar: display self-similarity error, it is a measure of scale regularity
    :param theta_threshold: rules phase instability
    :param sigma2: override normalization factors
    :param axes: custom axes: array of size 2
    :param labels: list of labels for each model output
    :param fontsize: labels fontsize
    :param single_plot: output all DescribedTensor on a single plot
    :param ylim: above y limit of modulus graph
    :return:
    """
    if isinstance(Rxs, DescribedTensor):
        Rxs = [Rxs]
    if labels is not None and len(Rxs) != len(labels):
        raise ValueError("Invalid number of labels")
    colors = colors or COLORS

    labels = labels or [''] * len(Rxs)
    columns = Rxs[0].descri.columns
    J = Rxs[0].descri.j.max() if 'j' in columns else Rxs[0].descri.jl1.max()

    c_wmw = torch.zeros(len(labels), J-1, dtype=Rxs[0].y.dtype)
    err_estim = torch.zeros(len(labels), J-1)
    err_self_simi = torch.zeros(len(labels), J-1)

    for i_lb, Rx in enumerate(Rxs):

        if 'phaseenv' not in Rx.descri.c_type.values:
            continue

        model_type = 'general'
        if 'a' in Rx.descri.columns:
            model_type = 'covreduced'

        B = Rx.y.shape[0]

        norm2 = sigma2
        if sigma2 is None:
            norm2 = Rx.select(rl=1, rr=1, q=2, low=False).real.mean(0)[None, :, 0]  # power spectrum averaged on batch
            if Rx.descri['nl'].iloc[0] != Rx.descri['nr'].iloc[0]:
                print("WARNING. Carefull, sigma2 should be given for left and right independently.")

        for a in range(1, J):
            if model_type == 'covreduced':
                c_mwm_n = Rx.select(c_type='phaseenv', a=a, low=False)
                assert c_mwm_n.shape[1] == 1, f"ERROR. Should be selecting 1 coefficient but got {c_mwm_n.shape[1]}"
                c_mwm_n = c_mwm_n[:, 0, 0]

                c_wmw[i_lb, a-1] = c_mwm_n.mean(0)
                err_estim[i_lb, a-1] = get_variance(c_mwm_n).pow(0.5)
            else:
                c_mwm_nj = torch.zeros(B, J-a, dtype=Rx.y.dtype)
                for j1 in range(a, J):
                    coeff = Rx.select(c_type='phaseenv', jl1=j1, jr1=j1-a, low=False)
                    assert coeff.shape[1] == 1, f"ERROR. Should be selecting 1 coefficient but got {coeff.shape[1]}"
                    coeff = coeff[:, 0, 0]
                    coeff /= norm2[:, j1, ...].pow(0.5) * norm2[:, j1-a, ...].pow(0.5)
                    c_mwm_nj[:, j1-a] = coeff

                # the mean in j of the variance of time estimators  # TODO. Unify with a unique variance function
                c_wmw[i_lb, a-1] = c_mwm_nj.mean(0).mean(0)
                err_self_simi_n = (torch.abs(c_mwm_nj).pow(2.0).mean(1) - torch.abs(c_mwm_nj.mean(1)).pow(2.0)) / \
                                  c_mwm_nj.shape[1]
                err_self_simi[i_lb, a-1] = err_self_simi_n.mean(0).pow(0.5)
                err_estim[i_lb, a-1] = get_variance(c_mwm_nj.mean(1)).pow(0.5)

    c_wmw_mod, cwmw_arg = np.abs(c_wmw.numpy()), np.angle(c_wmw.numpy())
    err_self_simi, err_estim = to_numpy(err_self_simi), to_numpy(err_estim)
    err_self_simi_arg, err_estim_arg = error_arg(c_wmw_mod, err_self_simi), error_arg(c_wmw_mod, err_estim)

    # phase instability at z=0
    for z_arg in [cwmw_arg, err_self_simi_arg, err_estim_arg]:
        z_arg[c_wmw_mod < theta_threshold] = 0.0

    def plot_modulus(i_lb, label, color, y, y_err_estim, y_err_self_simi):
        a_s = np.arange(1, J)
        plt.plot(a_s, y, color=color or 'green', label=label)
        if not estim_bar and not self_simi_bar:
            plt.scatter(a_s, y, color=color or 'green', marker='+')
        if self_simi_bar:
            plot_x_offset = -0.07 if estim_bar else 0.0
            plt.errorbar(a_s + plot_x_offset, y, yerr=y_err_self_simi, capsize=4, color=color, fmt=' ')
        if estim_bar:
            plot_x_offset = 0.07 if self_simi_bar else 0.0
            eb = plt.errorbar(a_s + plot_x_offset, y, yerr=y_err_estim, capsize=4, color=color, fmt=' ')
            eb[-1][0].set_linestyle('--')
        plt.title('Cross-spectrum' + '\n' + r'$|\Phi_3|$', fontsize=fontsize)
        plt.axhline(0.0, linewidth=0.7, color='black')
        plt.xticks(np.arange(1, J),
                   [(rf'${j}$' if j % 2 == 1 else '') for j in np.arange(1, J)], fontsize=fontsize)
        plt.xlabel(r'$a$', fontsize=fontsize)
        plt.ylim(-0.02, ylim)
        if i_lb == 0:
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.yticks(fontsize=fontsize)
        plt.locator_params(axis='y', nbins=5)

    def plot_phase(label, color, y, y_err_estim, y_err_self_simi):
        a_s = np.arange(1, J)
        plt.plot(a_s, y, color=color, label=label)
        if not estim_bar and not self_simi_bar:
            plt.scatter(a_s, y, color=color, marker='+')
        if self_simi_bar:
            plot_x_offset = -0.07 if estim_bar else 0.0
            plt.errorbar(a_s + plot_x_offset, y, yerr=y_err_self_simi, capsize=4, color=color, fmt=' ')
        if estim_bar:
            plot_x_offset = 0.07 if self_simi_bar else 0.0
            eb = plt.errorbar(a_s + plot_x_offset, y, yerr=y_err_estim, capsize=4, color=color, fmt=' ')
            eb[-1][0].set_linestyle('--')
        plt.xticks(np.arange(1, J), [(rf'${j}$' if j % 2 == 1 else '') for j in np.arange(1, J)], fontsize=fontsize)
        plt.yticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi],
                   [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=fontsize)
        plt.axhline(0.0, linewidth=0.7, color='black')
        plt.xlabel(r'$a$', fontsize=fontsize)
        plt.title('Cross-spectrum' + '\n' + r'Arg $\Phi_3$', fontsize=fontsize)

    if axes is None:
        plt.figure(figsize=(5, 10) if single_plot else (len(labels) * 5, 10))
        ax_mod = plt.subplot2grid((2, 1), (0, 0))
        ax_mod.yaxis.set_tick_params(which='major', direction='in', width=1.5, length=7)
    else:
        plt.sca(axes[0])
    for i_lb, lb in enumerate(labels):
        plot_modulus(i_lb, lb, colors[i_lb], c_wmw_mod[i_lb], err_estim[i_lb], err_self_simi[i_lb])
        if i_lb == len(labels) - 1 and any([lb != '' for lb in labels]):
            plt.legend(prop={'size': 15})

    if axes is None:
        plt.subplot2grid((2, 1), (1, 0))
    else:
        plt.sca(axes[1])
    for i_lb, lb in enumerate(labels):
        plot_phase(lb, colors[i_lb], cwmw_arg[i_lb], err_estim_arg[i_lb], err_self_simi_arg[i_lb])

    if axes is None:
        plt.tight_layout()

    for ax in axes.ravel():
        ax.grid(True)


def plot_scattering_spectrum(Rxs, estim_bar=False, self_simi_bar=False, bootstrap=True, theta_threshold=0.01,
                             sigma2=None,
                             axes=None, labels=None, colors=None, fontsize=40, ylim=2.0, d=1):
    """ Plot the scattering cross-spectrum C_S(a,b) as two graphs : |C_S| and Arg(C_S).

    :param Rxs: DescribedTensor or list of DescribedTensor
    :param estim_bar: display estimation error due to estimation on several realizations
    :param self_simi_bar: display self-similarity error, it is a measure of scale regularity
    :param theta_threshold: rules phase instability
    :param sigma2: override normalization factors
    :param axes: custom axes: array of size 2 x labels
    :param labels: list of labels for each model output
    :param fontsize: labels fontsize
    :param ylim: above y limit of modulus graph
    :return:
    """
    if isinstance(Rxs, DescribedTensor):
        Rxs = [Rxs]
    if labels is not None and len(Rxs) != len(labels):
        raise ValueError("Invalid number of labels.")
    if axes is not None and axes.size != 2 * len(Rxs):
        raise ValueError(f"Existing axes must be provided as an array of size {2 * len(Rxs)}")
    colors = colors or COLORS

    axes = None if axes is None else axes.reshape(2, len(Rxs))

    labels = labels or [''] * len(Rxs)
    i_graphs = np.arange(len(labels))

    # TODO. Following lines assume all Rx are obtained from same models.
    columns = Rxs[0].descri.columns
    J = Rxs[0].descri.j.max() if 'j' in columns else Rxs[0].descri.jl1.max()

    cs = torch.zeros(len(labels), J-1, J-1, dtype=Rxs[0].y.dtype)
    err_estim = torch.zeros(len(labels), J-1, J-1)
    err_self_simi = torch.zeros(len(labels), J-1, J-1)

    for i_lb, (Rx, lb, color) in enumerate(zip(Rxs, labels, colors)):

        if 'envelope' not in Rx.descri.c_type.values:
            continue

        model_type = 'general'
        if 'b' in Rx.descri.columns:
            model_type = 'covreduced'

        if self_simi_bar and model_type == 'covreduced':
            raise ValueError("Impossible to output self-similarity error on covreduced model. Use a cov model instead.")

        B = Rx.y.shape[0]

        norm2 = sigma2
        if sigma2 is None:
            norm2 = Rx.select(rl=1, rr=1, q=2, low=False).real.mean(0)[None, :, 0]  # power spectrum averaged on batch
            if Rx.descri['nl'].iloc[0] != Rx.descri['nr'].iloc[0]:
                print("WARNING. Carefull, sigma2 should be given for left and right independently.")

        for (a, b) in product(range(J-1), range(-J+1, 0)):
            if a - b >= J:
                continue

            # prepare covariances
            if model_type == "covreduced":
                coeff_ab = Rx.select(c_type='envelope', a=a, b=b, low=False)
                assert coeff_ab.shape[1] == 1, f"ERROR. Should be selecting 1 coefficient but got {coeff_ab.shape[1]}"
                coeff_ab = coeff_ab[:, 0, 0]
                cs[i_lb, a, J-1+b] = coeff_ab.mean(0)
            else:
                cs_nj = torch.zeros(B, J+b-a, dtype=Rx.y.dtype)
                for j1 in range(a, J+b):
                    coeff = Rx.select(c_type='envelope', jl1=j1, jr1=j1-a, j2=j1-b, low=False)
                    assert coeff.shape[1] == 1, f"ERROR. Should be selecting 1 coefficient but got {coeff.shape[1]}"
                    coeff = coeff[:, 0, 0]
                    coeff /= norm2[:, j1, ...].pow(0.5) * norm2[:, j1 - a, ...].pow(0.5)
                    cs_nj[:, j1 - a] = coeff

                cs_j = cs_nj.mean(0)
                cs[i_lb, a, J-1+b] = cs_j.mean(0)
                if b == -J+a+1:
                    err_self_simi[i_lb, a, J-1+b] = 0.0
                else:
                    err_self_simi[i_lb, a, J-1+b] = torch.abs(cs_j - cs_j.mean(0, keepdim=True)) \
                        .pow(2.0).sum(0).div(J+b-a-1).pow(0.5)
                # compute estimation error
                if bootstrap:
                    # mean, var = bootstrap_variance_complex(cs_nj.transpose(0, 1), cs_nj.shape[0], 20000)
                    mean, var = bootstrap_variance_complex(cs_nj.mean(1), cs_nj.shape[0], 20000)
                    err_estim[i_lb, a, J-1+b] = var.pow(0.5)
                else:
                    err_estim[i_lb, a, J-1+b] = (torch.abs(cs_nj).pow(2.0).mean(0) -
                                                 torch.abs(cs_nj.mean(0)).pow(2.0)) / (B - 1)

    cs, cs_mod, cs_arg = cs.numpy(), np.abs(cs.numpy()), np.angle(cs.numpy())
    err_self_simi, err_estim = to_numpy(err_self_simi), to_numpy(err_estim)
    err_self_simi_arg, err_estim_arg = error_arg(cs_mod, err_self_simi), error_arg(cs_mod, err_estim)

    # power spectrum normalization
    bs = np.arange(-J + 1, 0)[None, :] * d
    cs_mod /= (2.0 ** bs)
    err_self_simi /= (2.0 ** bs)
    err_estim /= (2.0 ** bs)

    # phase instability at z=0
    for z_arg in [cs_arg, err_self_simi_arg, err_estim_arg]:
        z_arg[cs_mod < theta_threshold] = 0.0

    def plot_modulus(label, y, y_err_estim, y_err_self_simi, title):
        for a in range(J-1):
            bs = np.arange(-J+1+a, 0)
            line = plt.plot(bs, y[a, a:], label=label if a == 0 else '')
            color = line[-1].get_color()
            if not estim_bar and not self_simi_bar:
                plt.scatter(bs, y[a, a:], marker='+')
            if self_simi_bar:
                plot_x_offset = -0.07 if self_simi_bar else 0.0
                plt.errorbar(bs + plot_x_offset, y[a, a:],
                             yerr=y_err_self_simi[a, a:], capsize=4, color=color, fmt=' ')
            if estim_bar:
                plot_x_offset = 0.07 if self_simi_bar else 0.0
                eb = plt.errorbar(bs + plot_x_offset, y[a, a:],
                                  yerr=y_err_estim[a, a:], capsize=4, color=color, fmt=' ')
                eb[-1][0].set_linestyle('--')
        plt.axhline(0.0, linewidth=0.7, color='black')
        plt.xticks(np.arange(-J + 1, 0), [(rf'${b}$' if b % 2 == 1 else '') for b in np.arange(-J+1, 0)],
                   fontsize=fontsize)
        plt.xlabel(r'$b$', fontsize=fontsize)
        plt.ylim(-0.02, ylim)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.yticks(fontsize=fontsize)
        plt.locator_params(axis='x', nbins=J - 1)
        plt.locator_params(axis='y', nbins=5)
        if title:
            plt.title('Cross-spectrum' + '\n' + r'$|\Phi_4|$', fontsize=fontsize)
        if label != '':
            plt.legend(prop={'size': 15})

    def plot_phase(y, y_err_estim, y_err_self_simi, title):
        for a in range(J-1):
            bs = np.arange(-J+1+a, 0)
            line = plt.plot(bs, y[a, a:], label=fr'$a={a}$')
            color = line[-1].get_color()
            if not estim_bar and not self_simi_bar:
                plt.scatter(bs, y[a, a:], marker='+')
            if self_simi_bar:
                plot_x_offset = -0.07 if estim_bar else 0.0
                plt.errorbar(bs + plot_x_offset, y[a, a:],
                             yerr=y_err_self_simi[a, a:], capsize=4, color=color, fmt=' ')
            if estim_bar:
                plot_x_offset = 0.07 if self_simi_bar else 0.0
                eb = plt.errorbar(bs + plot_x_offset, y[a, a:],
                                  yerr=y_err_estim[a, a:], capsize=4, color=color, fmt=' ')
                eb[-1][0].set_linestyle('--')
        plt.xticks(np.arange(-J+1, 0), [(rf'${b}$' if b % 2 == 1 else '') for b in np.arange(-J+1, 0)],
                   fontsize=fontsize)
        plt.yticks(np.arange(-2, 3) * np.pi / 8,
                   [r'$-\frac{\pi}{4}$', r'$-\frac{\pi}{8}$', r'$0$', r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$'],
                   fontsize=fontsize)
        plt.axhline(0.0, linewidth=0.7, color='black')
        plt.xlabel(r'$b$', fontsize=fontsize)
        if title:
            plt.title('Cross-spectrum' + '\n' + r'Arg$\Phi_4$', fontsize=fontsize)

    if axes is None:
        plt.figure(figsize=(max(len(labels), 5) * 3, 10))
    for i_lb, lb in enumerate(labels):
        if axes is not None:
            plt.sca(axes[0, i_lb])
            ax_mod = axes[0, i_lb]
        else:
            ax_mod = plt.subplot2grid((2, np.unique(i_graphs).size), (0, i_graphs[i_lb]))
        ax_mod.yaxis.set_tick_params(which='major', direction='in', width=1.5, length=7)
        ax_mod.yaxis.set_label_coords(-0.18, 0.5)
        plot_modulus(lb, cs_mod[i_lb], err_estim[i_lb], err_self_simi[i_lb], i_lb == 0)

    for i_lb, lb in enumerate(labels):
        if axes is not None:
            plt.sca(axes[1, i_lb])
            ax_ph = axes[1, i_lb]
        else:
            ax_ph = plt.subplot2grid((2, np.unique(i_graphs).size), (1, i_graphs[i_lb]))
        plot_phase(cs_arg[i_lb], err_estim_arg[i_lb], err_self_simi_arg[i_lb], i_lb == 0)
        if i_lb == 0:
            ax_ph.yaxis.set_tick_params(which='major', direction='in', width=1.5, length=7)

    if axes is None:
        plt.tight_layout()
        leg = plt.legend(loc='upper center', ncol=1, fontsize=35, handlelength=1.0, labelspacing=1.0,
                         bbox_to_anchor=(1.3, 2.25, 0, 0))
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)

    for ax in axes.ravel():
        ax.grid(True)


def plot_dashboard(Rxs, estim_bar=False, self_simi_bar=False, bootstrap=True, theta_threshold=None,
                   sigma2=None,
                   labels=None, colors=None,
                   linewidth=3.0, fontsize=20, ylim_phase=0.1, ylim_modulus=3.0,
                   figsize=None, axes=None):
    """ Plot the scattering covariance dashboard for multi-scale processes composed of:
        - (wavelet power spectrum) sigma^2(j)
        - (sparsity factors) s^2(j)
        - (phase-envelope cross-spectrum) C_{W|W|}(a) as two graphs : |C_{W|W|}| and Arg(C_{W|W|})
        - (scattering cross-spectrum) C_S(a,b) as two graphs : |C_S| and Arg(C_S)

    :param Rxs: DescribedTensor or list of DescribedTensor
    :param estim_bar: display estimation error due to estimation on several realizations
    :param self_simi_bar: display self-similarity error, it is a measure of scale regularity
    :param bootstrap: time variance computation method
    :param theta_threshold: rules phase instability
    :param sigma2: override normalization factors
    :param labels: list of labels for each model output
    :param linewidth: lines linewidth
    :param fontsize: labels fontsize
    :param ylim_phase: graph ylim for the phase
    :param ylim_modulus: graph ylim for the modulus
    :param figsize: figure size
    :param axes: custom array of axes, should be of shape (2, 2 + nb of representation to plot)
    :return:
    """
    if theta_threshold is None:
        theta_threshold = [0.005, 0.1]
    if isinstance(Rxs, DescribedTensor):
        Rxs = [Rxs]
    for Rx in Rxs:
        if 'nl' not in Rx.descri.columns:
            Rx.descri = Description(Model.make_description_compatible(Rx.descri))
        ns_unique = Rx.descri[['nl', 'nr']].dropna().drop_duplicates()
        if ns_unique.shape[0] > 1:
            raise ValueError("Plotting functions do not support multi-variate representation other than "
                             "univariate or single pair.")

    colors = colors or COLORS

    if axes is None:
        _, axes = plt.subplots(2, 2 + len(Rxs), figsize=figsize or (12+2*(len(Rxs)-1),8))

    # marginal moments sigma^2 and s^2
    plot_marginal_moments(Rxs, estim_bar, axes[:, 0], labels, colors, linewidth, fontsize)

    # phase-envelope cross-spectrum
    plot_phase_envelope_spectrum(Rxs, estim_bar, self_simi_bar, theta_threshold[0], sigma2,
                                 axes[:, 1], labels, colors, fontsize, False, ylim_phase)

    # scattering cross spectrum
    plot_scattering_spectrum(Rxs, estim_bar, self_simi_bar, bootstrap, theta_threshold[1], sigma2,
                             axes[:, 2:], labels, colors, fontsize, ylim_modulus)

    plt.tight_layout()

    return axes
