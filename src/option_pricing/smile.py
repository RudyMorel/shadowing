import numpy as np
import matplotlib.pyplot as plt

from src.utils import lighten_color
from src.option_pricing import price_BS


class Smile:

    def __init__(self, vol, Ts, Ks=None, rMness=None, date=None):
        """ Ts: 1d array, vol: list of 1d arrays, Ks: list of 1d arrays, Ms: list of 1d arrays """
        self.date = date

        self.vol, self.Ts, Ks, rMness = self.format_input(vol, Ts, Ks, rMness)

        self.vol_atm = self.extract_vol_atm(self.vol, Ks, rMness)  # 1d array

        self.Ks, self.rMness = self.get_x_coordinates(self.vol_atm, self.Ts, Ks, rMness)

        self.prices = [price_BS(Ks, T / 252, sig, S0=100.0, r=0.0) for (Ks, T, sig) in zip(self.Ks, self.Ts, self.vol)]

    def reduce_iTs(self, iTs):
        """ Return the smile that contains only the maturities selected through their index. """
        new_vol = [self.vol[iT] for iT in iTs]
        new_Ts = [self.Ts[iT] for iT in iTs]
        new_Ks = [self.Ks[iT] for iT in iTs]
        return Smile(vol=new_vol, Ts=new_Ts, Ks=new_Ks)

    @staticmethod
    def format_input(vol, Ts, Ks, rMness):
        if Ks is None and rMness is None:
            raise ValueError("Smile object should have Ks or rMness specified.")
        Ts = np.array(Ts)

        def format(x):
            if isinstance(x, list):
                x = [np.array(arr, dtype=np.float64) for arr in x]
            if isinstance(x, np.ndarray) and x.dtype != np.float64:
                x = x.astype(np.float64)
            if isinstance(x, np.ndarray) and x.ndim == 2:
                x = list(x)
            if (isinstance(x, np.ndarray) and x.ndim == 1) or (
                    isinstance(x, list) and not isinstance(x[0], np.ndarray)):
                x = [np.array(x)] * Ts.size
            return x

        vol = format(vol)
        Ks = format(Ks)
        rMness = format(rMness)
        return vol, Ts, Ks, rMness

    @staticmethod
    def extract_vol_atm(vol, Ks, rMness):

        def get_atm_vol(volT, KsT, atm_defnition):
            if atm_defnition in KsT:
                i_atm = np.where(KsT == atm_defnition)[0][0]
                sigma_atm = volT[i_atm]
            else:
                ip = np.argmax(KsT > atm_defnition)
                lam = (atm_defnition - KsT[ip - 1]) / (KsT[ip] - KsT[ip - 1])
                sigma_atm = (1 - lam) * volT[ip] + lam * volT[ip - 1]
            return sigma_atm

        if Ks is not None:
            vol_atm = np.array([get_atm_vol(v, K, 100.0) for (v, K) in zip(vol, Ks)])
        else:
            vol_atm = np.array([get_atm_vol(v, M, 0.0) for (v, M) in zip(vol, rMness)])

        return vol_atm

    @staticmethod
    def from_K_to_M(vol_atm, T, K):
        sigma_sqrtT = vol_atm * np.sqrt(T / 252)
        M = np.log(K / 100.0) / sigma_sqrtT
        return M

    @staticmethod
    def from_M_to_K(vol_atm, T, rMness):
        sigma_sqrtT = vol_atm * np.sqrt(T / 252)
        K = 100.0 * np.exp(rMness * sigma_sqrtT)
        return K

    @staticmethod
    def get_x_coordinates(vol_atm, Ts, Ks, rMness):
        sigma_sqrtT = vol_atm * np.sqrt(Ts / 252)
        if Ks is None:
            Ks = [100.0 * np.exp(M * norm) for norm, M in zip(sigma_sqrtT, rMness)]
        if rMness is None:
            rMness = [np.log(K / 100.0) / norm for norm, K in zip(sigma_sqrtT, Ks)]
        return Ks, rMness

    def interpolate_Ks(self, iT, k):
        mask = np.isnan(self.vol[iT])
        return np.interp(k, self.Ks[iT][~mask], self.vol[iT][~mask])

    def interpolate_Ms(self, iT, m):
        mask = np.isnan(self.vol[iT])
        return np.interp(m, self.rMness[iT][~mask], self.vol[iT][~mask])

    def get(self, T, K=None, rMness=None):
        """ Obtain smile values from interpolation. """
        if sum([K is not None, rMness is not None]) != 1:
            raise ValueError("Should provide strike, log-moneyness or rescaled log-moneyness.")

        # maturity interpolation
        if T not in self.Ts:
            if T <= np.min(self.Ts) or T >= np.max(self.Ts):
                raise ValueError(f"Required maturity {T} is out of scope, min={self.Ts.min()}, max={self.Ts.max()}")
            iT2 = np.argmax(T < self.Ts)
            iT1 = iT2 - 1
            lam = (T - self.Ts[iT1]) / (self.Ts[iT2] - self.Ts[iT1])
        else:
            iT1 = iT2 = np.where(self.Ts == T)[0][0]
            lam = 1.0

        # strike interpolation
        interpolator = self.interpolate_Ks if K is not None else self.interpolate_Ms
        x = K if K is not None else rMness
        vol1 = interpolator(iT1, x)
        vol2 = interpolator(iT2, x)
        vol = (1 - lam) * vol1 + lam * vol2

        return vol

    def save(self, file_path):
        if file_path.suffix != '.npz' or file_path.suffix == '':
            file_path = file_path.with_suffix('.npz')
        np.savez(file_path, Ts=self.Ts, Ks=np.array(self.Ks, dtype=object), vol=np.array(self.vol, dtype=object))

    @staticmethod
    def load(file_path):
        ld = dict(np.load(file_path, allow_pickle=True))
        # TODO. Temporary hack, remove it.
        if "vol_impli" in ld:
            ld["vol"] = ld["vol_impli"]
            del ld["vol_impli"]
        if "Ms" in ld:
            ld["rMness"] = ld["Ms"]
            del ld["Ms"]
        for k in ['Ks', 'rMness', 'vol']:
            if k in ld:
                ld[k] = list(ld[k])
        return Smile(**ld)

    def plot(self, ax=None, rescale=False, sub_idces=None, color='blue', legend=False):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        for i_T, T in enumerate(self.Ts):
            if sub_idces is not None and i_T not in sub_idces:
                continue

            color_here = lighten_color(color, amount=1 - np.linspace(0, 0.8, self.Ts.size)[i_T])
            xs = self.rMness[i_T] if rescale else self.Ks[i_T]
            ax.plot(xs, self.vol[i_T], color=color_here, label=f"T={T}")
            ax.scatter(xs, self.vol[i_T], color=color_here, marker="+")

        ax.grid(True)
        if legend:
            ax.legend()

        return ax
