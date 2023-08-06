""" Data classes and functions. """
from typing import *
from collections import OrderedDict
from abc import abstractmethod
from functools import partial
from multiprocessing import Pool
# from mpire import WorkerPool
import shutil
import numpy as np
import pandas as pd
from src.global_const import *
from src.utils import standardize, cumsum_zero, windows
from src.standard_models import fbm, mrw, skewed_mrw, poisson_mu, geom_brownian

""" TIME SERIES DATA classes
- stored as npz files (zipped archive)
- accessed and generated using Loader classes
- manipulated using TimeSeriesData classe

Notations
- S: number of syntheses
- B: number of batch (i.e. realizations of a process) used to average a representation
- N: number of data channels (N>1 : multivariate process)
- T: number of data samples (time samples)
"""


class TimeSeriesBase:
    """ A base class that stores generated or real world data. """

    def __init__(self, x, dts=None, name=""):
        self.x = x
        self.shape = x.shape
        self.dts = dts
        self.name = name

    def __getitem__(self, items):
        if isinstance(items, Iterable) and len(items) > 1:
            raise IndexError("Only simple slices on the last dimension are accepted for TimeSeriesBase objects.")
        x = self.x[..., items]
        dts = self.dts[..., items] if self.dts is not None else None
        return TimeSeriesBase(x, dts, self.name)

    def __repr__(self) -> str:
        return self.x.__repr__()

    def __str__(self) -> str:
        return self.x.__str__()

    def describe(self):
        return self.name

    def shape(self):
        return self.x.shape


class TimeSeriesNpzFile(TimeSeriesBase):
    """ A time-series class obtained from loading a .npz file. """
    def __init__(self, filepath=None):
        self.filepath = filepath
        ld = np.load(filepath, allow_pickle=True)
        R, N, T = ld['R'], ld['N'], ld['T']
        process_name, x = ld['process_name'], ld['x']
        super(TimeSeriesNpzFile, self).__init__(T, N, R)

        # add other attributes
        for (key, value) in ld.items():
            if key not in ['R', 'N', 'T', 'process_name', 'x']:
                self.__dict__[key] = value


class TimeSeriesDir(TimeSeriesBase):
    """ A time-series class obtained from a directory of trajectories. """
    def __init__(self, dirpath: Path,
                 R: Optional[int] = None,
                 f_idces: Optional[Iterator] = None,
                 f_names: Optional[Iterator] = None):
        if f_names is not None:
            self.fpaths = [dirpath / fn for fn in f_names]
        elif f_idces is not None:
            dpath_l = list(dirpath.iterdir())
            self.fpaths = [dpath_l[i_f] for i_f in f_idces]
        else:
            self.fpaths = list(dirpath.iterdir())
        count = 0
        x_l = []
        for fn in self.fpaths:
            x = np.load(str(fn))
            x_l.append(x)
            count += x.shape[0]
            if R is not None and count > R:
                break
        if len(self.fpaths) == 0:
            x = np.empty(0)
        else:
            x = np.concatenate(x_l)
            if R is not None and x.shape[0] < R:
                print(f"TimeSeriesDir could load only {x.shape[0]}/{R} time-series.")
            if R is not None:
                x = x[:R, :, :]
        super(TimeSeriesDir, self).__init__(x=x)


class PriceData(TimeSeriesBase):
    """ Handle price time-series, with the following data:
        - x: prices
        - dx: price increments
        - lnx: log prices
        - dlnx: log increments
    """

    def __init__(self, x=None, dx=None, lnx=None, dlnx=None, x_init=None, dts=None):
        if sum([x is None, dx is None, lnx is None, dlnx is None]) != 3:
            raise ValueError("One and only one argument x,dx,lnx,dlnx should be provided.")

        if x is None:
            if dx is not None:
                x = self.from_dx_to_x(dx)
            if lnx is not None:
                x = self.from_ln_to_x(lnx)
            if dlnx is not None:
                x = self.from_dln_to_x(dlnx)

        if x_init is not None:
            x_init = np.array(x_init)

        if x_init is not None and isinstance(x_init, np.ndarray) and (x_init.ndim > 0) and x_init.shape != x[
            ..., 0].shape:
            raise ValueError("Wrong x_init format in PriceData.")

        # set correct initial value through multiplication
        x = self.rescale(x, x_init, additive=dx is not None)

        self.lnx = np.log(x)
        self.dx = np.diff(x)
        self.dlnx = np.diff(np.log(x))

        super(PriceData, self).__init__(x=x, dts=dts)

    @staticmethod
    def rescale(x, x_init, additive):
        """ Impose the right starting point to each trajectory in x. """
        if x_init is not None:
            if additive:
                x = x - x[..., :1] + x_init[..., None]
            else:
                x *= x_init[..., None] / x[..., :1]
        return x

    @staticmethod
    def from_dx_to_x(dx):
        return cumsum_zero(dx)

    @staticmethod
    def from_ln_to_x(lnx):
        return np.exp(lnx)

    @staticmethod
    def from_dln_to_x(dlnx):
        lnx = cumsum_zero(dlnx)
        return PriceData.from_ln_to_x(lnx)


"""
LOADER classes create and access cached data 
"""


class ProcessDataLoader:
    """ Base process data loader class. """
    def __init__(self, model_name: str, dirname: Optional[Union[str, Path]] = None, num_workers: Optional[int] = 1):
        self.model_name = model_name
        self.dir_name = Path(__file__).parents[0] / '_cached_dir' if dirname is None else Path(dirname)
        self.num_workers = num_workers
        self.default_kwargs = None

        self.mkdir()

    def mkdir(self) -> None:
        self.dir_name.mkdir(parents=True, exist_ok=True)

    def dirpath(self, **kwargs) -> Path:    
        def format_path(key, value):
            if isinstance(value, dict):
                return "".join([format_path(k, v) for (k, v) in value.items()])
            elif value is None:
                return f"_none"
            elif isinstance(value, str):
                return f"_{key[:2]}_{value}"
            elif isinstance(value, int):
                return f"_{key[:2]}_{value}"
            elif isinstance(value, float):
                return f"_{key[:2]}_{value:.1e}"
            elif isinstance(value, bool):
                return f"_{key[:2]}_{int(value)}"
            else:
                return ''
        fname = (self.model_name + format_path(None, kwargs)).replace('.', '_').replace('-', '_')
        return self.dir_name / fname

    def generate_trajectory(self, **kwargs) -> np.ndarray:
        pass

    def worker(self, i: Any, **kwargs) -> None:
        np.random.seed(None)
        try:
            x = self.generate_trajectory(**kwargs)
            fname = f"{np.random.randint(1e7, 1e8)}.npy"
            np.save(str(kwargs['dirpath'] / fname), x)
            print(f"Saved: {kwargs['dirpath'].name}/{fname}")
        except ValueError as e:
            print(e)
            return

    def generate(self, dirpath: Path, n_jobs: int, **kwargs) -> dict:
        """ Performs a cached generation saving into dirpath. """
        print(f"{self.model_name}: generating data.")
        kwargs_gen = {key: value for key, value in kwargs.items() if key != 'n_files'}

        # multiprocess generation
        pool = Pool(self.num_workers)  # TODO. Catch error in a try
        pool.map(partial(self.worker, **{**kwargs_gen, **{'dirpath': dirpath}}), np.arange(n_jobs))

        # with WorkerPool(n_jobs=self.num_workers) as pool:
        #     for _ in pool.map(partial(self.worker, **{**kwargs_gen, **{'dirpath': dirpath}}), list(range(n_jobs)),
        #                       progress_bar=True):
        #         pass
        return kwargs

    def load(self, R=1, **kwargs) -> TimeSeriesDir:
        """ Loads the data required, generating it if not present in the cache. """
        full_kwargs = self.default_kwargs.copy()

        # TODO. Implement a way of associating a synthesis to certain data

        # add kwargs to default_args
        for (key, value) in kwargs.items():
            if key in self.default_kwargs or self.default_kwargs == {}:
                full_kwargs[key] = value
        dirpath = self.dirpath(**full_kwargs)
        if len(str(dirpath)) > 255:
            raise ValueError(f"Path is too long ({len(str(dirpath))} > 250).")

        # generate if necessary
        dirpath.mkdir(exist_ok=True)
        nfiles_available = len(list(dirpath.glob('*')))
        if nfiles_available < kwargs['n_files']:
            print(f"Data saving dir: {dirpath.name}")
            self.generate(n_jobs=kwargs['n_files']-nfiles_available, dirpath=dirpath, **full_kwargs)
        if len(list(dirpath.glob('*'))) < kwargs['n_files']:
            print(f"Incomplete generation {len(list(dirpath.glob('*')))} files/{kwargs['n_files']}.")

        # return available realizations
        print(f"Saved: {dirpath.name}")
        return TimeSeriesDir(dirpath=dirpath, R=R)

    def erase(self, **kwargs) -> None:
        """ Erase specified data if present in the cache. """
        full_kwargs = self.default_kwargs.copy()
        # add kwargs to default_args
        for (key, value) in kwargs.items():
            if key in self.default_kwargs:
                full_kwargs[key] = value
        shutil.rmtree(self.dirpath(**{key: value for (key, value) in full_kwargs.items() if key != 'n_files'}))


class PoissonLoader(ProcessDataLoader):
    def __init__(self, dirname: Optional[Union[str, Path]] = None):
        super(PoissonLoader, self).__init__('poisson', dirname)
        self.default_kwargs = OrderedDict({'T': 2 ** 12, 'mu': 0.01, 'signed': False})

    def generate_trajectory(self, **kwargs):
        return poisson_mu(T=kwargs['T'], mu=kwargs['mu'], signed=kwargs['signed'])[None, None, :]


class FBmLoader(ProcessDataLoader):
    def __init__(self, dirname: Optional[Union[str, Path]] = None):
        super(FBmLoader, self).__init__('fBm', dirname)
        self.default_kwargs = OrderedDict({'T': 2 ** 12, 'H': 0.5})

    def generate_trajectory(self, **kwargs):
        return fbm(R=1, T=kwargs['T'], H=kwargs['H'])[None, :]


class GBmLoader(ProcessDataLoader):
    def __init__(self, dirname: Optional[Union[str, Path]] = None):
        super(GBmLoader, self).__init__('gBm', dirname)
        self.default_kwargs = OrderedDict({'T': 2 ** 12, 'S0': 1.0, 'mu': 0.0, 'sigma': 1e-2})

    def generate_trajectory(self, **kwargs):
        return geom_brownian(R=1, T=kwargs['T'], S0=kwargs['S0'], mu=kwargs['mu'], sigma=kwargs['sigma'])


class MRWLoader(ProcessDataLoader):
    def __init__(self, dirname: Optional[Union[str, Path]] = None):
        super(MRWLoader, self).__init__('MRW', dirname)
        self.default_kwargs = OrderedDict({'T': 2 ** 12, 'L': None, 'H': 0.5, 'lam': 0.1})

    def generate_trajectory(self, **kwargs):
        L = kwargs['L'] or kwargs['T']
        return mrw(R=1, T=kwargs['T'], L=L, H=kwargs['H'], lam=kwargs['lam'])[None, :]


class SMRWLoader(ProcessDataLoader):
    def __init__(self, dirname: Optional[Union[str, Path]] = None):
        super(SMRWLoader, self).__init__('SMRW', dirname)
        self.default_kwargs = OrderedDict({'T': 2 ** 12, 'L': None, 'dt': 1, 'H': 0.5, 'lam': 0.1,
                                           'K': 0.035, 'alpha': 0.23, 'beta': 0.5, 'gamma': 1 / (2**12) / 64})

    def generate_trajectory(self, **kwargs):
        L = kwargs['L'] or kwargs['T']
        return skewed_mrw(R=1, T=kwargs['T'], L=L, dt=kwargs['dt'], H=kwargs['H'], lam=kwargs['lam'],
                          K0=kwargs['K'], alpha=kwargs['alpha'], beta=kwargs['beta'], gamma=kwargs['gamma'])[None, :]


class SPDaily(PriceData):
    def __init__(self, T=5884, filename=None):
        if T > 5884:
            raise ValueError('For SP500 only T <= 5884 is supported')
        if filename is None:
            filename = FINANCE_DATA_PATH / 'snp_extended_WSJ_23_05_2023.csv'

        df = pd.read_csv(filename)

        def formatter(dt_str):
            m, d, y = [x for x in dt_str.split('/')]
            m, d, y = int(m), int(d), int('20' + y)
            return pd.Timestamp(year=y, month=m, day=d)

        df.index = pd.DatetimeIndex(df['Date'].apply(formatter))
        df = df.sort_index()

        x = df[' Close'].values[:T]
        dts = df.index[:T]

        super(SPDaily, self).__init__(x=x[None, None, :], dts=dts)


class DatasetSnpSimple:
    def __init__(self, x, w, s):
        self.x = x
        self.w = w
        self.s = s

        self.nb_w = (x.shape[-1] - w - 0) // s + 1

        self.train = self.prepare(w=w, s=s, offset=0)
        self.test = self.prepare(w=w, s=s, offset=w - 1)

    def prepare(self, **kwargs):
        x_pad = np.pad(self.x, ((0, 0), (0, 0), (0, self.w)), 'constant', constant_values=(1e-7, 1e-7))
        index = windows(np.arange(x_pad.shape[-1]), **kwargs)
        xw = windows(x_pad, **kwargs)[0, :, :].transpose((1, 0, 2))

        return PriceData(x=xw[:self.nb_w, 0, :], dts=index[:self.nb_w, :])


class RoughHeston(TimeSeriesBase):
    """ The rough Heston model calibrated on SnP (from Zhang and Rosenbaum). """

    def __init__(self, B, T):
        if B > 20 or T > 7300:
            raise ValueError("Not enough rough heston data available.")

        self.filepath = DATA_PATH / 'synthetic' / 'roughHeston' / 'synthese' / 'spx_series.csv'
        df_data = pd.read_csv(self.filepath)  # 7300 x 21 dataframe
        log_prices = df_data.values[:, 1:].T  # 20 x 7300 array

        super(RoughHeston, self).__init__(x=log_prices[:B, None, :T], dts=None, name='roughHeston')


class RoughHestonVol(TimeSeriesBase):
    """ The log volatilities of a rough Heston model calibrated on SnP (from Zhang and Rosenbaum). """

    def __init__(self, B, T):
        if B > 20 or T > 7300:
            raise ValueError("Not enough rough heston data available.")

        self.filepath = DATA_PATH / 'synthetic' / 'roughHeston' / 'synthese' / 'vol_series.csv'
        df_data = pd.read_csv(self.filepath)  # 7300 x 21 dataframe
        log_vols = np.log10(df_data.values[:, 1:].T)  # 20 x 7300 array

        super(RoughHestonVol, self).__init__(x=log_vols[:B, None, :T], dts=None, name='roughHeston')


class NotEnoughDataException(Exception):
    pass
