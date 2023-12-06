""" Multiprocessed Path Shadowing: scanning over a generated dataset of trajectories. """
from typing import Dict, List, Callable, Tuple, Iterable
import re
from tqdm import tqdm
import shutil
from functools import partial
from mpire import WorkerPool
import numpy as np
from pathlib import Path

from shadowing.utils import windows, Softmax, Uniform, get_smallest, array_equal
from shadowing.path_shadowing.path_embedding import PathEmbedding, EMBEDDING_CHOICE
from shadowing.path_shadowing.path_distance import PathDistance, DISTANCE_CHOICE
from scatspectra.data_source import TimeSeriesDataset


def to_predict(x: np.ndarray, 
               Ts: Iterable, 
               vol: bool):
    x2 = x ** 2
    realized_variance = np.stack([x2[..., :T].mean(-1) for T in Ts], -1) * 252
    if vol:
        return realized_variance ** 0.5
    return realized_variance


class PathShadowing:
    def __init__(self,
                 dlnx_past: np.ndarray,
                 dataset: Path | TimeSeriesDataset, 
                 S: int,
                 cache_path: Path,
                 dirname: str | None = None,
                 n_splits : int = 1,
                 verbose: bool = False):
        """
        :param dlnx_past: shape (B, T), B time-series to do prediction on
        :param dataset_path: path of the generated dataset or the dataset iteself
        :param dirname: prefix name of the cache directory
        :param n_splits: number of splits of the dataset, does not impact 
            prediction results. Increase it for reducing memory usage.
        :param verbose:
        """
        self.dlnx_past = dlnx_past.astype(np.float32)
        self.n_dates, self.w_past = dlnx_past.shape

        # dataset used to "shadow" dlnx_past 
        if isinstance(dataset, Path):
            dataset = TimeSeriesDataset(dpath=dataset, R=S, load=False)
        self.dataset = dataset
        self.S = S
        if self.dataset.R < S:
            raise Exception(f"Dataset contains only R={self.dataset.R} < S={S} time-series.")
        self._T_max = dataset.T - 1  # T_max: length of a generated log-return time-series in the dataset

        # partition dataset for memory purpose
        self.datasets = self.dataset.split(num_splits=n_splits)
        self.B = len(self.datasets)
        self.verbose = verbose

        # initialize cache
        self.cache_path, self.subcache_paths = self.init_cache(cache_path, dirname)

    def init_cache(self, 
                   cache_path: Path, 
                   dirname) -> Tuple[Path, List[Path]]:
        """ Initialize cache path. """
        # define a "unique" path corresponding to dirpath and spath for the global directory
        # TODO: not perfect, find a better way of identifying uniquely the 
        # dataset directory name with the path shadowing cache directory
        main_name = dirname or 'main'
        unique_spath_suffix = re.sub(r'(.)\1+', r'\1', ''.join(c for c in self.dataset.dpath.name if c.isdigit()))
        self.cache_path = cache_path / 'path_shadowing' / (main_name + '_' + unique_spath_suffix)
        if self.verbose:
            print(f"Global cache path. {self.cache_path}")

        # create cache directory
        if not self.cache_path.is_dir():
            self.cache_path.mkdir(parents=True)

        # create sub-cache directories for each time-series to shadow
        self.subcache_paths = [self.cache_path / self.get_main_dirname(x) for x in self.dlnx_past]
        for dpath in self.subcache_paths:
            dpath.mkdir(exist_ok=True)

        return self.cache_path, self.subcache_paths

    @staticmethod
    def get_main_dirname(single_dlnx_past: np.ndarray) -> str:
        """ Encode dlnx_past -> string. """
        np.random.seed(0)
        key = np.random.randn(single_dlnx_past.size).astype(single_dlnx_past.dtype)
        code1 = np.dot(single_dlnx_past / single_dlnx_past.std(), key)
        idx = np.random.randint(single_dlnx_past.size)
        code2 = single_dlnx_past[idx]
        np.random.seed(None)

        to_remove = ['-', '+', 'e', '.']
        full_string = "".join([f"{x:.3e}" for x in [code1, code2]])
        for c in to_remove:
            full_string = full_string.replace(c, '')

        return full_string

    @staticmethod
    def get_dname(S: int, 
                  embedding: PathEmbedding, 
                  distance: PathDistance, 
                  n_paths: int, 
                  horizon: int) -> str:
        """ Name of the cache directory corresponding to these shadowing parameters. """
        return f"S{S}_emb_{embedding.get_unique_name()}_dist_{distance.get_unique_name()}"\
               f"_npaths{n_paths}_horizon{horizon}".replace('.', '_')

    def erase_cache(self, 
                    embedding: PathEmbedding, 
                    distance: PathDistance, 
                    n_paths: int, 
                    horizon: int) -> None:
        """ Erase global directories corresponding to past paths self.dlnx_past and syntheses directory self.spath. """
        if self.subcache_paths is None:
            return
        for path in self.subcache_paths:
            dirname = self.get_dname(self.S, embedding, distance, n_paths, horizon)
            shutil.rmtree(path / dirname)

    @staticmethod
    def _dist_fpath(dirpath: Path, 
                    b: int | None, 
                    B: int | None) -> Path:
        if b is None or B is None or b == B-1:
            return dirpath / f'distance.npy'
        return dirpath / f'distance_batch{b}_B{B}.npy'

    @staticmethod
    def _loc_fpath(dirpath: Path, 
                   b: int | None, 
                   B: int | None) -> Path:
        if b is None or B is None or b == B-1:
            return dirpath / f'locator.npy'
        return dirpath / f'locator_batch{b}_B{B}.npy'

    @staticmethod
    def _load_dist(dirpath: Path, 
                   b: int | None, 
                   B: int | None) -> np.ndarray:
        return np.load(PathShadowing._dist_fpath(dirpath, b, B))

    @staticmethod
    def _load_loc(dirpath: Path, 
                   b: int | None, 
                   B: int | None) -> np.ndarray:
        return np.load(PathShadowing._loc_fpath(dirpath, b, B))

    @staticmethod
    def _save_dist(dirpath: Path, 
                   b: int | None, 
                   B: int | None,
                   distances: np.ndarray) -> None:
        np.save(PathShadowing._dist_fpath(dirpath, b, B), distances)

    @staticmethod
    def _save_loc(dirpath: Path, 
                  b: int | None, 
                  B: int | None,
                  locators: np.ndarray) -> None:
        np.save(PathShadowing._loc_fpath(dirpath, b, B), locators)

    @staticmethod
    def _erase_dist(dirpath: Path, 
                    b: int | None, 
                    B: int | None) -> None:
        PathShadowing._dist_fpath(dirpath, b, B).unlink()

    @staticmethod
    def _erase_loc(dirpath: Path, 
                   b: int | None, 
                   B: int | None) -> None:
        PathShadowing._loc_fpath(dirpath, b, B).unlink()

    @staticmethod
    def path_distance(embedding: PathEmbedding, 
                      distance: PathDistance, 
                      dlnx_past: np.ndarray, 
                      dlnx_gen_past: np.ndarray) -> np.ndarray:
        return distance(
            embedding(dlnx_past),
            embedding(dlnx_gen_past)
        )

    def need_to_create_cache(self, 
                             embedding: PathEmbedding, 
                             distance: PathDistance,
                             n_paths: int, 
                             horizon: int) -> bool:
        """ Return wehther it is required to launch multiprocessed cache creation. """
        if self.subcache_paths is None:
            return True
        
        for single_dpath in self.subcache_paths:

            dirpath = single_dpath / self.get_dname(self.S, embedding, distance, n_paths, horizon)

            # filenames of dist, path for this single_dlnx_past and this batch of syntheses
            distance_fpath = self._dist_fpath(dirpath, None, None)
            locator_fpath = self._loc_fpath(dirpath, None, None)

            # skip if cache file is already present
            if not distance_fpath.is_file() or not locator_fpath.is_file():
                return True

        return False

    def _create_cache_worker(self, 
                             i_t: int, 
                             dlnx_gen: np.ndarray, 
                             b: int,
                             embedding: PathEmbedding, 
                             distance: PathDistance,
                             n_paths: int, 
                             horizon: int) -> None:
        """ Compute distances and closest paths for a single dlnx_past. """
        try:
            n_traj = dlnx_gen.shape[0]

            dirpath = self.subcache_paths[i_t] / self.get_dname(self.S, embedding, distance, n_paths, horizon)
            single_dlnx_past = self.dlnx_past[i_t, :]

            # skip if cache file already exists
            for bp in range(b, self.B):
                if self._dist_fpath(dirpath, bp, self.B).is_file():
                    return

            # compute distances with any path in dlnx_gen
            distances = np.empty((n_traj, self._T_max), dtype=np.float32)
            #TODO: this may be optimized through matrix multiplication or on gpu
            distances[:, :-self.w_past+1] = self.path_distance(embedding, distance, single_dlnx_past, dlnx_gen)  # n_traj x ..
            mask_edge_effect = np.tile(np.arange(self._T_max) <= self._T_max-self.w_past-horizon, (n_traj, 1))
            distances[~mask_edge_effect] = np.inf

            # get smallest distances this batch
            # a locator is a tuple (s_idx, t), where 0 <= s_idx < S and 0 <= t < T are integers
            idces_this_b, d_smallest_this_b = get_smallest(distances.ravel(), n_paths)
            offset_this_b = sum([dtst.R for dtst in self.datasets[:b]])
            locators_this_b = np.array([(i//self._T_max + offset_this_b, i%self._T_max) for i in idces_this_b], dtype=np.int32)

            if not distances.dtype == single_dlnx_past.dtype == dlnx_gen.dtype:
                raise ValueError(f"Inconsistent dtypes ({distances.dtype},{single_dlnx_past.dtype},{dlnx_gen.dtype}) "
                                f"may lead to overtimes.")

            # get smallest n_paths distances from all preceding batches
            if b > 0:
                d_smallest_last_b = self._load_dist(dirpath, b - 1, self.B)
                locators_last_b = self._load_loc(dirpath, b - 1, self.B)
                idces_smallest, d_smallest = get_smallest(np.concatenate([d_smallest_last_b, d_smallest_this_b]), n_paths)
                locators = np.concatenate([locators_last_b, locators_this_b])[idces_smallest, :]
                self._erase_dist(dirpath, b - 1, self.B)
                self._erase_loc(dirpath, b - 1, self.B)
                # TODO. At the end, should erase any other files having b not None and B not None
            else:
                d_smallest = d_smallest_this_b
                locators = locators_this_b

            self._save_dist(dirpath, b, self.B, d_smallest)
            self._save_loc(dirpath, b, self.B, locators)
        except Exception as e:
            print(f"Error in worker {b} {i_t}")
            raise e

    def _create_cache(self, 
                      embedding: PathEmbedding, 
                      distance: PathDistance, 
                      n_paths: int, 
                      horizon: int, 
                      num_workers: int, 
                      pbar: bool = True) -> None:
        """ Compute distances and locators. """
        # loop over batch of syntheses
        if self.verbose:
            print(f"Nb dataset splits: {self.B}, "
                  f"Avg split size: {self.S / self.B:.0f} (syntheses), "
                  f"Avg per split {self.S / self.B * (self._T_max-self.w_past):,.0f} (paths of size {self.w_past})")

        exp_name = self.get_dname(self.S, embedding, distance, n_paths, horizon)
        for main_path in self.subcache_paths:
            (main_path / exp_name).mkdir(exist_ok=True)

        for b, dataset in enumerate(tqdm(self.datasets, desc='Synthesis batch')):
            # Prepare all subpaths for this batch of syntheses
            lnx_gen = dataset.load()[:,0,:].astype(np.float32)
            dlnx_gen = np.diff(lnx_gen)  # n_traj x T
            dlnx_gen_w = windows(dlnx_gen, w=self.w_past, s=1)  # n_traj x Q x w_past

            # multiprocessed computation for each prediction date
            worker_create_partial = partial(self._create_cache_worker, dlnx_gen=dlnx_gen_w, b=b,
                                            embedding=embedding, distance=distance,
                                            n_paths=n_paths, horizon=horizon)
            if self.dlnx_past.shape[0] == 1:
                for i_t in range(self.dlnx_past.shape[0]):
                    worker_create_partial(i_t)
            else:
                with WorkerPool(n_jobs=num_workers) as pool:
                    for _ in pool.map(worker_create_partial, list(range(self.dlnx_past.shape[0])), progress_bar=pbar,
                                      progress_bar_options={'desc': 'Prediction date'}):
                        pass

    def _load_cache_worker(self, 
                           i_t: int, 
                           dlnx_gen_all: np.ndarray, 
                           embedding: PathEmbedding, 
                           distance: PathDistance, 
                           n_paths: int, 
                           horizon: int):
        try:
            ts = np.arange(self.w_past + horizon)
            single_dlnx_past = self.dlnx_past[i_t, :]
            dirpath = self.subcache_paths[i_t] / self.get_dname(self.S, embedding, distance, n_paths, horizon)

            # load distances
            distances_this_date = self._load_dist(dirpath, None, None)
            if distances_this_date.dtype != np.float32:
                print(f"WARNING. Loaded distances have dtype {distances_this_date.dtype} while user asked for {np.float32}.")

            # get paths
            locators = self._load_loc(dirpath, None, None)  # (s, t)
            locators = locators[:, 0] * self._T_max + locators[:, 1]
            indices = locators[:, None] + ts[None, :]
            paths_this_date = dlnx_gen_all[indices].reshape(-1, ts.size)

            # check # TODO: Remove checks at some point.
            assert not np.isinf(distances_this_date).any(), "Error, should not find np.inf values in distances."
            test_dist = self.path_distance(embedding, distance, single_dlnx_past, paths_this_date[:, :self.w_past])
            assert array_equal(distances_this_date, test_dist, 1e-3), \
                f"Inconsistency in closest paths selection on {distances_this_date.size} paths"

            return i_t, distances_this_date, paths_this_date
        except Exception as e:
            print(f"Error in worker {i_t}")
            raise e

    def shadow(self,
               embedding: PathEmbedding,
               distance: PathDistance,
               n_paths: int,
               horizon: int,
               num_workers: int):
        """ Load distances and closest paths to dlnx_past. """
        dlnx_gen_all = np.diff(self.dataset.load())[:,0,:].astype(np.float32).ravel()

        # create cache if not present
        if self.need_to_create_cache(embedding, distance, n_paths, horizon):
            if self.verbose:
                print("Creating cache ...")
            self._create_cache(embedding, distance, n_paths, horizon, num_workers)

        # loop over all prediction dates and all syntheses batch
        if self.verbose:
            print("Loading cache ...")

        # multiprocessed loading for each prediction date
        distances_l = []
        paths_l = []
        worker = partial(self._load_cache_worker, dlnx_gen_all=dlnx_gen_all,
                         embedding=embedding, distance=distance,
                         n_paths=n_paths, horizon=horizon)
        if self.dlnx_past.shape[0] == 1:
            for i_t in range(self.dlnx_past.shape[0]):
                _, distances, paths = worker(i_t)
                distances_l.append(distances)
                paths_l.append(paths)
        else:
            job_id = []
            with WorkerPool(n_jobs=num_workers) as pool:
                for result in pool.map(worker, list(range(self.dlnx_past.shape[0])),
                                       progress_bar=True, progress_bar_options={'desc': 'Prediction date'}):
                    i_t, distances, paths = result
                    job_id.append(i_t)
                    distances_l.append(distances)
                    paths_l.append(paths)
            if job_id != list(range(self.dlnx_past.shape[0])):
                raise ValueError("Workers did not return results in same order than prediction data")

        print("DONE.")

        return distances_l, paths_l

    @staticmethod
    def init_embedding_and_distance(embedding_name: str, 
                                    distance_name: str, 
                                    embedding_kwargs: Dict, 
                                    distance_kwargs: Dict) -> Tuple[PathEmbedding, PathDistance]:
        embedding = EMBEDDING_CHOICE[embedding_name](**embedding_kwargs)
        distance = DISTANCE_CHOICE[distance_name](**distance_kwargs)
        return embedding, distance

    @staticmethod
    def init_averaging_proba(proba_name: str, 
                             distances: np.ndarray, 
                             eta: float):
        """ The averaging proba used for approximating E{q(x)|x_past}. """
        if proba_name == "uniform":
            return Uniform()
        elif proba_name == "softmax":
            return Softmax(distances, eta)
        else:
            raise ValueError("Unrecognized averaging proba")

    def predict_from_paths(self, 
                           distances: List[np.ndarray], 
                           paths: List[np.ndarray], 
                           to_predict: Callable, 
                           proba_name: str, 
                           eta: float):
        """ Agregate predictions on shadowing paths. """

        # define averaging operators
        empirical_probas = [self.init_averaging_proba(proba_name, d, eta) for d in distances]

        if not empirical_probas[0].w.dtype == paths[0].dtype == distances[0].dtype:
            raise ValueError(f"Inconsistent dtypes, may lead to overtimes.")

        if self.verbose:
            print("Prediction ...")

        # get prediction
        predictions = np.stack([prob.avg(to_predict(p[:, self.w_past:]), axis=0) for (prob, p) in zip(empirical_probas, paths)])
        predictions_std = np.array([prob.std(to_predict(p[:, self.w_past:]), axis=0) for (prob, p) in zip(empirical_probas, paths)])
        if self.verbose:
            print("FINISHED prediction.")

        return {
            'paths': paths,
            'distances': distances,
            'weights': empirical_probas,
            'predictions': predictions,
            'predictions_std': predictions_std,
        }

    def predict(self, 
                embedding: PathEmbedding, 
                distance: PathDistance, 
                n_paths: int,
                horizon: int, 
                to_predict: Callable,
                eta: float,
                proba_name: str = "uniform", 
                num_workers: int = 1):
        """
        Prediction through paths matching at time horizons Ts.

        :param embedding: embedding that yields past history
        :param distance: distance used for measuring path matching
        :param n_paths: threshold defining "close" paths
        :param horizon: prediction time horizons
        :param to_predict: quantity to predict, should not rely on arrays of size > horizon
        :param proba_name: name of the averaging proba
        :param eta: parameter used for a softmax averaging
        :param num_workers: number of multiprocess workers
        :return:
        """
        # initialize max time horizon in the paths' future
        # loading cache
        distances, paths = self.shadow(embedding, distance, n_paths, horizon, num_workers)

        return self.predict_from_paths(distances, paths, to_predict, proba_name, eta)
