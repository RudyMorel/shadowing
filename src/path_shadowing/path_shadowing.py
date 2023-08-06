""" Multiprocessed Path Shadowing: scanning over a generated dataset of trajectories. """
import re
from tqdm import tqdm
import shutil
import pickle
from functools import partial
from mpire import WorkerPool
import numpy as np

from src.global_const import *
from src.utils import windows, Softmax, Uniform, get_smallest, array_equal
from src.data_source import TimeSeriesDir
from src.path_shadowing import EMBEDDING_CHOICE, DISTANCE_CHOICE


class PathShadowing:
    def __init__(self,
                 x_past,
                 spath, S,
                 dirname,
                 mem_multiplier=1.0,
                 verbose=False):
        """
        :param x_past: array (B, T), past trajectories to predict future on
        :param spath: path of the model syntheses' directory
        :param dirname: prefix name of the cache directory
        :param verbose:
        """
        self.x_past = x_past.astype(np.float32)
        self.n_dates, self.w_past = x_past.shape

        # dataset parameters
        self.spath = spath
        self.S = S
        self.mem_multiplier = mem_multiplier
        self.verbose = verbose

        # path to the directory where files will be stored
        self.cache_path = self.get_global_dirpath(dirname, spath)
        if self.verbose:
            print(f"Global cache path. {self.cache_path}")

        # infer horizon of trajectories in spath
        self.T = self.infer_synthesis_horizon(spath)

        # create main directories
        self.main_paths = self._create_main_directories(x_past)

        # partition syntheses for memory purpose
        self.synt_partition = self.get_syntheses_partition(self.get_snames(), S, self.T, self.w_past, mem_multiplier)
        self.B = len(self.synt_partition)

    @staticmethod
    def infer_synthesis_horizon(spath):
        """ Infer horizon of trajectories in spath. """
        dtld = TimeSeriesDir(spath, R=1)
        return dtld.x.shape[-1]

    @staticmethod
    def get_global_dirpath(dirname, spath):
        """ Get a unique path corresponding to dirpath and spath for the global directory. """
        main_name = dirname or 'main'
        unique_spath_suffix = re.sub(r'(.)\1+', r'\1', ''.join(c for c in spath.name if c.isdigit()))
        return CACHE_PATH / 'path_shadowing' / (main_name + '_' + unique_spath_suffix)

    @staticmethod
    def get_main_dirname(single_x_past):
        """ Encode x_past -> string. """
        np.random.seed(0)
        key = np.random.randn(single_x_past.size).astype(single_x_past.dtype)
        code1 = np.dot(single_x_past / single_x_past.std(), key)
        idx = np.random.randint(single_x_past.size)
        code2 = single_x_past[idx]
        np.random.seed(None)

        to_remove = ['-', '+', 'e', '.']
        full_string = "".join([f"{x:.3e}" for x in [code1, code2]])
        for c in to_remove:
            full_string = full_string.replace(c, '')

        return full_string

    def _create_main_directories(self, x_past):
        """ Create cache directory depending only on x_past and spath. """
        if not self.cache_path.is_dir():
            self.cache_path.mkdir(parents=True)

        main_paths = [self.cache_path / self.get_main_dirname(x) for x in x_past]
        for dpath in main_paths:
            dpath.mkdir(exist_ok=True)

        return main_paths

    @staticmethod
    def get_dname(S, embedding, distance, n_paths, T_max):
        return f"S{S}_emb_{embedding.get_unique_name()}_dist_{distance.get_unique_name()}"\
               f"_npaths{n_paths}_maxT{T_max}".replace('.', '_')

    @staticmethod
    def load_syntheses(spath, f_names):
        """ Load the syntheses specified in f_idces. """
        dtld = TimeSeriesDir(spath, f_names=f_names)
        return dtld.x[:, 0, :]

    def get_snames(self):
        """ Load S syntheses names in spath considered for path matching. """
        # TODO. Make it a json, the value S and spath should not be changed actually.
        fpath = self.cache_path / 'syntheses_names.pkl'

        if not fpath.is_file():
            dict_snames = {}
        else:
            with open(fpath, 'rb') as f:
                unpickler = pickle.Unpickler(f)
                dict_snames = unpickler.load()

        k = (self.S, self.spath.name)
        if k not in dict_snames:
            # add to dictionary and save
            snames_available = [dn.name for dn in self.spath.iterdir()]
            if len(snames_available) < self.S:
                raise ValueError("Not enough syntheses available.")
            dict_snames[k] = snames_available[:self.S]
            with open(fpath, 'wb') as f:
                pickle.dump(dict_snames, f)

        return dict_snames[k]

    @staticmethod
    def get_syntheses_partition(snames, S, T, w_past, memory_multiplier):
        """ Partition full syntheses for memory purpose.
        Increasing multiplier allows more memory to be used by reducing the number of batches.  """
        # make sure that each synthesis batch with all its paths contains ~ 300 000 000 samples
        # must have batch_size * (T-w_past) * w_past = 300 000 000
        batch_size = int(memory_multiplier * 300000000 / (T - w_past) / w_past)
        return np.array_split(np.array(snames[:S]), S // batch_size + (1 if S < batch_size else 0))

    def erase_cache(self, embedding, distance, n_paths, T_max):
        """ Erase global directories corresponding to past paths self.x_past and syntheses directory self.spath. """
        for path in self.main_paths:
            dirname = self.get_dname(self.S, embedding, distance, n_paths, T_max)
            shutil.rmtree(path / dirname)

    @staticmethod
    def _dist_fpath(dirpath, b, B):
        if b is None or B is None or b == B-1:
            return dirpath / f'distance.npy'
        return dirpath / f'distance_batch{b}_B{B}.npy'

    @staticmethod
    def _loc_fpath(dirpath, b, B):
        if b is None or B is None or b == B-1:
            return dirpath / f'locator.npy'
        return dirpath / f'locator_batch{b}_B{B}.npy'

    @staticmethod
    def _load_dist(dirpath, b, B):
        return np.load(PathShadowing._dist_fpath(dirpath, b, B))

    @staticmethod
    def _load_loc(dirpath, b, B):
        return np.load(PathShadowing._loc_fpath(dirpath, b, B))

    @staticmethod
    def _save_dist(dirpath, b, B, distances):
        np.save(PathShadowing._dist_fpath(dirpath, b, B), distances)

    @staticmethod
    def _save_loc(dirpath, b, B, locators):
        np.save(PathShadowing._loc_fpath(dirpath, b, B), locators)

    @staticmethod
    def _erase_dist(dirpath, b, B):
        PathShadowing._dist_fpath(dirpath, b, B).unlink()

    @staticmethod
    def _erase_loc(dirpath, b, B):
        PathShadowing._loc_fpath(dirpath, b, B).unlink()

    @staticmethod
    def path_distance(embedding, distance, x_past, x_synt_past):
        return distance(
            embedding(x_past),
            embedding(x_synt_past)
        )

    def need_to_create_cache(self, embedding, distance, n_paths, T_max):
        """ Return wehther it is required to launch multiprocessed cache creation. """
        for single_dpath in self.main_paths:

            dirpath = single_dpath / self.get_dname(self.S, embedding, distance, n_paths, T_max)

            # filenames of dist, path for this single_x_past and this batch of syntheses
            distance_fpath = self._dist_fpath(dirpath, None, None)
            locator_fpath = self._loc_fpath(dirpath, None, None)

            # skip if cache file is already present
            if not distance_fpath.is_file() or not locator_fpath.is_file():
                return True

        return False

    def _create_cache_worker(self, i_t, x_synt_w, b,
                             embedding, distance,
                             n_paths, T_max):
        """ Compute distances and closest paths for a single x_past. """
        n_traj = x_synt_w.shape[0]

        dirpath = self.main_paths[i_t] / self.get_dname(self.S, embedding, distance, n_paths, T_max)
        single_x_past = self.x_past[i_t, :]

        # skip if cache file already exists
        for bp in range(b, self.B):
            if self._dist_fpath(dirpath, bp, self.B).is_file():
                return

        # compute distances with any path in x_synt
        distances = np.empty((n_traj, self.T), dtype=np.float32)
        distances[:, :-self.w_past+1] = self.path_distance(embedding, distance, single_x_past, x_synt_w)  # n_traj x ..
        mask_edge_effect = np.tile(np.arange(self.T) <= self.T-self.w_past-T_max, (n_traj, 1))
        distances[~mask_edge_effect] = np.inf

        # get smallest distances this batch
        # a locator is a tuple (s_idx, t), where 0 <= s_idx < S and 0 <= t < T are integers
        idces_this_b, d_smallest_this_b = get_smallest(distances.ravel(), n_paths)
        offset_this_b = sum([sn.shape[0] for sn in self.synt_partition][:b])
        locators_this_b = np.array([(i//self.T + offset_this_b, i%self.T) for i in idces_this_b], dtype=np.int32)

        if not distances.dtype == single_x_past.dtype == x_synt_w.dtype:
            raise ValueError(f"Inconsistent dtypes ({distances.dtype},{single_x_past.dtype},{x_synt_w.dtype}) "
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

    def _create_cache(self, embedding, distance, n_paths, T_max, num_workers, pbar=True):
        """ Compute distances and locators. """
        # loop over batch of syntheses
        if self.verbose:
            print(f"Nb batches: {self.B}, "
                  f"Avg batch size: {self.S / self.B:.0f} (syntheses), "
                  f"Avg per batch {self.S / self.B * (self.T-self.w_past):.0f} (paths of size {self.w_past})")

        exp_name = self.get_dname(self.S, embedding, distance, n_paths, T_max)
        for main_path in self.main_paths:
            (main_path / exp_name).mkdir(exist_ok=True)

        for b, snames in enumerate(tqdm(self.synt_partition, desc='Synthesis batch')):
            # Prepare all subpaths for this batch of syntheses
            x_synt = self.load_syntheses(self.spath, snames).astype(np.float32)  # n_traj x T
            x_synt_w = windows(x_synt, w=self.w_past, s=1)  # n_traj x Q x w_past

            # multiprocessed computation for each prediction date
            worker_create_partial = partial(self._create_cache_worker, x_synt_w=x_synt_w, b=b,
                                            embedding=embedding, distance=distance,
                                            n_paths=n_paths, T_max=T_max)
            if self.x_past.shape[0] == 1:
                for i_t in range(self.x_past.shape[0]):
                    worker_create_partial(i_t)
            else:
                with WorkerPool(n_jobs=num_workers) as pool:
                    for _ in pool.map(worker_create_partial, list(range(self.x_past.shape[0])), progress_bar=pbar,
                                      progress_bar_options={'desc': 'Prediction date'}):
                        pass

    @staticmethod
    def init_embedding_and_distance(embedding_name, distance_name, embedding_kwargs, distance_kwargs):
        embedding = EMBEDDING_CHOICE[embedding_name](**embedding_kwargs)
        distance = DISTANCE_CHOICE[distance_name](**distance_kwargs)
        return embedding, distance

    def _load_cache_worker(self, i_t, x_synt_all, embedding, distance, n_paths, T_max):
        ts = np.arange(self.w_past + T_max)
        single_x_past = self.x_past[i_t, :]
        dirpath = self.main_paths[i_t] / self.get_dname(self.S, embedding, distance, n_paths, T_max)

        # load distances
        distances_this_date = self._load_dist(dirpath, None, None)
        if distances_this_date.dtype != np.float32:
            print(f"WARNING. Loaded distances have dtype {distances_this_date.dtype} while user asked for {np.float32}.")

        # get paths
        locators = self._load_loc(dirpath, None, None)  # (s, t)
        locators = locators[:, 0] * self.T + locators[:, 1]
        indices = locators[:, None] + ts[None, :]
        paths_this_date = x_synt_all[indices].reshape(-1, ts.size)

        # check # TODO. Remove checks.
        assert not np.isinf(distances_this_date).any(), "Error, should not find np.inf values in distances."
        test_dist = self.path_distance(embedding, distance, single_x_past, paths_this_date[:, :self.w_past])
        assert array_equal(distances_this_date, test_dist, 1e-3), \
            f"Inconsistency in closest paths selection on {distances_this_date.size} paths"

        return i_t, distances_this_date, paths_this_date

    def load_cache(self, embedding, distance, n_paths, T_max, num_workers):
        """ Load distances and closest paths to x_past. """
        # name of the S synthesis in spath
        all_snames = self.get_snames()
        x_synt_all = self.load_syntheses(self.spath, all_snames).astype(np.float32).ravel()

        # create cache if not present
        if self.need_to_create_cache(embedding, distance, n_paths, T_max):
            if self.verbose:
                print("Creating cache ...")
            self._create_cache(embedding, distance, n_paths, T_max, num_workers)

        # loop over all prediction dates and all syntheses batch
        if self.verbose:
            print("Loading cache ...")
            # print("Will have to remove distance check at some point.")

        # multiprocessed loading for each prediction date
        distances_l = []
        paths_l = []
        worker = partial(self._load_cache_worker, x_synt_all=x_synt_all,
                         embedding=embedding, distance=distance,
                         n_paths=n_paths, T_max=T_max)
        if self.x_past.shape[0] == 1:
            for i_t in range(self.x_past.shape[0]):
                _, distances, paths = worker(i_t)
                distances_l.append(distances)
                paths_l.append(paths)
        else:
            job_id = []
            with WorkerPool(n_jobs=num_workers) as pool:
                for result in pool.map(worker, list(range(self.x_past.shape[0])),
                                       progress_bar=True, progress_bar_options={'desc': 'Prediction date'}):
                    i_t, distances, paths = result
                    job_id.append(i_t)
                    distances_l.append(distances)
                    paths_l.append(paths)
            if job_id != list(range(self.x_past.shape[0])):
                raise ValueError("Workers did not return results in same order than prediction data")

        return distances_l, paths_l

    @staticmethod
    def init_averaging_proba(proba_name, distances, eta):
        """ The averaging proba used for approximating E{q(x)|x_past}. """
        if proba_name == "uniform":
            return Uniform()
        elif proba_name == "softmax":
            return Softmax(distances, eta)
        else:
            raise ValueError("Unrecognized averaging proba")

    def predict_from_paths(self, distances, paths, to_predict, proba_name, eta):
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

    def predict(self, embedding, distance, n_paths,
                T_max, to_predict=None, proba_name="uniform", eta=None,
                num_workers=1):
        """
        Prediction through paths matching at time horizons Ts.

        :param embedding: embedding that yields past history
        :param distance: distance used for measuring path matching
        :param n_paths: threshold defining "close" paths
        :param T_max: prediction time horizons
        :param to_predict: quantity to predict, should not rely on arrays of size > T_max
        :param proba_name: name of the averaging proba
        :param eta: parameter used for a softmax averaging
        :param num_workers: number of multiprocess workers
        :return:
        """
        # initialize max time horizon in the paths' future
        # loading cache
        distances, paths = self.load_cache(embedding, distance, n_paths, T_max, num_workers)

        return self.predict_from_paths(distances, paths, to_predict, proba_name, eta)
