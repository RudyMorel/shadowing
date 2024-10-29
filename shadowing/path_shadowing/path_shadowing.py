""" Multiprocessed Path Shadowing: scanning over a generated dataset of trajectories. """
from typing import Callable
from pathlib import Path
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from scatspectra import TimeSeriesDataset, Softmax, Uniform, DiscreteProba
from shadowing.path_shadowing.path_embedding import (
    PathEmbedding, ContextManagerBase, PredictionContext, ArrayType
)
from shadowing.path_shadowing.path_distance import PathDistance


def _dim_array(x: ArrayType) -> ArrayType:
    """ Unsqueeze x to become of shape (B, C, T). """
    if x is None:
        return x
    if x.ndim == 1:  # assume time is provided
        return x[None, None, :]
    if x.ndim == 2:  # assume batch and time are provided
        return x[:, None, :]
    if x.ndim == 3:
        return x
    raise Exception("Array cannot be formatted to (B, C, T) shape.")


def _torch(x: ArrayType) -> torch.Tensor:
    """ Convert x to a torch float tensor. """
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=torch.float32)


def _numpy(x: ArrayType) -> np.ndarray:
    """ Convert x to a numpy float tensor. """
    if isinstance(x, np.ndarray):
        return x
    return x.cpu().numpy()


def select_cartesian_product(
    indices: torch.Tensor, 
    tensors: list[torch.Tensor]
) -> torch.Tensor:
    """ Select the cartesian product of the tensors at the given indices.
    This is equivalent to torch.cartesian_prod(*tensors)[indices] but does not 
    require storing the cartesian product torch.cartesian_prod(*tensors) in memory.

    :param indices: 1d-tensor (or 2d-tensor with batch dimension)
    :param tensors: list of 1d-tensors
    """
    dims = torch.tensor([t.shape[0] for t in tensors], dtype=torch.int64)
    factors = dims.flip(0).cumprod(-1).flip(0)
    factors = torch.cat([factors,torch.tensor([1])])
    coordinates = [(indices // f) % d for (d, f) in zip(dims, factors[1:])]
    return torch.stack([tsr[c] for (tsr, c) in zip(tensors, coordinates)], dim=-1)


class PathShadowing:
    """ Path shadowing object.

    Path Shadowing consists in scanning over a generated dataset of paths for 
    paths which are close to the observed context (e.g. past log-returns).
    It provides functionality for calculating the proximity between paths.

    Attributes:
        - embedding: reduces the dimensionality of a path
        - distance: measures distance between embedded paths
        - dataset: the dataset used for scanning for shadowing paths
        - context: specify what is shadowed and what is predicted 
            (e.g. prediction: in-context=past, out-context=future)
    """

    def __init__(
        self, 
        embedding: PathEmbedding,
        distance: PathDistance,
        dataset: ArrayType | Path | TimeSeriesDataset,
        context: ContextManagerBase | None = None,
    ):
        # dataset to scan for shadowing paths 
        if isinstance(dataset, Path):
            dataset = TimeSeriesDataset(dpath=dataset, R=None).load()
        if isinstance(dataset, TimeSeriesDataset):
            dataset = dataset.load()
        self.dataset = dataset

        # notion of proximity between paths
        self.embedding = embedding
        self.distance = distance

        # initialize context object: tells what to shadow in the context data
        self.context = context or PredictionContext(horizon=None)

    def batched_distance(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        k: int, 
        n_splits: int, 
        cuda: bool
    ) -> tuple[torch.Tensor,torch.Tensor]:
        """ Compute the k-smallest distances between x and y.
        Embed paths x -> h(x), y -> h(y) and retain 
        the k smallest distances d(h(x),h(y))
        where h is self.embedding, e.g. foveal multiscale
        where d is self.distance, e.g. Euclidian distance

        :param x: tensor of shape (B, C, T) 
            --- B: batch_size, number of paths
            --- C: data_channels, number of time-series
            --- T: number of time-steps in a time-series
        :param y: tensor of shape (S, C, T)
            --- S: dataset_size, number of long time-series in the dataset
            --- C: data_channels, number of time-series
            --- T: number of time-steps in a time-series
        :param k: number of smallest distances to keep
        :param n_splits: number of splits of the dataset
        :param cuda: if True, use cuda for accelerated computation
        """

        embedding = self.embedding
        distance = self.distance

        n_data = x.shape[0]  # nb of time-series to shadow
        S = y.shape[0]  # dataset size: nb of long data samples in the dataset
        d = self.embedding.kernel.shape[0]  # embedding dimension
        d = d if d != 0 else x.shape[-1]  # if embedding is the identity, use the data dimension

        if cuda:
            x = x.cuda()
            embedding = embedding.cuda()
            distance = distance.cuda()

        # embed: x -> h(x)
        x = embedding(x)[:,0,:]

        embedding = embedding.adjust_to_context(self.context)

        # the distances of closest paths and indices of the corresponding time-series in the dataset
        dists = x.new_empty(n_data, k).fill_(float("inf"))
        idces = torch.empty(n_data, k, y.ndim-1, dtype=torch.int32, device=x.device).fill_(-1)

        # split the dataset for memory purpose
        dataset_splits = torch.arange(S, dtype=torch.int32).split(S//n_splits)

        for split in dataset_splits:  # split: indices of the current batch
            
            # select a batch from the dataset
            y_batch = y[split,...]
            if cuda:
                y_batch = y_batch.cuda()
                split = split.cuda()

            # embed: y -> h(y)
            y_batch = embedding(y_batch)

            # compute the distance: d(h(x), h(y)), on a batch of y
            x_unsqueeze = x.view((n_data,) + (1,)*(y.ndim-1) + (d,))
            new_dists = distance(x_unsqueeze, y_batch[None,...])

            # get k-smallest distances on this batch
            new_dists, idces_tmp = torch.topk(new_dists.view(n_data,-1), k=k, dim=-1, largest=False)
            cartesian_prod_idces = [split] + [torch.arange(s, dtype=torch.int32, device=x.device) for s in y_batch.shape[1:-1]]
            new_idces = select_cartesian_product(idces_tmp.to(torch.int32), cartesian_prod_idces)

            # get k-smallest distances so far, including this batch
            dists = torch.cat([dists, new_dists], dim=1)
            idces = torch.cat([idces,new_idces], dim=1)
            dists, idces_tmp = torch.topk(dists, k=k, dim=-1, largest=False)
            idces = idces[torch.arange(n_data, dtype=torch.int32)[:,None], idces_tmp]

        if cuda:
            dists = dists.cpu()
            idces = idces.cpu()

        return dists, idces

    def shadow(
        self,
        x_context: ArrayType,
        k: int = 1,
        n_splits: int = 1,
        cuda: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Perform path shadowing on x_context with data from the dataset:
        scan the dataset for paths that are close to x_context. 

        :param x_context: (B, C, T) array, the B paths to "shadow"
        :param k: number of closest paths to keep
        :param n_splits: dataset splits: increase it to reduce memory usage
        :param cuda: if True, use cuda for accelerated computation
        """
        if self.embedding.kernel.shape[-1] != 0 and self.embedding.kernel.shape[-1] != x_context.shape[-1]:
            raise Exception("The embedding kernel should be of the same size as the context.")

        # data to "shadow"
        x_context = _dim_array(x_context)
        x_context = _torch(x_context)

        # dataset scanned for shadowing paths
        x_dataset = _dim_array(self.dataset)
        x_dataset = _torch(x_dataset)

        # computing k-smallest distances
        d_smallest, i_smallest = self.batched_distance(x_context, x_dataset, k, n_splits, cuda)

        # collect closest paths (with their out-context)
        ts = torch.arange(x_context.shape[-1]+self.context.get_out_times(), dtype=torch.int32)
        locators = i_smallest[:,:,0] * x_dataset.shape[-1] + i_smallest[:,:,1]
        indices = locators[:,:,None] + ts[None,None,:]
        x_dataset = rearrange(x_dataset, 'r c t -> c (r t)')
        paths_smallest = x_dataset[:,indices]
        paths_smallest = rearrange(paths_smallest, 'c b k t -> b k c t')

        return _numpy(d_smallest), _numpy(paths_smallest), _numpy(i_smallest)
    
    @staticmethod
    def init_averaging_proba(
        proba_name: str,
        distances: np.ndarray,
        eta: float | None
    ) -> DiscreteProba:
        """ The averaging probability used for averaging out-context predictions. """
        if proba_name == "uniform":
            return Uniform()
        elif proba_name == "softmax":
            return Softmax(distances, eta)
        else:
            raise ValueError("Unrecognized averaging proba")

    def predict_from_paths(
        self,
        distances: np.ndarray,
        paths: np.ndarray,
        to_predict: Callable,
        proba_name: str,
        eta: float | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Agregate predictions on shadowing paths. """

        # extract out-context e.g. future of log-returns for prediction
        paths = self.context.select_out_context(paths)

        # define averaging operators
        empirical_proba = self.init_averaging_proba(proba_name, distances[:,:,None], eta)

        # get prediction through weighted average on shadowing paths
        predictions = empirical_proba.avg(to_predict(paths), axis=1)
        predictions_std = empirical_proba.std(to_predict(paths), axis=1)

        return predictions, predictions_std

    def predict(
        self,
        x_context: ArrayType,
        k: int,
        to_predict: Callable,
        eta: float | None = None,
        proba_name: str = "softmax",
        n_dataset_splits: int = 1,
        n_context_splits: int = 1,
        cuda: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prediction on shadowing paths. Obtained by averaging out-context data
        from shadowing paths whose in-context, matches, or "shadows", 
        the observed data: x_context.

        :param x_context: (B, C, T) array, the B paths to "shadow"
        :param k: threshold defining "close" paths
        :param to_predict: quantity to predict
        :param eta: parameter used for a softmax averaging
        :param proba_name: name of the averaging proba
        :param n_dataset_splits: dataset splits: increase it to reduce memory usage
        :param n_context_splits: context splits: increase it to reduce memory usage
        :return:
        """
        x_context = _dim_array(x_context)
        x_context = _torch(x_context)

        B = x_context.shape[0]  # nb of paths to shadow

        splits = torch.arange(B).split(B//n_context_splits)

        predictions, predictions_std = [], []

        for bs in tqdm(splits):

            # perform path shadowing (in-context)
            distances, paths, _ = self.shadow(x_context[bs,...], k, n_dataset_splits, cuda)

            # aggregate predictions (out-context)
            res = self.predict_from_paths(distances, paths, to_predict, proba_name, eta)

            predictions.append(res[0])
            predictions_std.append(res[1])

        return np.concatenate(predictions), np.concatenate(predictions_std)
