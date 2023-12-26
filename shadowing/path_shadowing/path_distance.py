from typing import Tuple
from abc import abstractmethod
from itertools import product
import torch
import torch.nn as nn


class PathDistance(nn.Module):

    def forward_topk(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        k: int,
        n_splits: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Compute the k-smallest distances between x and y.

        :param x: tensor (B1, ..., d)
        :param y: tensor (B2, ..., d)
        :param diag: If True, compute the diagonal of the distance matrix
        :param n_splits: Number of splits for batch computation
        :return: 
            - tensor (B1, k): k smallest distances
            - tensor (B1, k, y.ndim-1): indices of the k smallest distances
        """
        splits = torch.arange(y.shape[0], device=x.device).split(y.shape[0]//n_splits)
        ds = x.new_empty(x.shape[0], k).fill_(float('inf'))
        idces = torch.empty(x.shape[0], k, y.ndim-1, dtype=torch.int32, device=x.device).fill_(2147483647)

        for bs in splits:

            # the new batch of distances and their indices
            x_unsqueeze = x.view((x.shape[0],) + (1,)*(y.ndim-1) + (x.shape[-1],))
            new_ds = self(x_unsqueeze, y[None,bs,...])
            new_idces = product(bs, *[range(s) for s in y.shape[1:-1]])
            new_idces = torch.tensor(list(new_idces), device=x.device)
            new_idces = new_idces.expand(x.shape[0], *new_idces.shape)

            # regroup smallest distances so far with the new ones
            ds = torch.cat([ds, new_ds.view(x.shape[0],-1)], dim=1)
            idces = torch.cat([idces, new_idces], dim=1)

            # get k-smallest distances so far with the new batch
            ds, idces_tmp = torch.topk(ds, k=k, dim=-1, largest=False)
            batch_indices = torch.arange(x.shape[0])[:,None]
            idces = idces[batch_indices, idces_tmp]

        return ds, idces

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Compute distance between x and y on a y batch.

        :param x_obs: tensor (T,), represents the observed path
        :param x_synt: tensor (..., T), represents Q consecutive paths on n_traj different syntheses
        :return: tensor (...), represents the calculated distances
        """
        pass


class RelativeMSE(PathDistance):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).norm(dim=-1) / x.norm(dim=-1) 
