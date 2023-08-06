""" General usefull nn Modules"""
from typing import *
import numpy as np
import torch
import torch.nn as nn

from src.layers.described_tensor import DescribedTensor


class TensorProduct(nn.Module):
    """ Tensor product application x -> x \times y = (x1,...,xr, y1, ..., yn)  """
    def __init__(self, x: torch.tensor) -> None:
        super(TensorProduct, self).__init__()
        self.register_buffer("x", x)

    def forward(self, y: torch.tensor):
        return torch.cat([self.x, y], dim=-1)


class NormalizationLayer(nn.Module):
    """ Divide certain dimension by specified values. """
    def __init__(self,
                 dim: int,
                 sigma: torch.tensor,
                 on_the_fly: bool) -> None:
        super(NormalizationLayer, self).__init__()
        self.dim = dim
        self.sigma = sigma
        self.on_the_fly = on_the_fly

    def forward(self, x: torch.tensor) -> torch.tensor:
        if self.on_the_fly:  # normalize on the fly
            sigma = torch.abs(x).pow(2.0).mean(-1, keepdim=True).pow(0.5)
            # sigma = torch.abs(x).pow(2.0).pow(0.5)
            return x / sigma
        # test = self.sigma[(..., *(None,) * (x.ndim - 1 - self.dim))]
        return x / self.sigma[(..., *(None,) * (x.ndim - 1 - self.dim))]


class SkipConnection(nn.Module):
    """ Skip connection. """
    def __init__(self,
                 module: nn.Module) -> None:
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        y = self.module(x)
        return [x, y]


class Modulus(nn.Module):
    """ Modulus. """
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class ChunkedModule(nn.Module):
    """ Manage chunks on batch dimension. """
    def __init__(self,
                 module: nn.Module,
                 nchunks: int) -> None:
        super(ChunkedModule, self).__init__()
        self.nchunks = nchunks
        self.module = module

    def forward(self, x: torch.tensor) -> DescribedTensor:
        """
        Chunked forward on the batch dimension.

        :param x: B x ... tensor
        :return:
        """
        batch_split = np.array_split(np.arange(x.shape[0]), self.nchunks)
        Rxs = []
        for bs in batch_split:
            Rx_chunked = self.module(x[bs, ...])
            Rxs.append(Rx_chunked)
        return DescribedTensor(x=x, y=torch.cat([Rx.y for Rx in Rxs]), descri=Rxs[-1].descri, past=Rxs[-1]._past)


class PhaseOperator(nn.Module):
    """ Sample complex phases and creates complex phase channels. """
    def __init__(self, A):
        super(PhaseOperator, self).__init__()
        phases = torch.tensor(np.linspace(0, np.pi, A, endpoint=False))
        self.phases = torch.cos(phases) + 1j * torch.sin(phases)

    def cpu(self):
        self.phases = self.phases.cpu()
        return self

    def cuda(self):
        self.phases = self.phases.cuda()
        return self

    def forward(self, x):
        """ Computes Re(e^{i alpha} x) for alpha in self.phases. """
        return (self.phases[..., :, None] * x).real  # TODO. add relu
