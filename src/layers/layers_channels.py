from itertools import product
import numpy as np
import torch
import torch.nn as nn

import src.utils.complex_utils as cplx


class LinearLayer(nn.Module):
    def __init__(self,
                 L: torch.tensor) -> None:
        super(LinearLayer, self).__init__()
        self.register_buffer("L", L)

    def forward(self, x: torch.tensor, c1=-4, c2=-1):
        """
        Perform Lx a linear transform on x along certain dimension.

        :param x: tensor (B) x T
        :param c1: the index of the 1st dimension to take product on
        :param c2: the index of the 2nd dimension to take product on
        :return:
        """
        L = self.L
        if x.is_complex():
            L = torch.complex(self.L, torch.zeros_like(self.L))
        c1p = c2 if c1 % x.shape[0] == -1 else c1
        x_temp = x.transpose(c2, -1).transpose(c1p, -2)
        return (L @ x_temp).transpose(c1p, -2).transpose(c2, -1)


class IdentityLayer(LinearLayer):
    def __init__(self) -> None:
        super(IdentityLayer, self).__init__(torch.eye(1))

    def forward(self, x, c1=-4, c2=-1):
        return x


class Orthogonal(LinearLayer):
    """Linear orthogonal operator."""

    def __init__(self,
                 N,
                 phase) -> None:
        self.N = N

        if N != 2:
            raise ValueError("Orthogonal layer supports only N=2")

        B = np.array([[np.cos(phase), np.sin(phase)], [-np.sin(phase), np.cos(phase)]])
        B = torch.DoubleTensor(B)

        super(Orthogonal, self).__init__(B)


class ScaleLayer(LinearLayer):
    """Handles transformations along scale axis."""

    def __init__(self,
                 sc_idxer,
                 Wl) -> None:
        super(ScaleLayer, self).__init__(None, Wl)  # Wl should belist of  Sr x Mr x J x J
        self.sc_idxer = sc_idxer

    def __call__(self, x, r, c1=None, c2=None):
        """
        Transforms channels m1...m{r-1} jr into m1...m{r-1} mr j{r+1}

        :param x: 1 x N x Jr x T x 2 tensor with all Jr band pass channels of order r : m1...m{r-1}jr
        :param r:
        :return: 1 x N x J{r+1} x T x 2 tensor with all J{r+1} channels of order r
        """
        # embed m1...m{r-1}jr in m1...m{r-1} x jr in order to vectorize transform along jr
        M0Mrm1, _, mapping1 = self.sc_idxer.residual_scales(r, r - 1, False)
        shape = x.shape[:2] + (M0Mrm1, self.sc_idxer.J * (self.sc_idxer.Q if r == 1 else 1)) + x.shape[-2:]
        y = torch.cuda.DoubleTensor(shape).fill_(0) if self.is_cuda else torch.DoubleTensor(shape).fill_(0)
        idx_r = self.sc_idxer.sc_paths[r - 1][~self.sc_idxer.low_pass_mask[r - 1], -1]
        idx2 = np.arange(mapping1.shape[0])
        y[..., mapping1, idx_r, :, :] = x[..., idx2, :, :]  # vector m1...m{r-1} x jr containing paths m1...m{r-1}jr

        # add the j{r+1} dimension
        y = torch.stack([y] * (self.sc_idxer.J + 1), dim=-3)  # only exact octaves are taken for j2, j3 ...

        # add the S dimension : taking into account the values of the sum m1 + ... + m{r-1}
        bkcondmax = self.sc_idxer.backward_conditions_unique_coded[r - 1].max_gap() + 1
        y = torch.stack([y] * bkcondmax, dim=-5)  # now y is 1 x N x |m1...m{r-1}| x S x jr x j{r+1} x T x 2

        # perform scale transform, m1...m{r-1} is (constant) batched, jr is the transform variable and j{r+1} is (possibly constant) batched
        y = super(ScaleLayer, self).__call__(y, -4, -2, r - 1)  # correspond to m0...m{r-1} x S x mr x j{r+1} x 2

        # project on m1...m{r-1} mr j{r+1}
        sums, p_idx_next = self.sc_idxer.backward_conditions_unique_coded[r - 1], self.sc_idxer.sc_paths[r]
        _, _, mapping2 = self.sc_idxer.residual_scales(r + 1, r - 1, True)
        y = y[..., mapping2, sums, p_idx_next[:, -2], p_idx_next[:, -1], :, :]
        # from layers.utils import to_numpy
        # test0 = to_numpy(y)
        # test1 = to_numpy(self.W[r-1])
        return y


def form_matrix(matrices, J, Q, r, Sr, Sunique, Mr, jr_csc, adapt_s, adapt_jrp1, csc):
    Qr = Q if r == 1 else 1
    eps = int(not csc) * (r - 1)
    Wr = torch.zeros((Sr, J + 1, (Mr + eps) * Qr, J * Qr, 2), dtype=torch.float64)
    for (s, jrp1) in product(range(Wr.shape[0]), range(r, Wr.shape[1])):
        j_min, j_max = (Sunique[s] + 1 if adapt_s else 0), (min(jr_csc, jrp1 if adapt_jrp1 else J)) * Qr
        if j_max <= j_min:
            continue
        i_trans = 0 if csc else j_min  # because in csc the Fourier modes starts at 0
        modes = min(j_max - j_min, Mr * Qr)
        Wr[s, jrp1, i_trans:modes + i_trans, j_min:j_max, :] = cplx.from_np(matrices[j_max - j_min][:modes])
    return Wr


class DiracScaleLayer(ScaleLayer):
    def __init__(self,
                 sc_idxer) -> None:
        Wl = []
        J, Q = sc_idxer.J, sc_idxer.Q
        matrices = [np.eye(dj) for dj in range(J * Q + 1)]
        for r in range(1, 3):
            Sr, Mr, jr_csc = sc_idxer.backward_conditions_unique_coded[r - 1].max_gap() + 1, sc_idxer.M[r - 1], \
                             sc_idxer.j_csc[r - 1]
            Wl.append(form_matrix(matrices, J, Q, r, Sr, sc_idxer.backward_conditions_unique[r - 1], Mr, jr_csc, True,
                                  sc_idxer.adaptative, False))
        super(DiracScaleLayer, self).__init__(sc_idxer, Wl)
