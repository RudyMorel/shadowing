""" Utils function for complex. """
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function


def real(z):
    return z[..., 0]


def imag(z):
    return z[..., 1]


def conjugate(z):
    z_copy = z.clone()
    z_copy[..., 1] = -z_copy[..., 1]
    return z_copy


def from_real(x):
    return torch.stack([x, torch.zeros_like(x)], dim=-1)


def from_imag(x):
    return torch.stack([torch.zeros_like(x), x], dim=-1)


def to_c(a, b):
    return torch.stack([a, b], dim=-1)


def to_r2(z):
    return real(z), imag(z)


def eitheta(theta):
    """ Return e^i theta. """
    return to_c(torch.cos(theta), torch.sin(theta))


def diag(z):
    real_diag = torch.diag(real(z))
    imag_diag = torch.diag(imag(z))
    return torch.stack([real_diag, imag_diag], dim=-1)


def inv(z):
    return conjugate(z) / (z**2).sum(dim=-1, keepdim=True)


def minv(A):
    """ Complex matrix inversion, by using numpy library. Not optimized. """
    if torch.cuda.is_available():
        A = A.detach().x.cpu().numpy()
    else:
        A = A.detach().numpy()
    A = A[..., 0] + 1j * A[..., 1]

    Ainv = np.linalg.inv(A)
    return from_np(Ainv)


def mul_real(z1, z2):
    """ Efficient calculation of real(z1 * z2) """
    zr = real(z1) * real(z2) - imag(z1) * imag(z2)
    return zr


def mul_imag(z1, z2):
    """ Efficient calculation of imag(z1 * z2) """
    zi = real(z1) * imag(z2) + imag(z1) * real(z2)
    return zi


def mul(z1, z2):
    zr = real(z1) * real(z2) - imag(z1) * imag(z2)
    zi = real(z1) * imag(z2) + imag(z1) * real(z2)
    z = torch.stack((zr, zi), dim=-1)
    return z


def mmm(z1, z2, z3):
    return mm(z1, mm(z2, z3))


def modulus(z):
    z_mod = z.norm(p=2, dim=-1)
    return z_mod


def norm(z, p=2, dim=-2):
    z_abs = modulus(z).unsqueeze(-1)
    z_norm = torch.norm(z_abs, p=p, dim=dim, keepdim=True)
    return z_norm


def mm(z1, z2):
    """
    Complex matrix multiplication

    :param z1: tensor of size (B*, M, N, 2)
    :param z2: tensor of size (B*, N, P, 2)
    :return: tensor of size (B*, M, P, 2)
    """
    zr = real(z1) @ real(z2) - imag(z1) @ imag(z2)
    zi = imag(z1) @ real(z2) + real(z1) @ imag(z2)
    z = torch.stack((zr, zi), dim=-1)
    return z


def relu(z):
    zr = F.relu(real(z))
    return torch.stack([zr, torch.zeros_like(zr)], dim=-1)


class StablePhaseExp(Function):
    @staticmethod
    def forward(ctx, z, r):
        eitheta = z / r
        eitheta.masked_fill_(r == 0, 0)

        ctx.save_for_backward(eitheta, r)
        return eitheta

    @staticmethod
    def backward(ctx, grad_output):
        eitheta, r = ctx.saved_tensors

        dldz = grad_output / r
        dldz.masked_fill_(r == 0, 0)

        dldr = - eitheta * grad_output / r
        dldr.masked_fill_(r == 0, 0)
        dldr = dldr.sum(dim=-1).unsqueeze(-1)

        return dldz, dldr


phaseexp = StablePhaseExp.apply


class StablePhase(Function):
    @staticmethod
    def forward(ctx, z):
        z = z.detach()
        x, y = real(z), imag(z)
        r = z.norm(p=2, dim=-1)

        # NaN positions
        eps = 1e-32
        mask_real_neg = (torch.abs(y) <= eps) * (x <= 0)
        mask_zero = r <= eps

        x_tilde = r + x
        # theta = torch.atan(y / x_tilde) * 2
        theta = torch.atan2(y, x)

        # relace NaNs
        theta.masked_fill_(mask_real_neg, np.pi)
        theta.masked_fill_(mask_zero, 0.)

        # ctx.save_for_backward(x.detach(), y.detach(), r.detach())
        ctx.save_for_backward(x, y, r, mask_real_neg, mask_zero)
        return theta

    @staticmethod
    def backward(ctx, grad_output):
        x, y, r, mask_real_neg, mask_zero = ctx.saved_tensors

        # some intermediate variables
        x_tilde = r + x
        e = x_tilde ** 2 + y ** 2

        # derivative with respect to the real part
        dtdx = - y * x_tilde * 2 / (r * e)
        mask_real_neg_bis = (torch.abs(y) == 0) * (x <= 0)
        dtdx.masked_fill_(mask_real_neg, 0)
        dtdx.masked_fill_(mask_zero, 0)

        # derivative with respect to the imaginary part
        dtdy = x * x_tilde * 2 / (r * e)
        dtdy[mask_real_neg] = -1 / r[mask_real_neg]
        # dtdy.masked_fill_(mask, 0)
        dtdy.masked_fill_(mask_zero, 0)

        dtdz = grad_output.unsqueeze(-1) * torch.stack((dtdx, dtdy), dim=-1)
        return dtdz


phase = StablePhase.apply


def from_np(z_np, tensor=torch.DoubleTensor):
    z = np.stack((np.real(z_np), np.imag(z_np)), axis=-1)
    return tensor(z)


def to_np(z_torch):
    if z_torch.is_cuda:
        z_np = z_torch.detach().cpu().numpy()
    else:
        z_np = z_torch.detach().numpy()
    return z_np[..., 0] + 1j * z_np[..., 1]


def adjoint(z, dims=(0, 1)):
    return conjugate(z.transpose(*dims))


def ones_like(z):
    re = torch.ones_like(z[..., 0])
    im = torch.zeros_like(z[..., 1])
    return torch.stack((re, im), dim=-1)


def pows(z, max_k, dim=0):
    z_pows = [ones_like(z)]
    if max_k > 0:
        z_pows.append(z)
        z_acc = z
        for k in range(2, max_k + 1):
            z_acc = mul(z_acc, z)
            z_pows.append(z_acc)
    z_pows = torch.stack(z_pows, dim=dim)
    return z_pows


def log2_pows(z, max_pow_k, dim=0):
    z_pows = [ones_like(z)]
    if max_pow_k > 0:
        z_pows.append(z)
        z_acc = z
        for k in range(2, max_pow_k + 1):
            z_acc = mul(z_acc, z_acc)
            z_pows.append(z_acc)            
    assert len(z_pows) == max_pow_k + 1
    z_pows = torch.stack(z_pows, dim=dim)
    return z_pows
