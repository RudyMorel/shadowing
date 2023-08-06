import numpy as np


class PathEmbedding:

    name = None

    def __call__(self, x):
        pass

    def astype(self, dtype):
        return self

    def get_unique_name(self):
        return self.name


class CustomEmbedding(PathEmbedding):
    def __init__(self, custom_name, custom_embedding):
        super(CustomEmbedding, self).__init__()
        self.name = custom_name
        self.custom_embedding = custom_embedding

    def __call__(self, x):
        return self.custom_embedding(x)


class Identity(PathEmbedding):

    name = "identity"

    def __init__(self, dim):
        super(Identity, self).__init__()
        self.dim = dim

    def __call__(self, x):
        return x[..., -self.dim:]


class AverageVol(PathEmbedding):

    name = "avgvol"

    def __call__(self, x):
        return (256 * (x ** 2).mean(-1, keepdims=True)) ** 0.5


def foveal_embedding(x, slices, beta):
    multi_scale_past_averages = np.empty((*x.shape[:-1], len(slices)), dtype=x.dtype)
    norms = np.empty(len(slices), dtype=x.dtype)
    for i_sl, sl in enumerate(slices):
        wx = x[..., sl]
        n = wx.shape[-1]
        multi_scale_past_averages[..., i_sl] = wx.sum(-1)
        norms[i_sl] = n ** beta
    return multi_scale_past_averages / norms


class FovealDisjoint(PathEmbedding):

    name = "foveal_disjoint"

    def __init__(self, alpha, beta, cut):
        super(FovealDisjoint, self).__init__()
        if alpha < 1.0:
            raise ValueError("Alpha should be >= 1.0 in Foveal model.")

        self.alpha = alpha
        self.beta = beta
        self.cut = cut

        dim = 1
        slices = []
        left, right = -1, None
        slices.append(slice(left, right))
        while True:
            right = left
            left = left - int(alpha ** dim)
            if -left > cut:
                break
            slices.append(slice(left, right))
            dim += 1

        self.slices = slices
        self.dim = len(slices)

    def get_unique_name(self):
        return self.name + f'_alpha{self.alpha:.3f}_beta{self.beta:.3f}_cut{self.cut}'.replace('.', '_')

    def __call__(self, x):
        """ Return multiscale averages of x on past consecutive dyadic periods. """
        if -self.slices[-1].start > x.shape[-1]:
            raise ValueError("Path is too short for this Multiscale embedding, try to reduce dyadic_size or offset.")

        return foveal_embedding(x, self.slices, self.beta)


class FovealFixed(PathEmbedding):
    name = "foveal_fixed"

    def __init__(self, alpha, beta, cut):
        super(FovealFixed, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.cut = cut

        self.dim = int(np.floor(np.log(cut) / np.log(alpha)))

        lengths = [int(alpha ** n) for n in range(1, 1 + self.dim)]
        self.slices = [slice(-le, None) for le in lengths]

    def get_unique_name(self):
        return self.name + f'_alpha{self.alpha:.3f}_beta{self.beta:.3f}_cut{self.cut}'.replace('.', '_')

    def __call__(self, x):
        """ Return multiscale averages of x on past consecutive dyadic periods. """
        if -self.slices[-1].start > x.shape[-1]:
            raise ValueError("Path is too short for this Multiscale embedding, try to reduce dyadic_size or offset.")
        return foveal_embedding(x, self.slices, self.beta)


EMBEDDING_CHOICE = {
    Identity.name: Identity,
    AverageVol.name: AverageVol,
    FovealDisjoint.name: FovealDisjoint,
    FovealFixed.name: FovealFixed,
}
