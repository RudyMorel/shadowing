from typing import Callable 
from abc import abstractmethod
import numpy as np


class PathDistance:

    name = None

    @abstractmethod
    def __call__(self, 
                 x_obs: np.ndarray, 
                 x_synt: np.ndarray) -> np.ndarray:
        """ Distance between observed path and syntheses paths.

        :param x_obs: array (T,), observed path
        :param x_synt: array (..., T), observed path Q consecutive paths on n_traj different syntheses
        :return: array (...)
        """
        pass

    def get_unique_name(self):
        return self.name


class CustomDistance(PathDistance):

    def __init__(self, 
                 custom_name: str, 
                 custom_distance: Callable):
        super(CustomDistance, self).__init__()
        self.name = custom_name
        self.custom_distance = custom_distance

    def __call__(self, x_obs, x_synt):
        return self.custom_distance(x_obs, x_synt)


class MSE(PathDistance):

    name = "MSE"

    def __call__(self, 
                 x_obs: np.ndarray, 
                 x_synt: np.ndarray):
        mse = ((x_obs - x_synt) ** 2).mean(-1) ** 0.5
        return mse


class RelativeMSE(PathDistance):

    name = "relativeMSE"

    def __init__(self):
        super(RelativeMSE, self).__init__()

    def __call__(self, 
                 x_obs: np.ndarray, 
                 x_synt: np.ndarray):
        mse = ((x_obs - x_synt) ** 2).mean(-1) ** 0.5
        l2_obs = (x_obs ** 2).mean(-1) ** 0.5
        return mse / l2_obs


DISTANCE_CHOICE = {
    MSE.name: MSE,
    RelativeMSE.name: RelativeMSE
}
