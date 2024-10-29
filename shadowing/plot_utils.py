import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from scatspectra import Softmax


def plot_closest(
    dlnx_current: np.ndarray,
    close_paths: np.ndarray,
    num_trajectories: int = 20,
    color_decay: float = 1.2,
    date=None, 
    color: str = 'blue'
) -> None:
    """ Plot the closest paths to the current time-series

    :param dlnx_current: 1d array, the current log-return time-sries
    :param close_paths: 3d array (k_closest, 1, T), the close paths
    :param ntrajs: int, the number of close paths to plot
    :param color_decay: float, the decay of the color of the close paths
    :param date: pandas datetime, the date of the current log-return time-series
    :param color: str, the color of the current time-series
    """
    # infer horizon
    w_past = dlnx_current.shape[-1]
    horizon = close_paths.shape[-1] - w_past

    # plot 
    plt.figure(figsize=(4,2))
    plt.plot(np.arange(-w_past+1,1), dlnx_current, color=color, label=r'$\mathrm{present}$')
    for i in range(min(close_paths.shape[0], num_trajectories)):
        label = None if i > 0 else r'$\mathrm{generated}$'
        plt.plot(np.arange(-w_past+1,horizon+1), close_paths[i,0,:], alpha=0.5/(color_decay**i), color='black', label=label)
    plt.grid(None)
    plt.xlim(-2-w_past,horizon+2)
    plt.axhline(0.0, color='black', linewidth=0.5)
    plt.axvline(0.0, color='black', linestyle='dashed', linewidth=1.5)
    plt.gca().tick_params(axis='x', labelsize=20)
    plt.gca().tick_params(axis='y', labelsize=15)
    ylim = np.abs(dlnx_current).max() * 1.1
    plt.ylim(-ylim,ylim)
    plt.locator_params(axis='x', nbins=6)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    plt.legend(loc='lower right', fontsize=10)
    if date is not None:
        plt.title(r'$\mathrm{' + date.strftime('%Y/%m/%d') + '}$', fontsize=20, color=color)


def plot_shadow(
    dlnx_current: np.ndarray, 
    distances: np.ndarray, 
    close_paths: np.ndarray, 
    eta: float, 
    date=None, 
    color='blue'
) -> None:
    """ 
    Plot the shadow of a path, defined as the area within 1sigma of 
    the Gaussian average of the close paths.

    :param dlnx_current: 1d array, the current log-return time-sries 
    :param distances: 1d array, the distances between the current log-return time-series and the close paths
    :param close_paths: 3d array (k_closest, 1, T), the close paths
    :param eta: float, the width of a Gaussian in the Gaussian average 
    :param date: pandas datetime, the date of the current log-return time-series
    :param color: str, the color of the current time-series 
    """
    # infer horizon
    w_past = dlnx_current.shape[-1]
    horizon = close_paths.shape[-1] - w_past

    # the proba used to average 
    proba = Softmax(distances=distances, eta=eta)
    mean = proba.avg(close_paths, axis=0)[0,:]
    std = proba.std(close_paths, axis=0)[0,:]
    
    # the shadow lower-bound and upper-bound
    lower_bound = mean - std
    upper_bound = mean + std

    # plot
    plt.figure(figsize=(4,2))
    plt.plot(np.arange(-w_past+1,1), dlnx_current, color=color, label=r'$\mathrm{present}$')
    plt.fill_between(np.arange(-w_past+1,horizon+1),lower_bound, upper_bound, color='gray', alpha=0.5, label=r'$\mathrm{shadow}$');
    plt.grid(None)
    ylim = np.abs(dlnx_current).max() * 1.1
    plt.ylim(-ylim,ylim)
    plt.xlim(-2-w_past,horizon+2)
    plt.axhline(0.0, color='black', linewidth=0.5)
    plt.axvline(0.0, color='black', linestyle='dashed', linewidth=1.5)
    plt.gca().tick_params(axis='x', labelsize=20)
    plt.gca().tick_params(axis='y', labelsize=15)
    plt.locator_params(axis='x', nbins=6)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    plt.legend(loc='lower right', fontsize=10)
    plt.xlabel(r'$\mathrm{day}$', fontsize=20)
    if date is not None:
        plt.title(r'$\mathrm{' + date.strftime('%Y/%m/%d') + '}$', fontsize=20, color=color)


def plot_volatility(
    dlnx_current: np.ndarray, 
    standard_dev: np.ndarray, 
    Ts: list | np.ndarray, 
    distances: np.ndarray, 
    close_paths: np.ndarray, 
    eta: float, 
    date=None, 
    color='blue', 
    color_vol='black'
) -> None:
    """ Plot the predicted volatility vols of the current history dlnx_current
    
    :param dlnx_current: 1d array, the current log-return time-sries
    :param vols: 2d array (len(Ts), T), the predicted volatilities
    :param Ts: list of int, the horizons of the predicted volatilities
    :param date: pandas datetime, the date of the current log-return time-series
    :param color: str, the color of the current time-series
    :param color_vol: str, the color of the predicted volatilities
    . """

    # infer horizon
    w_past = dlnx_current.shape[-1]
    horizon = close_paths.shape[-1] - w_past

    # the proba used to average 
    proba = Softmax(distances=distances, eta=eta)
    mean = proba.avg(close_paths, axis=0)[0,:]
    std = proba.std(close_paths, axis=0)[0,:]
    
    # the shadow lower-bound and upper-bound
    lower_bound = mean - std
    upper_bound = mean + std

    # plot 
    plt.figure(figsize=(4,2))
    plt.plot(np.arange(-w_past+1,1), dlnx_current, color=color, label=r'$\mathrm{present}$')
    plt.fill_between(np.arange(-w_past+1,1),lower_bound[:w_past], upper_bound[:w_past], color='gray', alpha=0.5, label=r'$\mathrm{shadow}$');
    for i_T, T in enumerate(Ts):
        label = r'$\mathrm{vol~prediction}$' if i_T == 0 else None
        plt.fill_between(np.arange(T+1), -standard_dev[i_T], standard_dev[i_T], color=color_vol, alpha=0.1, label=label)
    plt.grid(None)
    ylim = np.abs(dlnx_current).max() * 1.1
    plt.ylim(-ylim,ylim)
    plt.xlim(-2-w_past,horizon+2)
    plt.axhline(0.0, color='black', linewidth=0.5)
    plt.axvline(0.0, color='black', linestyle='dashed', linewidth=1.5)
    plt.gca().tick_params(axis='x', labelsize=20)
    plt.gca().tick_params(axis='y', labelsize=15)
    plt.locator_params(axis='x', nbins=6)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    plt.legend(loc='lower right', fontsize=10)
    if date is not None:
        plt.title(r'$\mathrm{' + date.strftime('%Y/%m/%d') + '}$', fontsize=20, color=color)