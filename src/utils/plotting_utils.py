import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot2ts(x1, x2, figsize=(20, 5), linewidth=0.5, ylim=None, labels=['real', 'generated'], legend=True):
    """ Plot two time-series. """
    if x1.size != x2.size:
        print("WARNING. The two plotted time-series are of different size.")

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    axes[0].plot(x1, color='lightskyblue', linewidth=linewidth, label=labels[0])
    axes[1].plot(x2, color='coral', linewidth=linewidth, label=labels[1])

    # same y axis limits
    if ylim is None:
        ylim = min(ax.get_ylim()[0] for ax in axes), max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(-ylim, ylim)

    if legend:
        for ax in axes:
            ax.legend()

    return axes


def plot2Rx(x1, x2, phi, ploter, J=None, labels=['real', 'generated']):
    """ Plot scattering spectra of two time-seirs. """
    Rx1 = phi(x1, J=J)
    Rx2 = phi(x2, J=J)

    axes = ploter([Rx1, Rx2], labels=labels)

    return axes
