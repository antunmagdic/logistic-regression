
import matplotlib.pyplot as plt
import numpy as np


def _get_colors(y, cmap):
    cmap = plt.get_cmap(cmap) if type(cmap) == str else cmap
    y2i = dict()
    i = 0
    for y_ in y:
        if y_ in y2i: continue
        y2i[y_] = i
        i += 1
    N = len(y2i)
    return np.array([cmap(y2i[y_] / (N-1))[0:3] for y_ in y])


def plot_classification(X, y_real, y_predicted=None, cmap='Paired', s=15, 
                        linewidth=0.7):
    """
    Plots given datapoints.

    If y_predicted is given and y_real[i] == y_predicted[i] marker for
    X[i] will be a circle, if y_real[i] != y_predicted[i] marker will 
    be a square. If y_predicted is not given markers for all X[i] will
    be circles.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
        Datapoints. n_samples is the number of samples.
    y_real : ndarray of shape (n_samples,)
        Real class values. n_samples is the number of samples. 
        y_real[i] is the real class of the datapoint X[i].
    y_predicted : ndarray of shape (n_samples,), optional
        Predicted class values. n_samples is the number of samples. 
        y_predicted[i] is the predicted class of the datapoint X[i].
    cmap : str or callable, default='Paired'
        Either a registered Colormap name or a callable that takes an
        int as the only argument and returns a tuple (r, g, b, a).
        r, g, b, a should be floats between 0 and 1. Values from y_real
        are sent to the colormap and return values are used as the 
        colors of corresponding datapoints.
    s : int, default=15
        The marker size in points**2.
    linewidth : float, default=0.7
        The linewidth of marker edges.
    """
    correct = None
    if y_predicted is None:
        correct = np.ones_like(y_real, dtype=np.bool)
    else:
        correct = y_real == y_predicted
    incorrect = np.logical_not(correct)
    colors = _get_colors(y_real, cmap)
    plt.scatter(X[correct, 0], X[correct, 1], marker='o', s=s, 
                linewidth=linewidth, edgecolors='black', c=colors[correct])
    plt.scatter(X[incorrect, 0], X[incorrect, 1], marker='s', s=s, 
                linewidth=linewidth, edgecolors='black', c=colors[incorrect])


def plot_surface(f, xrange=(-5, 15), yrange=(-5, 15), n=300, offset=0, 
                 linewidth=1, cmap='rainbow', levels=None, only_levels=False,
                 hide_levels=False):
    """
    Plots the heatmap and the contours of f(x, y) in the given range.

    Parameters
    ----------
    f : callable
        Takes an ndarray of shape (N, 2) where each row is an x, y pair
        for which value of f should be evaluated. f should return 
        values of f for all input pairs in a form of an ndarray of 
        shape (N,).
    xrange : (float, float), default=(-5, 15)
        Range of x values to be plotted.
    yrange : (float, float), default=(-5, 15)
        Range of y values to be plotted.
    n : int, default=300
        Resolution of heatmap. n x and n y uniform values will be used 
        to create a grid with n*n elements. f will be evaluated for 
        every pair of x and y.
    offset : float, default=0
        offset will be subtracted from all f values before plotting.
    linewidth : float, default=1
        Linewidth of contours.
    cmap : str, default='rainbow'
        Registered Colormap name.
    levels : iterable of floats, default=(offset,)
        Values of f for which contours should be plotted.
    only_levels : bool, default=False
        Only contours will be plotted (no heatmap).
    hide_levels : bool, default=False
        No contours will be plotted.
    """
    x_ = np.linspace(xrange[0], xrange[1], n)
    y_ = np.linspace(yrange[0], yrange[1], n)
    x, y = np.meshgrid(x_, y_)
    xy = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    z = f(xy).reshape(x.shape) - offset
    if not only_levels:
        zlimit = max(abs(np.min(z)), abs(np.max(z)))
        plt.pcolormesh(x, y, z, cmap=cmap, vmin=-zlimit, vmax=zlimit,
                       antialiased=True)
    if levels is None:
        levels = 0
    else:
        levels = [l - offset for l in levels]
    if not hide_levels:
        plt.contour(x, y, z, colors=('#222222'), linewidths=linewidth, 
                    antialiased=True, levels=levels, linestyles='solid')
