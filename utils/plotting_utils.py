#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'February 2020'

import plotly.graph_objects as go
import numpy as np


def visdom_plot_losses(viz, win, it, zero_nan=True, xylabel=('epoch', 'loss'), **kwargs):
    """Plot multiple loss curves
    """

    for name, value in kwargs.items():
        if value == 0. and zero_nan:
            value = np.nan
        viz.line(X=np.array([it]), Y=np.array([value]), win=win, update='append', name=name)

    viz.update_window_opts(win=win, opts={'title': win, 'legend': [name for name in kwargs.keys()],
                                          'xlabel': xylabel[0], 'ylabel': xylabel[1]})
    return win


def visdom_scatter(viz, title, X, y, sizes):
    """Create a scatter plot of samples
    X: embeddings
    Y: class labels
    """

    # make labels y in 1 .. K for visdom
    # yl = np.zeros_like(y)
    # for k, clid in enumerate(sorted(np.unique(y))):
    #     yl[y == clid] = k+1
    # plot! colorscale='Viridis'
    if isinstance(y, np.ndarray):
        labels = list(map(lambda x: int(x), y.flatten()))
    else:
        labels = y
    fig = go.Figure(data=[go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers',
        marker=dict(color=labels,
                    size=sizes),
    )])

    win = viz.plotlyplot(fig)
    viz.update_window_opts(win=win, opts={'title': title})
    return win


def visdom_scatter_update(viz, win, x, xylabel=('ep', 'prec'), **kwargs):

    for name, val in kwargs.items():
        viz.scatter(X=np.array([x]), Y=np.array([val]), win=win, update='append', name=name)

    viz.update_window_opts(win=win, opts={'title': win, 'xlabel': xylabel[0], 'ylabel': xylabel[1]})

    return win
