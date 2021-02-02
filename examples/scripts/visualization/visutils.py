# -*- coding: utf-8 -*-
"""
Visualization utilities to display clustering regions and distributions
on 2D data.
"""
# Author: Juan S. Lara <julara@unal.edu.co>
# License: MIT

import numpy as _np
import matplotlib.pyplot as _plt
from matplotlib.backends.backend_pdf import PdfPages as _PdfPages
import ternary as _ternary
from ternary.helpers import simplex_iterator as _simplex_iterator

def _make_grid(X, n_points=256):
    """
    This function makes a grid from the original data points.

    Parameters
    ----------
    X : array-like, shape=(n_samples, 2)
        Input data matrix.
    n_points : int, default=256
        Number of points on each axis of the grid.

    Returns
    -------
    X_grid: array-like, shape=(n_points**2, 2)
        Generated grid points.
    X1: array-like, shape=(n_points, n_points)
        First axis grid.
    X2 : array-like, shape=(n_points, n_points)
        Second axis grid.
    """

    # ranges for each axis
    ran_x1 = X[:, 0].max() - X[:, 0].min()
    ran_x2 = X[:, 1].max() - X[:, 1].min()

    # linspace on each axis
    x1 = _np.linspace(
            X[:, 0].min() - ran_x1 * 0.05,
            X[:,0].max() + ran_x1 * 0.05,
            n_points
            )

    x2 = _np.linspace(
            X[:, 1].min() - ran_x2 * 0.05,
            X[:, 1].max() + ran_x2 * 0.05,
            n_points
            )

    # combined grid
    X1, X2 = _np.meshgrid(x1, x2)

    # resultant grid matrix
    X1_flat = X1.reshape(-1, 1)
    X2_flat = X2.reshape(-1, 1)
    return _np.hstack([X1_flat, X2_flat]), X1, X2

def _imshow_regions(X_grid, y, ax, shape, cmap="ocean"):
    """
    Displays an image with the decision regions.

    Parameters
    ----------
    X_grid: array-like, shape=(n_points**2, 2)
        Generated grid points.
    y : array-like, shape=(n_points**2, )
        Colors for each data point.
    ax : matplotlib.axes.Axes
        Matplotlib axes object.
    shape : tuple
        Image shape.
    cmap : str
        Matplotlib colormap.
    """

    ax.imshow(
            y.reshape(shape), interpolation='nearest', 
            extent=(
                X_grid[:, 0].min(), X_grid[:, 0].max(), 
                X_grid[:, 1].min(), X_grid[:, 1].max()
                ),
            cmap=cmap, aspect='auto', origin='lower', 
            alpha=0.5
            )
    ax.axis("off")

def _scatter_data(X, ax, x1lims, x2lims):
    """
    Generates a scatter plot from the input matrix in the specified axis.

    Parameters
    ----------
    X : array-like, shape=(n_samples, 2)
        Input data matrix
    ax : matplotlib.axes.Axes
        Matplotlib axes object.
    x1lims : list
        Limits of first axis.
    x2lims : list
        Limits of second axis.
    """
    ax.scatter(
            X[:, 0], X[:, 1], 
            alpha=0.4, c="k"
            )
    ax.set_xlim(x1lims)
    ax.set_ylim(x2lims)


def decision_region(
        encoder_model, title, X, 
        figsize=(10, 10), batch_size=32, 
        n_points=256):
    """
    Plots the decision region from the soft-assignments

    Parameters
    ----------
    encoder_model : tf.keras.Model
        Keras models to compute the soft-assignments.
    title : str
        Figure title.
    X : array-like, shape=(n_samples, 2)
        Input data matrix
    figsize : tuple, default=(10, 10)
        Figure size
    batch_size : int, default=32
        Batch size for the keras model predictions.
    n_points : int, default=256
        Number of points on each axis of the grid.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure.
    ax : matplotlib.axes.Axes
        Matplotlib axes.
    """

    # figure definition
    fig, ax = _plt.subplots(1, 1, figsize=figsize)
    
    # make grid
    X_grid, _, _ = _make_grid(X, n_points)

    # soft-assignments for the grid
    preds = encoder_model.predict(
            X_grid, batch_size=batch_size
            )

    # hard-assignments for the grid
    y_pred = _np.argmax(preds, axis=1)

    # decision regions
    _imshow_regions(
            X_grid, y_pred, ax,
            (n_points, n_points)
            )

    # scatter dataset
    _scatter_data(
            X, ax, 
            [X_grid[:, 0].min(), X_grid[:, 0].max()],
            [X_grid[:, 1].min(), X_grid[:, 1].max()],
            )
    
    return fig, ax

def probability_region(
        encoder_model, title, X,
        n_clusters, figsize=(10, 10), 
        batch_size=32, rows=2, cols=2,
        n_points=256):
    """
    Generates a figure to visualize the posterior distributions.

    Parameters
    ----------
    encoder_model : tf.keras.Model
        Keras model to compute the soft-assignments.
    title : str
        Figure title.
    X : array-like, shape=(n_samples, 2)
        Input data matrix.
    n_clusters : int
        Number of clusters.
    figsize : tuple, default=(10, 10)
        Figure size.
    batch_size : int, default=32
        Batch size for the keras model predictions.
    rows : int
        Number of subplot rows.
    cols: int
        Number of subplot columns.
    n_points : int, default=256
        Number of points on each axis of the grid.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure.
    ax : matplotlib.axes.Axes
        Matplotlib axes.
    """
    # make grid
    X_grid, X1, X2 = _make_grid(X, n_points)

    # soft-assignments for the grid
    preds = encoder_model.predict(
            X_grid, batch_size=batch_size
            )

    # creates a figure with subplots        
    fig, ax = _plt.subplots(
            rows, cols, 
            figsize=figsize
            )
    fig.suptitle(title)

    for i in range(n_clusters):
        axi = ax[i//cols, i%cols] if rows>1 else ax[i]
        _imshow_regions(
                X_grid, preds[:, i], axi,
                (n_points, n_points), 
                "Blues"
                )
        axi.set_title(f"$P(z_{i+1}"+"=1|\mathbf{X})$")

        _scatter_data(
                X, axi, 
                [X_grid[:, 0].min(), X_grid[:, 0].max()],
                [X_grid[:, 1].min(), X_grid[:, 1].max()],
                )
        axi.contour(
                X1, X2,
                preds[:, i].reshape(X1.shape), 
                colors='k', linestyles='-')
    return fig, ax

def _make_dict(y_pred, scale):
    #TODOC
    mapping = [
            (0.5, 0.75, 0.5),
            (1, 1, 1),
            (0.5, 0.56, 0.62),
            (0.62, 0.81, 0.87)
            ]

    data = dict()
    for pos, simplex_point in enumerate(_simplex_iterator(scale)):
        data[simplex_point] = mapping[
                int(y_pred[pos])
                ]
    return data

def decision_simplex(
        encoder_model, title, X,
        scale=100, batch_size=32,
        rows=2, cols=2):
    # TODOC
    X_grid = _np.array(
            list(
                _simplex_iterator(scale)
                )
            ) / scale

    fig, ax = _ternary.figure(scale=scale)

    preds = encoder_model.predict(X_grid)
    y_pred = _np.argmax(preds, axis=1)

    data = _make_dict(y_pred, scale)

    ax.heatmap(
            data, 
            style="triangle",
            cmap="ocean", 
            use_rgba=True,
            )

    # TORM
    ax.boundary()
    fig.savefig("im1.pdf")
    fig, ax = _ternary.figure(scale=scale)
    #

    ax.scatter(
            X*scale,
            color=(0, 0, 0, 0.5), 
            zorder=2,
            s=10
            )
    ax.boundary()

    #TORM
    fig.savefig("im2.pdf")
    # 

    ax.set_title(title)
    return fig, ax


def save_pdf(figs, filename):
    """
    Generates a pdf file from a list o figures.

    Parameters
    ----------
    figs : list
        List of mathplotlib figures.
    filename :
        Filename to save the pdf file.
    """
    with _PdfPages(filename) as pdf:
        for fig in figs:
            pdf.savefig(fig)
