# -*- coding: utf-8 -*-
"""
The :mod:`dmae.dissimilarities` module implements several dissimilarity functions
in tensorflow. 
"""
# Author: Juan S. Lara <julara@unal.edu.co>
# License: MIT

import tensorflow as _tf
import itertools as _itertools
from dmae import normalizers as _normalizers

def euclidean(X, Y):
    """
    Computes a pairwise Euclidean distance between two matrices :math:`\mathbf{D}_{ij}=||\mathbf{x}_i-\mathbf{y}_j||`.

    Parameters
    ----------
    X : array-like, shape=(batch_size, n_features)
        Input batch matrix.
    Y : array-like, shape=(n_clusters, n_features)
        Matrix in which each row represents a centroid of a cluster.

    Returns
    --------
    Z : array-like, shape=(batch_size, n_clusters)
        Pairwise dissimilarity matrix.
    """

    func = lambda x_i: _tf.sqrt(
            _tf.reduce_sum((x_i-Y)**2, axis=1)
            )
    Z = _tf.vectorized_map(func, X)
    return Z

def cosine(X, Y):
    """
    Computes a pairwise cosine distance between two matrices :math:`\mathbf{D}_{ij}=(\mathbf{x}_i \cdot \mathbf{y}_j)/(||\mathbf{x}_i|| \cdot ||\mathbf{y}_j||)`.

    Parameters
    ----------
    X : array-like, shape=(batch_size, n_features)
        Input batch matrix.
    Y : array-like, shape=(n_clusters, n_features)
        Matrix in which each row represents a centroid of a cluster.

    Returns
    --------
    Z : array-like, shape=(batch_size, n_clusters)
        Pairwise dissimilarity matrix.
    """
   
    norm_X = _tf.nn.l2_normalize(X, axis=1)
    norm_Y = _tf.nn.l2_normalize(Y, axis=1)
    D = 1-_tf.matmul(norm_X, norm_Y, transpose_b=True)
    return D

def manhattan(X, Y):
    """
    Computes a pairwise Manhattan distance between two matrices :math:`\mathbf{D}_{ij}=\sum |\mathbf{x}_i|-|\mathbf{y}_j|`.

    Parameters
    ----------
    X : array-like, shape=(batch_size, n_features)
        Input batch matrix.
    Y : array-like, shape=(n_clusters, n_features)
        Matrix in which each row represents a centroid of a cluster.

    Returns
    --------
    Z : array-like, shape=(batch_size, n_clusters)
        Pairwise dissimilarity matrix.
    """

    func = lambda x_i: _tf.reduce_sum(_tf.abs(x_i-Y), axis=1)
    Z = _tf.vectorized_map(func, X)
    return Z

def minkowsky(X, Y, p):
    """
    Computes a pairwise Minkowsky distance between two matrices :math:`\mathbf{D}_{ij}=( \sum |\mathbf{x}_i - \mathbf{y}_j|^p)^{1/p}`.

    Parameters
    ----------
    X : array-like, shape=(batch_size, n_features)
        Input batch matrix.
    Y : array-like, shape=(n_clusters, n_features)
        Matrix in which each row represents a centroid of a cluster.
    p : float
        Order of the Minkowsky distance.

    Returns
    --------
    Z : array-like, shape=(batch_size, n_clusters)
        Pairwise dissimilarity matrix.
    """
  
    if p>1:
        func = lambda x_i: _tf.reduce_sum(
                _tf.abs(x_i-Y)**p, axis=1
                )**(1/p)
    else:
        func = lambda x_i: _tf.reduce_sum(
                _tf.abs(x_i-Y),
                axis=1
                )
    Z = _tf.vectorized_map(func, X)
    return Z

def chebyshev(X, Y):
    """
    Computes a pairwise Chevyshev distance between two matrices :math:`\max{(|\mathbf{x}_i-\mathbf{y}_j|)}`.

    Parameters
    ----------
    X : array-like, shape=(batch_size, n_features)
        Input batch matrix.
    Y : array-like, shape=(n_clusters, n_features)
        Matrix in which each row represents a centroid of a cluster.

    Returns
    --------
    Z : array-like, shape=(batch_size, n_clusters)
        Pairwise dissimilarity matrix.
    """

    func = lambda x_i: _tf.reduce_max(
            _tf.abs(x_i-Y), 
            axis=1
            )
    Z = _tf.vectorized_map(func, X)
    return Z


def mahalanobis(X, Y, cov):
    """
    Computes a pairwise Mahalanobis distance :math:`\mathbf{D}_{ij}=(\mathbf{x}_i-\mathbf{y}_j)^T \Sigma_j (\mathbf{x}_i-\mathbf{y}_j)`.

    Parameters
    ----------
    X : array-like, shape=(batch_size, n_features)
        Input batch matrix.
    Y : array-like, shape=(n_features, n_features)
        Matrix in which each row represents a centroid of a cluster.
    cov: array-like, shape=(n_clusters, n_features, n_features)
        3D Tensor with the inverse covariance matrices of all the clusters.

    Returns
    --------
    Z : array-like, shape=(batch_size, n_clusters)
        Pairwise dissimilarity matrix.
    """

    def func(x_i):
        diff = _tf.expand_dims(x_i-Y, axis=-1)
        return _tf.squeeze(
                _tf.reduce_sum(
                    _tf.matmul(cov, diff)*diff,
                    axis=1
                    )
                )
    Z = _tf.vectorized_map(func, X)
    return Z

def toroidal_euclidean(
        X, Y,
        interval=_tf.constant(
            (2.0, 2.0)
            )
        ):
    """
    Euclidean dissimilarity that considers circular boundaries.

    Parameters
    ----------
    X : array-like, shape=(batch_size, n_features)
        Input batch matrix.
    Y : array-like, shape=(n_features, n_features)
        Matrix in which each row represents a centroid of a cluster.
    interval : array-like, default=tf.constant((2.0, 2.0))
        Array representing the range on each axis.

    Returns
    -------
    Z : array-like, shape=(batch_size, n_clusters)
        Pairwise dissimilarity matrix.
    """
    def _toroidal_dis(
            x_i, Y, 
            interval=_tf.constant((2.0, 2.0))):
        d = _tf.reduce_sum(
                (x_i - Y) ** 2,
                axis=1
                )
        for val in _itertools.product(
                [0.0, 1.0, -1.0], 
                repeat=2
                ):
            delta = _tf.constant(val) * interval
            d = _tf.minimum(
                    _tf.reduce_sum(
                        (x_i - Y + delta) ** 2,
                        axis=1
                        ),
                    d
                    )
        return d
   
    func = lambda x_i: _toroidal_dis(
            x_i, Y, interval
            )
    return _tf.vectorized_map(func, X)

def kullback_leibler(
        loggit_P, loggit_Q, 
        eps=1e-3, normalization="softmax_abs"):
    """
    Kullback Leibler divergence. :math:`\sum_x P_x \log{P_x}-P_x \log{Q_x}`

    Parameters
    ----------
    loggit_P : array-like, shape=(batch_size, n_features)
        Input batch matrix of loggits.
    loggit_Q : array-like, shape=(n_features, n_features)
        Matrix in which each row represents the unsigned loggit of a cluster.
    eps: float, default=1e-3
        Hyperparameter to avoid numerical issues.
    normalization: {str, function}, default="softmax_abs"
        Specifies which normalization function is used to transform the data into
        probabilities. You can specify a custom functon `f(X, eps)` with the arguments 
        `X` and `eps`, or use a predefined function {"softmax_abs", "softmax_relu", "squared_sum", "abs_sum", "relu_sum", "identity"}

    Returns
    -------
    Z : array-like, shape=(batch_size, n_clusters)
        Pairwise dissimilarity matrix.
    """

    if normalization=="softmax_abs":
        norm = _normalizers.softmax_abs
    elif normalization=="softmax_relu":
        norm = _normalizers.softmax_relu
    elif normalization=="squared_sum":
        norm = _normalizers.squared_sum
    elif normalization=="abs_sum":
        norm = _normalizers.abs_sum
    elif normalization=="relu_sum":
        norm = _normalizers.relu_sum
    elif normalization=="identity":
        norm = _normalizers.identity
    else: 
        norm = normalization

    P = norm(loggit_P, eps)
    Q = norm(loggit_Q, eps)
    term1 = _tf.reduce_sum(
            P * _tf.math.log(P),
            axis=1
            )
    def func(p_i):
        p_i = _tf.reshape(
            p_i,
            (1, -1)
            )
        return _tf.squeeze(
            _tf.matmul(
                p_i, _tf.math.log(Q),
                transpose_b=True
                )
            )
    
    Z = _tf.vectorized_map(func, P) 
    return -Z + _tf.reshape(
            term1, (-1, 1)
            )

