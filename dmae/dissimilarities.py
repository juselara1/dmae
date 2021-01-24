# -*- coding: utf-8 -*-
"""
The :mod:`dmae.dissimilarities` module implements several dissimilarity functions
in tensorflow. 
"""
# Author: Juan S. Lara <julara@unal.edu.co>
# License: MIT

import tensorflow as _tf

def euclidean(X, Y):
    """
    Computes a pairwise Euclidean distance between two matrices:
    :math:`\mathbf{D}_{ij}=||\mathbf{x}_i-\mathbf{y}_j||`

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
    Computes a pairwise cosine distance between two matrices: 
    :math:`\mathbf{D}_{ij}=(\mathbf{x}_i \cdot \mathbf{y}_j)/(||\mathbf{x}_i|| \cdot ||\mathbf{y}_j||)`

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
    Computes a pairwise Manhattan distance between two matrices:
    :math:`\mathbf{D}_{ij}=\sum |\mathbf{x}_i|-|\mathbf{y}_j|`

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
    Computes a pairwise Minkowsky distance between two matrices:
    :math:`\mathbf{D}_{ij}=( \sum |\mathbf{x}_i - \mathbf{y}_j|^p)^{1/p}`

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
    Computes a pairwise Chevyshev distance between two matrices:
    :math:`\max{(|\mathbf{x}_i-\mathbf{y}_j|)}`

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
    Computes a pairwise Mahalanobis distance: :math:`\mathbf{D}_{ij}=(\mathbf{x}_i-\mathbf{y}_j)^T \Sigma_j (\mathbf{x}_i-\mathbf{y}_j)`

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
