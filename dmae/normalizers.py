# -*- coding: utf-8 -*-
"""
The :mod:`dmae.normalizers` module implements some matrix normalization functions
in tensorflow. 
"""
# Author: Juan S. Lara <julara@unal.edu.co>
# License: MIT


import tensorflow as _tf

def identity(X, eps):
    """
    Identity normalization.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Input batch matrix.
    eps: float, default=1e-3
        Hyperparameter to avoid numerical issues.

    Returns
    -------
    Z : array-like, shape=(n_samples, n_clusters)
        Normalized matrix.
    """
    X_pos = X + eps
    return X_pos / _tf.reshape(
            _tf.reduce_sum(X_pos, axis=1), 
            (-1, 1)
            )


def softmax_abs(X, eps):
    """
    Normalization using the softmax of the absolute values.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Input batch matrix.
    eps: float, default=1e-3
        Hyperparameter to avoid numerical issues.

    Returns
    -------
    Z : array-like, shape=(n_samples, n_clusters)
        Normalized matrix.
    """
    return _tf.nn.softmax(
            _tf.abs(X) + eps 
            )

def softmax_relu(X, eps):
    """
    Normalization using the softmax of the relu values.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Input batch matrix.
    eps: float, default=1e-3
        Hyperparameter to avoid numerical issues.

    Returns
    -------
    Z : array-like, shape=(n_samples, n_clusters)
        Normalized matrix.
    """
    return _tf.nn.softmax(
            _tf.nn.relu(X) + eps 
            )

def squared_sum(X, eps):
    """
    Normalization of squared elements.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Input batch matrix.
    eps: float, default=1e-3
        Hyperparameter to avoid numerical issues.

    Returns
    -------
    Z : array-like, shape=(n_samples, n_clusters)
        Normalized matrix.
    """

    X_pos = X**2 + eps
    return X_pos / _tf.reshape(
            _tf.reduce_sum(X_pos, axis=1), 
            (-1, 1)
            )

def abs_sum(X, eps):
    """
    Normalization using absolute elements.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Input batch matrix.
    eps: float, default=1e-3
        Hyperparameter to avoid numerical issues.

    Returns
    -------
    Z : array-like, shape=(n_samples, n_clusters)
        Normalized matrix.
    """

    X_pos = _tf.abs(X) + eps
    return X_pos / _tf.reshape(
            _tf.reduce_sum(X_pos, axis=1), 
            (-1, 1)
            )

def relu_sum(X, eps):
    """
    Normalization using the relu activation function.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Input batch matrix.
    eps: float, default=1e-3
        Hyperparameter to avoid numerical issues.

    Returns
    -------
    Z : array-like, shape=(n_samples, n_clusters)
        Normalized matrix.
    """

    X_pos = _tf.nn.relu(X) + eps
    return X_pos / _tf.reshape(
            _tf.reduce_sum(X_pos, axis=1), 
            (-1, 1)
            )

