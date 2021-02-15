# -*- coding: utf-8 -*-
"""
The :mod:`dmae.losses` module implements several loss functions for each
dissimilarity in :mod:`dmae.dissimilarities`.
"""
# Author: Juan S. Lara <julara@unal.edu.co>
# License: MIT

import tensorflow as _tf
import itertools as _itertools
from dmae import normalizers as _normalizers

def euclidean_loss(X, mu_tilde, pi_tilde, alpha):
    """
    Computes the Euclidean loss.

    Parameters
    ----------
    X: array-like, shape=(batch_size, n_features)
        Input batch matrix.
    mu_tilde: array-like, shape=(batch_size, n_features)
        Matrix in which each row represents the assigned mean vector.
    pi_tilde: array-like, shape=(batch_size, )
        Vector in which each element represents the assigned mixing coefficient.
    alpha: float
        Softmax inverse temperature.

    Returns
    --------
    loss: array-like, shape=(batch_size, )
        Computed loss for each sample.
    """
    
    return _tf.reduce_sum(
            _tf.sqrt(
                _tf.reduce_sum(
                    (X-mu_tilde)**2,
                    axis=1
                    )
                ) - _tf.math.log(pi_tilde)/alpha
            )

def cosine_loss(X, mu_tilde, pi_tilde, alpha):
    """
    Computes the cosine loss.

    Parameters
    ----------
    X: array-like, shape=(batch_size, n_features)
        Input batch matrix.
    mu_tilde: array-like, shape=(batch_size, n_features)
        Matrix in which each row represents the assigned mean vector.
    pi_tilde: array-like, shape=(batch_size, )
        Vector in which each element represents the assigned mixing coefficient.
    alpha: float
        Softmax inverse temperature.

    Returns
    --------
    loss: array-like, shape=(batch_size, )
        Computed loss for each sample.
    """   

    X_norm = _tf.nn.l2_normalize(X, axis=1)
    mu_tilde_norm = _tf.nn.l2_normalize(mu_tilde, axis=1)
    return _tf.reduce_sum(
            (1-_tf.reduce_sum(
                X_norm*mu_tilde_norm,
                axis=1
                )
                ) - _tf.math.log(pi_tilde)/alpha
            )

def manhattan_loss(X, mu_tilde, pi_tilde, alpha):
    """
    Computes the Manhattan loss.

    Parameters
    ----------
    X: array-like, shape=(batch_size, n_features)
        Input batch matrix.
    mu_tilde: array-like, shape=(batch_size, n_features)
        Matrix in which each row represents the assigned mean vector.
    pi_tilde: array-like, shape=(batch_size, )
        Vector in which each element represents the assigned mixing coefficient.
    alpha: float
        Softmax inverse temperature.

    Returns
    --------
    loss: array-like, shape=(batch_size, )
        Computed loss for each sample.
    """
    return _tf.reduce_sum(
            _tf.reduce_sum(
                _tf.abs(X-mu_tilde), 
                axis=1
                ) - _tf.math.log(pi_tilde)/alpha
            )

def minkowsky_loss(X, mu_tilde, pi_tilde, alpha, p):
    """
    Computes the Minkowsky loss.

    Parameters
    ----------
    X: array-like, shape=(batch_size, n_features)
        Input batch matrix.
    mu_tilde: array-like, shape=(batch_size, n_features)
        Matrix in which each row represents the assigned mean vector.
    pi_tilde: array-like, shape=(batch_size, )
        Vector in which each element represents the assigned mixing coefficient.
    alpha: float
        Softmax inverse temperature.
    p: float
        Order of the Minkowsky distance

    Returns
    --------
    loss: array-like, shape=(batch_size, )
        Computed loss for each sample.
    """


    if p>1:
        return _tf.reduce_sum(
                _tf.reduce_sum(
                    _tf.abs(X-mu_tilde)**p,
                    axis=1
                    )**(1/p) - _tf.math.log(pi_tilde)/alpha
                )
    else:
        return _tf.reduce_sum(
                _tf.reduce_sum(
                    _tf.abs(X-mu_tilde)**p,
                    axis=1
                    ) - _tf.math.log(pi_tilde)/alpha
                )

def chebyshev_loss(X, mu_tilde, pi_tilde, alpha):
    """
    Computes the Chebyshev loss.

    Parameters
    ----------
    X: array-like, shape=(batch_size, n_features)
        Input batch matrix.
    mu_tilde: array-like, shape=(batch_size, n_features)
        Matrix in which each row represents the assigned mean vector.
    pi_tilde: array-like, shape=(batch_size, )
        Vector in which each element represents the assigned mixing coefficient.
    alpha: float
        Softmax inverse temperature.

    Returns
    --------
    loss: float
        Computed loss for each sample.
    """

    return _tf.reduce_sum(
            _tf.reduce_max(
                _tf.abs(X-mu_tilde),
                axis=1
                ) - _tf.math.log(pi_tilde)/alpha
            )

def mahalanobis_loss(
        X, mu_tilde, Cov_tilde, 
        pi_tilde, alpha
        ):
    """
    Computes the Mahalanobis loss.

    Parameters
    ----------
    X: array-like, shape=(batch_size, n_features)
        Input batch matrix.
    mu_tilde: array-like, shape=(batch_size, n_features)
        Matrix in which each row represents the assigned mean vector.
    Cov_tilde: array-like, shape=(batch_size, n_features, n_features)
        Tensor with the assigned covariances.
    pi_tilde: array-like, shape=(batch_size, )
        Vector in which each element represents the assigned mixing coefficient.
    alpha: float
        Softmax inverse temperature.

    Returns
    --------
    loss: array-like, shape=(batch_size, )
        Computed loss for each sample.
    """
   
    diff = _tf.expand_dims(X-mu_tilde, axis=1)
    return _tf.reduce_sum(
            _tf.squeeze(
                _tf.matmul(
                    _tf.matmul(
                        diff,
                        Cov_tilde
                        ),
                    _tf.transpose(
                        diff,
                        perm = [0, 2, 1]
                        )
                    )
                )-_tf.math.log(pi_tilde)/alpha
            )

def toroidal_euclidean_loss(
        X, mu_tilde, pi_tilde, 
        alpha, interval=_tf.constant((2.0, 2.0))
        ):
    """
    Loss for the toroidal euclidean dissimilarity.

    Parameters
    ----------
    X: array-like, shape=(batch_size, n_features)
        Input batch matrix.
    mu_tilde: array-like, shape=(batch_size, n_features)
        Matrix in which each row represents the assigned mean vector.
    pi_tilde: array-like, shape=(batch_size, )
        Vector in which each element represents the assigned mixing coefficient.
    alpha: float
        Softmax inverse temperature.
    interval : array-like, default=tf.constant((2.0, 2.0))
        Array representing the range on each axis.

    Returns
    --------
    loss: float
        Computed loss for each batch.

    """

    d = _tf.reduce_sum(
            (X - mu_tilde) ** 2,
            axis=1
            )
    for val in _itertools.product(
            [0.0, 1.0, -1.0],
            repeat=2
            ):
        delta = _tf.constant(val) * interval
        d = _tf.minimum(
                _tf.reduce_sum(
                    (X - mu_tilde + delta) ** 2,
                    axis=1
                    ),
                d
                )
    return _tf.reduce_sum(
            d - _tf.math.log(pi_tilde)/alpha
            )

def kullback_leibler_loss(
        loggit_P, loggit_Q_tilde,
        pi_tilde, alpha, eps=1e-3,
        normalization="softmax_abs"
        ):
    """
    Loss for the Kullback Leibler divergence.

    Parameters
    ----------
    loggit_P: array-like, shape=(batch_size, n_features)
        Input batch loggits (pre-normalization values).
    loggit_Q_tilde: array-like, shape=(batch_size, n_features)
        Cluster loggits (pre-normalization values)
    pi_tilde: array-like, shape=(batch_size, )
        Vector in which each element represents the assigned mixing coefficient.
    alpha: float
        Softmax inverse temperature.
    normalization: {str, function}, default="softmax_abs"
        Specifies which normalization function is used to transform the data into
        probabilities. You can specify a custom functon `f(X, eps)` with the arguments 
        `X` and `eps`, or use a predefined function {"softmax_abs", "softmax_relu", "squared_sum", "abs_sum", "relu_sum", "identity"}

    Returns
    --------
    loss: float
        Computed loss for each batch.

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
    Q = norm(loggit_Q_tilde, eps)

    return _tf.reduce_sum(
            P * _tf.math.log(P) -\
                    P * _tf.math.log(Q)
                    )

