# -*- coding: utf-8 -*-
"""
The :mod:`dmae.losses` module implements several loss functions for each
dissimilarity in :mod:`dmae.dissimilarities`.
"""
# Author: Juan S. Lara <julara@unal.edu.co>
# License: MIT

import tensorflow as _tf

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
    loss: array-like, shape=(batch_size, )
        Computed loss for each sample.
    """

    return _tf.reduce_sum(
            _tf.reduce_max(
                _tf.abs(X-mu_tilde),
                axis=1
                ) - _tf.math.log(pi_tilde)/alpha
            )

def mahalanobis_loss(X, mu_tilde, Cov_tilde, pi_tilde, alpha):
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
