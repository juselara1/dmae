"""
Implementation of: Dissimilarity Mixture Autoencoder (DMAE) for Deep Clustering.

**This package contains the tensorflow implementation of different loss functions for each dissimilarity that are required in DMAE.**

Author: Juan Sebastián Lara Ramírez <julara@unal.edu.co> <https://github.com/larajuse>
"""

import tensorflow as tf

def euclidean_loss(X, mu_tilde):
    """
    Computes the Euclidean loss.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        mu_tilde: array-like, shape=(batch_size, n_features)
            Matrix in which each row represents the assigned mean vector.
    Returns:
        loss: array-like, shape=(batch_size, )
            Computed loss for each sample.
    """
    
    return tf.sqrt(tf.reduce_sum((X-mu_tilde)**2, axis=1))

def cosine_loss(X, mu_tilde):
    """
    Computes the Cosine loss.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        mu_tilde: array-like, shape=(batch_size, n_features)
            Matrix in which each row represents the assigned mean vector.
    Returns:
        loss: array-like, shape=(batch_size, )
            Computed loss for each sample.
    """
    
    X_norm = tf.nn.l2_normalize(X, axis=1)
    mu_tilde_norm = tf.nn.l2_normalize(mu_tilde, axis=1)
    return 1-tf.reduce_sum(X_norm*mu_tilde_norm, axis=1)

def correlation_loss(X, mu_tilde):
    """
    Computes the Correlation loss.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        mu_tilde: array-like, shape=(batch_size, n_features)
            Matrix in which each row represents the assigned mean vector.
    Returns:
        loss: array-like, shape=(batch_size, )
            Computed loss for each sample.
    """
    
    centered_X = X-tf.reshape(tf.reduce_mean(X, axis=0),(1, -1))
    centered_mu_tilde = mu_tilde-tf.reshape(tf.reduce_mean(mu_tilde, axis=1), (-1,1))
    return cosine_loss(centered_X, centered_mu_tilde)

def manhattan_loss(X, mu_tilde):
    """
    Computes the Manhattan loss.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        mu_tilde: array-like, shape=(batch_size, n_features)
            Matrix in which each row represents the assigned mean vector.
    Returns:
        loss: array-like, shape=(batch_size, )
            Computed loss for each sample.
    """
    
    return tf.reduce_sum(tf.abs(X-mu_tilde), axis=1)

def minkowsky_loss(X, mu_tilde, p):
    """
    Computes the Manhattan loss.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        mu_tilde: array-like, shape=(batch_size, n_features)
            Matrix in which each row represents the assigned mean vector.
        p: float
            Order of the Minkowski distance.
    Returns:
        loss: array-like, shape=(batch_size, )
            Computed loss for each sample.
    """
    
    return tf.reduce_sum(tf.abs(X-mu_tilde)**p, axis=1)**(1/p)

def chebyshev_loss(X, mu_tilde):
    """
    Computes the Chebyshev loss.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        mu_tilde: array-like, shape=(batch_size, n_features)
            Matrix in which each row represents the assigned mean vector.
    Returns:
        loss: array-like, shape=(batch_size, )
            Computed loss for each sample.
    """
    
    return tf.reduce_max(tf.abs(X-mu_tilde), axis=1)

def mahalanobis_loss(X, mu_tilde, Cov_tilde):
    """
    Computes the Mahalanobis loss.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        mu_tilde: array-like, shape=(batch_size, n_features)
            Matrix in which each row represents the assigned mean vector.
        Cov_tilde: array-like, shape=(batch_size, n_features, n_features)
            Tensor with the assigned covariances.
    Returns:
        loss: array-like, shape=(batch_size, )
            Computed loss for each sample.
    """
    
    diff = tf.expand_dims(X-mu_tilde, axis=1)
    return tf.squeeze(tf.matmul(tf.matmul(diff, Cov_tilde), tf.transpose(diff, perm = [0, 2, 1])))

def mahalanobis_loss_decomp(X, mu_tilde, Cov_tilde):
    """
    Computes the Mahalanobis loss using the decomposition of the covariance matrices.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        mu_tilde: array-like, shape=(batch_size, n_features)
            Matrix in which each row represents the assigned mean vector.
        Cov_tilde: array-like, shape=(batch_size, n_features, n_features)
            Tensor with the assigned decomposition.
    Returns:
        loss: array-like, shape=(batch_size, )
            Computed loss for each sample.
    """
    
    cov = tf.matmul(Cov_tilde, tf.transpose(Cov_tilde, [0, 2, 1]))
    diff = tf.expand_dims(X-mu_tilde, axis=1)
    return tf.squeeze(tf.matmul(tf.matmul(diff, cov), tf.transpose(diff, perm = [0, 2, 1])))

class losses():
    def __init__(self):
        self.euclidean_loss = euclidean_loss
        self.cosine_loss = cosine_loss
        self.manhattan_loss =manhattan_loss
        self.minkowsky_loss = minkowsky_loss
        self.correlation_loss = correlation_loss
        self.mahalanobis_loss = mahalanobis_loss
        self.chebyshev_loss = chebyshev_loss
        self.mahalanobis_loss_decomp = mahalanobis_loss_decomp