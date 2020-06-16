"""
Implementation of: Dissimilarity Mixture Autoencoder (DMAE) for Deep Clustering.

**This package contains the tensorflow implementation of different pairwise dissimilarity functions that are required in DMAE.**

Author: Juan Sebastián Lara Ramírez <julara@unal.edu.co> <https://github.com/larajuse>
"""

import tensorflow as tf

@tf.function
def euclidean(X, Y, batch_size=32):
    """
    Computes a pairwise Euclidean distance between two matrices: ||x-y||^2.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
            Matrix in which each row represents the mean vector of each cluster.
        batch_size: int, default=32
            Batch size that is used to compute the paiwise dissimilarities.
    Returns:
        D: array-like, shape=(batch_size, n_clusters)
            Matrix of paiwise dissimilarities between the batch and the cluster's parameters.
    """
    
    Z = []
    for i in range(batch_size):
        Z.append(tf.reshape(tf.sqrt(tf.reduce_sum((X[i]-Y)**2, axis=1)), (1, -1)))
    return tf.concat(Z, axis=0)

@tf.function
def cosine(X, Y, batch_size=32):
    """
    Computes a pairwise Cosine distance between two matrices: (x·y)/(||x||·||y||).
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
            Matrix in which each row represents the mean vector of each cluster.
        batch_size: int, default=32
            Batch size that is used to compute the paiwise dissimilarities.
    Returns:
        D: array-like, shape=(batch_size, n_clusters)
            Matrix of paiwise dissimilarities between the batch and the cluster's parameters.
    """
    
    norm_X = tf.nn.l2_normalize(X, axis=1)
    norm_Y = tf.nn.l2_normalize(Y, axis=1)
    D = 1-tf.matmul(norm_X, norm_Y, transpose_b=True)
    return D

@tf.function
def correlation(X, Y, batch_size=32):
    """
    Computes a pairwise correlation between two matrices: ((x-mu_x)·(y-mu_y))/(||x-mu_x||·||y-mu_y||).
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
            Matrix in which each row represents the mean vector of each cluster.
        batch_size: int, default=32
            Batch size that is used to compute the paiwise dissimilarities.
    Returns:
        D: array-like, shape=(batch_size, n_clusters)
            Matrix of paiwise dissimilarities between the batch and the cluster's parameters.
    """
    
    centered_X = X-tf.reshape(tf.reduce_mean(X, axis=1),(-1,1))
    centered_Y = Y-tf.reshape(tf.reduce_mean(Y, axis=1), (-1,1))
    norm_X = tf.nn.l2_normalize(centered_X, axis=1)
    norm_Y = tf.nn.l2_normalize(centered_Y, axis=1)
    D = 1-tf.matmul(norm_X, norm_Y, transpose_b=True)
    return D

@tf.function
def manhattan(X, Y, batch_size=32):
    """
    Computes a pairwise manhattan distance between two matrices: sum(|x_i|-|y_j|).
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
            Matrix in which each row represents the mean vector of each cluster.
        batch_size: int, default=32
            Batch size that is used to compute the paiwise dissimilarities.
    Returns:
        D: array-like, shape=(batch_size, n_clusters)
            Matrix of paiwise dissimilarities between the batch and the cluster's parameters.
    """
    
    Z = []
    for i in range(batch_size):
        Z.append(tf.reshape(tf.reduce_sum(tf.abs(X[i]-Y), axis=1), (1, -1)))
    return tf.concat(Z, axis=0)

@tf.function
def minkowsky(X, Y, p, batch_size=32):
    """
    Computes a pairwise Minkowski distance between two matrices: sum(|x_i-y_j|^p)^(1/p).
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
            Matrix in which each row represents the mean vector of each cluster.
        p: float
            Order of the Minkowski distance.
        batch_size: int, default=32
            Batch size that is used to compute the paiwise dissimilarities.
    Returns:
        D: array-like, shape=(batch_size, n_clusters)
            Matrix of paiwise dissimilarities between the batch and the cluster's parameters.
    """
    
    Z = []
    for i in range(batch_size):
        if p>1:
            Z.append(tf.reshape(tf.reduce_sum(tf.abs(X[i]-Y)**p, axis=1), (1, -1))**(1/p))
        else:
            Z.append(tf.reshape(tf.reduce_sum(tf.abs(X[i]-Y)**p, axis=1), (1, -1)))
    return tf.concat(Z, axis=0)

@tf.function
def chebyshev(X, Y, batch_size=32):
    """
    Computes a pairwise Chevyshev distance between two matrices: max(|x_i-y_j|).
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
            Matrix in which each row represents the mean vector of each cluster.
        batch_size: int, default=32
            Batch size that is used to compute the paiwise dissimilarities.
    Returns:
        D: array-like, shape=(batch_size, n_clusters)
            Matrix of paiwise dissimilarities between the batch and the cluster's parameters.
    """
    
    Z = []
    for i in range(batch_size):
        Z.append(tf.reshape(tf.reduce_max(tf.abs(X[i]-Y), axis=1), (1, -1)))
    return tf.concat(Z, axis=0)

@tf.function
def mahalanobis(X, Y, cov, batch_size=32):
    """
    Computes a pairwise Mahalanobis distance between two matrices: (x-y)^T Cov (x-y).
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
            Matrix in which each row represents the mean vector of each cluster.
        cov: array-like, shape=(n_clusters, n_features, n_features)
            Tensor with all the covariance matrices.
        batch_size: int, default=32
            Batch size that is used to compute the paiwise dissimilarities.
    Returns:
        D: array-like, shape=(batch_size, n_clusters)
            Matrix of paiwise dissimilarities between the batch and the cluster's parameters.
    """
    
    Z = []
    for i in range(batch_size):
        diff = tf.expand_dims(X[i]-Y, axis=-1)
        Z.append(tf.reshape(tf.reduce_sum(tf.matmul(cov, diff)*diff, axis=1), (1, -1)))
    return tf.concat(Z, axis=0)

class dissimilarities():
    def __init__(self):
        self.euclidean = euclidean
        self.cosine = cosine
        self.correlation = correlation
        self.manhattan = manhattan
        self.minkowsky = minkowsky
        self.chebyshev = chebyshev
        self.mahalanobis = mahalanobis