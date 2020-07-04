"""
Implementation of: Dissimilarity Mixture Autoencoder (DMAE) for Deep Clustering.

**This package contains the tensorflow implementation of different pairwise dissimilarity functions that are required in DMAE.**

Author: Juan Sebastián Lara Ramírez <julara@unal.edu.co> <https://github.com/larajuse>
"""

import tensorflow as tf

def euclidean(X, Y):
    """
    Computes a pairwise Euclidean distance between two matrices: D_ij=||x_i-y_j||^2.
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
            Matrix in which each row represents the mean vector of each cluster.
    Returns:
        D: array-like, shape=(batch_size, n_clusters)
            Matrix of paiwise dissimilarities between the batch and the cluster's parameters.
    """
    func = lambda x_i: tf.sqrt(tf.reduce_sum((x_i-Y)**2, axis=1))
    Z = tf.vectorized_map(func, X)
    return Z

def cosine(X, Y):
    """
    Computes a pairwise Cosine distance between two matrices: D_ij=(x_i·y_j)/(||x_i||·||y_j||).
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
            Matrix in which each row represents the mean vector of each cluster.
    Returns:
        D: array-like, shape=(batch_size, n_clusters)
            Matrix of paiwise dissimilarities between the batch and the cluster's parameters.
    """
    
    norm_X = tf.nn.l2_normalize(X, axis=1)
    norm_Y = tf.nn.l2_normalize(Y, axis=1)
    D = 1-tf.matmul(norm_X, norm_Y, transpose_b=True)
    return D

def correlation(X, Y):
    """
    Computes a pairwise correlation between two matrices: D_ij=(x_i-mu_x_i)·(y_j-mu_y_j)/(||x_i-mu_x_i||·||y_j-mu_y_j||).
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
            Matrix in which each row represents the mean vector of each cluster.
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

def manhattan(X, Y):
    """
    Computes a pairwise manhattan distance between two matrices: D_ij=sum(|x_i|-|y_j|).
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
            Matrix in which each row represents the mean vector of each cluster.
    Returns:
        D: array-like, shape=(batch_size, n_clusters)
            Matrix of paiwise dissimilarities between the batch and the cluster's parameters.
    """
    
    func = lambda x_i: tf.reduce_sum(tf.abs(x_i-Y), axis=1)
    Z = tf.vectorized_map(func, X)
    return Z

def minkowsky(X, Y, p):
    """
    Computes a pairwise Minkowski distance between two matrices: D_ij=sum(|x_i-y_j|^p)^(1/p).
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
            Matrix in which each row represents the mean vector of each cluster.
        p: float
            Order of the Minkowski distance.
    Returns:
        D: array-like, shape=(batch_size, n_clusters)
            Matrix of paiwise dissimilarities between the batch and the cluster's parameters.
    """
    if p>1:
        func = lambda x_i: tf.reduce_sum(tf.abs(x_i-Y)**p, axis=1)**(1/p)
    else:
        func = lambda x_i: tf.reduce_sum(tf.abs(x_i-Y), axis=1)
    Z = tf.vectorized_map(func, X)
    return Z

def chebyshev(X, Y):
    """
    Computes a pairwise Chevyshev distance between two matrices: max(|x_i-y_j|).
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
            Matrix in which each row represents the mean vector of each cluster.
    Returns:
        D: array-like, shape=(batch_size, n_clusters)
            Matrix of paiwise dissimilarities between the batch and the cluster's parameters.
    """
    
    func = lambda x_i: tf.reduce_max(tf.abs(x_i-Y), axis=1)
    Z = tf.vectorized_map(func, X)
    return Z

@tf.function
def mahalanobis(X, Y, cov):
    """
    Computes a pairwise Mahalanobis distance between two matrices: (x_i-y_j)^T Cov_j (x_i-y_j).
    Arguments:
        X: array-like, shape=(batch_size, n_features)
            Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
            Matrix in which each row represents the mean vector of each cluster.
        cov: array-like, shape=(n_clusters, n_features, n_features)
            Tensor with all the covariance matrices.
    Returns:
        D: array-like, shape=(batch_size, n_clusters)
            Matrix of paiwise dissimilarities between the batch and the cluster's parameters.
    """
    
    def func(x_i):
        diff = tf.expand_dims(x_i-Y, axis=-1)
        return tf.squeeze(tf.reduce_sum(tf.matmul(cov, diff)*diff, axis=1))
    Z = tf.vectorized_map(func, X)
    return Z

class dissimilarities():
    def __init__(self):
        self.euclidean = euclidean
        self.cosine = cosine
        self.correlation = correlation
        self.manhattan = manhattan
        self.minkowsky = minkowsky
        self.chebyshev = chebyshev
        self.mahalanobis = mahalanobis