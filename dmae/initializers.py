# -*- coding: utf-8 -*-
"""
The :mod:`dmae.initializers` module implements some initializers for DMAE.
"""
# Author: Juan S. Lara <julara@unal.edu.co>
# License: MIT

import numpy as _np
import tensorflow as _tf
from tensorflow.keras.initializers import Initializer as _Initializer
from dmae import dissimilarities as _dissimilarities

class InitPlusPlus(_Initializer):
    """
    A tf.keras initializer based on K-Means++ that allows dissimilarities.
    
    Parameters
    ----------
    X: array-like, shape=(n_samples, n_features)
        Input data.
    n_clusters: int
        Number of clusters.
    dissimilarity: function, default: :mod:`dmae.dissimilarities.euclidean`
        A tensorflow function that computes a paiwise dissimilarity function between a batch
        of points and the cluster's parameters.
    iters: int, default: 100
        Number of interations to run the K-means++ initialization.
    """
    
    def __init__(self, X, n_clusters, dissimilarity=_dissimilarities.euclidean, iters=100):
        self.__X = X
        self.__n_clusters = n_clusters
        self.__dissimilarity = dissimilarity
        self.__iters = iters
    
    def __call__(self, shape, dtype):
        """
        Estimates `n_clusters` using K-means++

        Parameters
        ----------
        shape : tuple
            Expected parameter shape.
        dtype : str
            Parameter's type.

        Returns
        -------
        init_vals : array-like, shape=(n_clusters, n_features)
            Matrix with the initial weights.
        """

        idx = _np.arange(self.__X.shape[0])
        _np.random.shuffle(idx)
        selected = idx[:self.__n_clusters]
        init_vals = self.__X[idx[:self.__n_clusters]]

        for i in range(self.__iters):
            clus_sim = self.__dissimilarity(
                    init_vals, 
                    init_vals
                    ).numpy()

            _np.fill_diagonal(
                    clus_sim, 
                    _np.inf
                    )

            candidate = self.__X[
                    _np.random.randint(
                        self.__X.shape[0]
                        )
                    ].reshape(1, -1)
            candidate_sims = self.__dissimilarity(
                    candidate, 
                    init_vals
                    ).numpy().flatten()

            closest_sim = candidate_sims.min()
            closest = candidate_sims.argmin()

            if closest_sim>clus_sim.min():
                replace_candidates_idx = _np.array(
                        _np.unravel_index(
                            clus_sim.argmin(),
                            clus_sim.shape
                            )
                        )
                replace_candidates = init_vals[replace_candidates_idx, :]

                closest_sim = self.__dissimilarity(
                        candidate, 
                        replace_candidates
                        ).numpy().flatten()

                replace = _np.argmin(closest_sim)
                init_vals[replace_candidates_idx[replace]] = candidate

            else:
                candidate_sims[candidate_sims.argmin()] = _np.inf
                second_closest = candidate_sims.argmin()
                if candidate_sims[second_closest] > clus_sim[closest].min():
                    init_vals[closest] = candidate

        return _tf.cast(init_vals, dtype)

class InitKMeans(_Initializer):
    """
    A tf.keras initializer to assign the clusters from a sklearn's KMeans model.
    
    Parameters
    ----------
    kmeans_model: :mod:`sklearn.cluster.KMeans`
        Pretrained KMeans model to initialize DMAE.
    """

    def __init__(self, kmeans_model):
        self.__kmeans = kmeans_model
        
    def __call__(self, shape, dtype):
        """
        Converts KMeans centroids into tensors.

        Parameters
        ----------
        shape : tuple
            Expected parameter shape.
        dtype : str
            Parameter's type.

        Returns
        -------
        init_vals : array-like, shape=(n_clusters, n_features)
            Matrix with the initial weights.
        """

        return _tf.cast(
                self.__kmeans.cluster_centers_,
                dtype
                )
    
class InitIdentityCov(_Initializer):
    """
    A tf.keras initializer to assign identity matrices to the covariance parameters. 
    
    Parameters
    ----------
    X: array-like, shape=(n_samples, n_features)
        Input data.
    n_clusters: int
        Number of clusters.
    """
    
    def __init__(self, X, n_clusters):
        self.__X = X
        self.__n_clusters = n_clusters
    
    def __call__(self, shape, dtype):
        """
        Generates identity matrices for the given shape and type.

        Parameters
        ----------
        shape : tuple
            Expected parameter shape.
        dtype : str
            Parameter's type.

        Returns
        -------
        init_vals : array-like, shape=(n_clusters, n_features)
            Matrix with the initial weights.
        """

        return _tf.eye(self.__X.shape[1], batch_shape=[self.__n_clusters])
        
class InitKMeansCov(_Initializer):
    """
    A tf.keras initializer to compute covariance matrices from K-means.
    
    Parameters
    ----------
    kmeans_model: :mod:`sklearn.cluster.KMeans`
        Pretrained KMeans model to initialize DMAE.
    X: array-like, shape=(n_samples, n_features)
        Input data.
    n_clusters: int
        Number of clusters.
    """

    def __init__(self, kmeans_model, X, n_clusters):
        self.__kmeans_model = kmeans_model
        self.__X = X
        self.__n_clusters = n_clusters
    
    def __call__(self, shape, dtype):
        """
        Computes covariance matrices from the KMeans predictions.

        Parameters
        ----------
        shape : tuple
            Expected parameter shape.
        dtype : str
            Parameter's type.

        Returns
        -------
        init_vals : array-like, shape=(n_clusters, n_features)
            Matrix with the initial weights.
        """

        res = []
        preds = self.__kmeans_model.predict(self.__X)
        for i in range(self.__n_clusters):
            clus_points = self.__X[preds==i]
            res.append(
                    _np.expand_dims(
                        _np.linalg.cholesky(
                            _np.linalg.inv(
                                _np.cov(clus_points.T)
                                )
                            ),
                        axis=0
                        )
                    )

        return _tf.cast(
                _np.concatenate(res, axis=0), 
                dtype
                )
