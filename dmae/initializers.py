"""
Implementation of: Dissimilarity Mixture Autoencoder (DMAE) for Deep Clustering.

**This package contains tf.keras.initializers that can be used to initialize DMAE.**

Author: Juan Sebastián Lara Ramírez <julara@unal.edu.co> <https://github.com/larajuse>
"""

import numpy as np
import tensorflow as tf
from dmae import dissimilarities
from sklearn.cluster import KMeans

class InitPlusPlus(tf.keras.initializers.Initializer):
    """
    A tf.keras initializer similar to the K-Means++ initialization (Arthur, David, and Sergei Vassilvitskii. k-means++: The advantages of careful seeding. Stanford, 2006.) that allows dissimilarities.
    
    Arguments:
        X: array-like, shape=(n_samples, n_features)
            Input data.
        n_clusters: int
            Number of clusters.
        dissimilarity: function, default: dmae.dissimilarities.euclidean
            A tensorflow function that computes a paiwise dissimilarity function between a batch
            of points and the cluster's parameters (means and covariances).
        iters: int, default: 100
            Number of interations to run the K-means++ initialization.
    """
    
    def __init__(self, X, n_clusters, dissimilarity=dissimilarities.euclidean, iters=100):
        self.__X = X
        self.__n_clusters = n_clusters
        self.__dissimilarity = dissimilarity
        self.__iters = iters
    
    def __call__(self, shape, dtype):
        idx = np.arange(self.__X.shape[0])
        np.random.shuffle(idx)
        selected = idx[:self.__n_clusters]
        init_vals = self.__X[idx[:self.__n_clusters]]

        for i in range(self.__iters):
            clus_sim = self.__dissimilarity(init_vals, init_vals).numpy()
            np.fill_diagonal(clus_sim, np.inf)

            candidate = self.__X[np.random.randint(self.__X.shape[0])].reshape(1, -1)
            candidate_sims = self.__dissimilarity(candidate, init_vals).numpy().flatten()
            closest_sim = candidate_sims.min()
            closest = candidate_sims.argmin()
            if closest_sim>clus_sim.min():
                replace_candidates_idx = np.array(np.unravel_index(clus_sim.argmin(), clus_sim.shape))
                replace_candidates = init_vals[replace_candidates_idx, :]

                closest_sim = self.__dissimilarity(candidate, replace_candidates).numpy().flatten()
                replace = np.argmin(closest_sim)
                init_vals[replace_candidates_idx[replace]] = candidate
            else:
                candidate_sims[candidate_sims.argmin()] = np.inf
                second_closest = candidate_sims.argmin()
                if candidate_sims[second_closest] > clus_sim[closest].min():
                    init_vals[closest] = candidate
        return tf.cast(init_vals, dtype)

class InitKMeans(tf.keras.initializers.Initializer):
    """
    A tf.keras initializer to assign the clusters from a sklearn's KMeans model to DMAE.
    
    Arguments:
        kmeans_model: sklearn.cluster.KMeans
            Pretrained KMeans model to initialize DMAE.
    """
    
    def __init__(self, kmeans_model):
        self.__kmeans = kmeans_model
        
    def __call__(self, shape, dtype):
        return tf.cast(self.__kmeans.cluster_centers_, dtype)
    
class InitIdentityCov(tf.keras.initializers.Initializer):
    """
    A tf.keras initializer to assign identity matrices to covariances in DMAE.
    
    Arguments:
        X: array-like, shape=(n_samples, n_features)
            Input data.
        n_clusters: int
            Number of clusters.
    """
    
    def __init__(self, X, n_clusters):
        self.__X = X
        self.__n_clusters = n_clusters
    
    def __call__(self, shape, dtype):
        return tf.eye(self.__X.shape[1], batch_shape=[self.__n_clusters])
        
class InitKMeansCov(tf.keras.initializers.Initializer):
    """
    A tf.keras initializer to compute covariance matrices from K-means assignments to initialize DMAE.
    
    Arguments:
        kmeans_model: sklearn.cluster.KMeans
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
        res = []
        preds = self.__kmeans_model.predict(self.__X)
        for i in range(self.__n_clusters):
            clus_points = self.__X[preds==i]
            res.append(np.expand_dims(np.linalg.cholesky(np.linalg.inv(np.cov(clus_points.T))), axis=0))
        return tf.cast(np.concatenate(res, axis=0), dtype)
