# -*- coding: utf-8 -*-
"""
Implementation of: Dissimilarity Mixture Autoencoder (DMAE) for Deep Clustering.

**This package contains different tf.keras layers that are required to build the DMAE model.**

Author: Juan Sebastián Lara Ramírez <julara@unal.edu.co> <https://github.com/larajuse>
"""

import tensorflow as tf
from DMAE import Dissimilarities


class DissimilarityMixtureAutoencoder(tf.keras.layers.Layer):
    """
    A tf.keras layer that contains the Dissimilarity Mixture Autoencoder.

    Arguments:
        alpha: float
            Softmax inverse temperature (sparsity control).
        n_clusters: int
            Number of clusters.
        dissimilarity: function, default: DMAE.Dissimilarities.euclidean
            A tensorflow function that computes a paiwise dissimilarity function between a batch
            of points and the cluster's parameters (means).
        trainable: dict, default: {"centers": True, "mixers":False}
            A dictionary of bool variables to specify which parameters must be trained.
        initializers: dict, default: {"centers": tf.keras.initializers.RandomUniform(-1,1), "mixers": tf.keras.initializers.Constant(1.0)}
            A dictionary with tf.keras.initializers to initialize each parameter.
    """

    def __init__(
        self,
        alpha,
        n_clusters,
        dissimilarity=Dissimilarities.euclidean,
        trainable={"centers": True, "mixers": False},
        initializers={
            "centers": tf.keras.initializers.RandomUniform(-1, 1),
            "mixers": tf.keras.initializers.Constant(1.0),
        },
        regularizers={"centers": None, "mixers": None},
        **kwargs
    ):
        self.__alpha = tf.constant(alpha, dtype=tf.float32)
        self.__n_clusters = n_clusters
        self.__dissimilarity = dissimilarity
        self.__trainable = trainable
        self.__initializers = initializers
        self.__regularizers = regularizers
        super(DissimilarityMixtureAutoencoder, self).__init__(**kwargs)

    def call(self, x):
        """Forward pass in DMAE"""

        D = self.__dissimilarity(x, self.centers)  # Compute pairwise dissimilarities
        assigns = tf.nn.softmax(
            -self.__alpha * D + tf.math.log(tf.math.abs(self.mixers))
        )  # Soft-assignments
        mu_tilde = tf.matmul(
            assigns, self.centers
        )  # Reconstruction of the assigned mean.
        pi_tilde = tf.reduce_sum(assigns * self.mixers, axis=1)
        return mu_tilde, pi_tilde

    def build(self, input_shape):
        """Defines and initializes each parameter"""

        self.centers = self.add_weight(
            name="centers",
            initializer=self.__initializers["centers"],
            shape=(self.__n_clusters, input_shape[1]),
            trainable=self.__trainable["centers"],
            regularizer=self.__regularizers["centers"],
        )
        self.mixers = self.add_weight(
            name="mixers",
            initializer=self.__initializers["mixers"],
            shape=(1, self.__n_clusters),
            trainable=self.__trainable["mixers"],
            regularizer=self.__regularizers["mixers"],
        )
        super(DissimilarityMixtureAutoencoder, self).build(input_shape)


class DissimilarityMixtureEncoder(tf.keras.layers.Layer):
    """
    A tf.keras layer that contains the Dissimilarity Mixture Encoder.

    Arguments:
        alpha: float
            Softmax inverse temperature (sparsity control).
        n_clusters: int
            Number of clusters.
        dissimilarity: function, default: DMAE.Dissimilarities.euclidean
            A tensorflow function that computes a paiwise dissimilarity function between a batch
            of points and the cluster's parameters (means).
        trainable: dict, default: {"centers": True, "mixers":False}
            A dictionary of bool variables to specify which parameters must be trained.
        initializers: dict, default: {"centers": tf.keras.initializers.RandomUniform(-1,1), "mixers": tf.keras.initializers.Constant(1.0)}
            A dictionary with tf.keras.initializers to initialize each parameter.
    """

    def __init__(
        self,
        alpha,
        n_clusters,
        dissimilarity=Dissimilarities.euclidean,
        trainable={"centers": True, "mixers": False},
        initializers={
            "centers": tf.keras.initializers.RandomUniform(-1, 1),
            "mixers": tf.keras.initializers.Constant(1.0),
        },
        **kwargs
    ):
        self.__alpha = tf.constant(alpha, dtype=tf.float32)
        self.__n_clusters = n_clusters
        self.__dissimilarity = dissimilarity
        self.__trainable = trainable
        self.__initializers = initializers
        super(DissimilarityMixtureEncoder, self).__init__(**kwargs)

    def call(self, x):
        """Forward pass in DM-Encoder"""

        D = self.__dissimilarity(x, self.centers)  # Compute pairwise dissimilarities
        assigns = tf.nn.softmax(
            -self.__alpha * D + tf.math.log(tf.nn.relu(self.mixers))
        )  # Soft-assignments
        return assigns

    def build(self, input_shape):
        """Defines and initializes each parameter"""

        self.centers = self.add_weight(
            name="centers",
            initializer=self.__initializers["centers"],
            shape=(self.__n_clusters, input_shape[1]),
            trainable=self.__trainable["centers"],
        )
        self.mixers = self.add_weight(
            name="mixers",
            initializer=self.__initializers["mixers"],
            shape=(1, self.__n_clusters),
            trainable=self.__trainable["mixers"],
        )
        super(DissimilarityMixtureEncoder, self).build(input_shape)


class DissimilarityMixtureAutoencoderCov(tf.keras.layers.Layer):
    """
    A tf.keras layer that contains the Dissimilarity Mixture Autoencoder with Covariance Matrices.

    Arguments:
        alpha: float
            Softmax inverse temperature (sparsity control).
        n_clusters: int
            Number of clusters.
        dissimilarity: function, default: DMAE.Dissimilarities.mahalanobis
            A tensorflow function that computes a paiwise dissimilarity function between a batch
            of points and the cluster's parameters (means and covariances).
        trainable: dict, default: {"centers": True, "cov": True, "mixers":False}
            A dictionary of bool variables to specify which parameters must be trained.
        initializers: dict, default: {"centers": tf.keras.initializers.RandomUniform(-1,1), "cov": tf.initializers.RandomUniform(-1,1), "mixers": tf.keras.initializers.Constant(1.0)}
            A dictionary with tf.keras.initializers to initialize each parameter.
        grad_modifier: float, default=1
            A value that scales the gradients for the covariances matrices.
    """

    def __init__(
        self,
        alpha,
        n_clusters,
        dissimilarity=Dissimilarities.mahalanobis,
        trainable={"centers": True, "cov": True, "mixers": True},
        initializers={
            "centers": tf.keras.initializers.RandomUniform(-1, 1),
            "cov": tf.keras.initializers.RandomUniform(-1, 1),
            "mixers": tf.keras.initializers.Constant(1.0),
        },
        grad_modifier=1,
        regularizers={"centers": None, "cov": None, "mixers": None},
        **kwargs
    ):
        self.__alpha = tf.constant(alpha, dtype=tf.float32)
        self.__n_clusters = n_clusters
        self.__dissimilarity = dissimilarity
        self.__trainable = trainable
        self.__initializers = initializers
        self.__grad_modifier = grad_modifier
        self.__regularizers = regularizers
        super(DissimilarityMixtureAutoencoderCov, self).__init__(**kwargs)

    @tf.custom_gradient
    def __psd_matrix(self, X):
        """Computes a positive semidefinite matrix and modifies its gradients"""

        res = tf.matmul(X, tf.transpose(X, [0, 2, 1]))

        def grad(dy):
            return dy * self.__grad_modifier

        return res, grad

    def call(self, x):
        """Forward pass in DMAE"""

        cov = self.__psd_matrix(self.cov)  # Defining a PSD matrix
        D = self.__dissimilarity(
            x, self.centers, cov
        )  # Compute pairwise dissimilarities
        bias = tf.math.log(
            tf.nn.relu(self.mixers)
        )  # Computes the bias using the mixers
        assigns = tf.nn.softmax(-self.__alpha * D + bias)  # Soft-assignments
        mu_hat = tf.matmul(
            assigns, self.centers
        )  # Reconstruction of the assigned mean.
        Cov_hat = tf.tensordot(
            assigns, cov, axes=[[1], [0]]
        )  # Reconstruction of the assigned covariance.
        pi_tilde = tf.reduce_sum(assigns * self.mixers, axis=1)
        return mu_hat, Cov_hat, pi_tilde

    def build(self, input_shape):
        """Defines and initializes each parameter"""

        self.centers = self.add_weight(
            name="centers",
            initializer=self.__initializers["centers"],
            shape=(self.__n_clusters, input_shape[1]),
            trainable=self.__trainable["centers"],
            regularizer=self.__regularizers["centers"],
        )
        self.cov = self.add_weight(
            name="cov",
            initializer=self.__initializers["cov"],
            shape=(self.__n_clusters, input_shape[1], input_shape[1]),
            trainable=self.__trainable["cov"],
            regularizer=self.__regularizers["cov"],
        )
        self.mixers = self.add_weight(
            name="mixers",
            initializer=self.__initializers["mixers"],
            shape=(1, self.__n_clusters),
            trainable=self.__trainable["mixers"],
            regularizer=self.__regularizers["mixers"],
        )
        super(DissimilarityMixtureAutoencoderCov, self).build(input_shape)


class DissimilarityMixtureEncoderCov(tf.keras.layers.Layer):
    """
    A tf.keras layer that contains the Dissimilarity Mixture Encoder with Covariance Matrices.

    Arguments:
        alpha: float
            Softmax inverse temperature (sparsity control).
        n_clusters: int
            Number of clusters.
        dissimilarity: function, default: DMAE.Dissimilarities.mahalanobis
            A tensorflow function that computes a paiwise dissimilarity function between a batch
            of points and the cluster's parameters (means and covariances).
        trainable: dict, default: {"centers": True, "cov": True, "mixers":False}
            A dictionary of bool variables to specify which parameters must be trained.
        initializers: dict, default: {"centers": tf.keras.initializers.RandomUniform(-1,1), "cov": tf.initializers.RandomUniform(-1,1), "mixers": tf.keras.initializers.Constant(1.0)}
            A dictionary with tf.keras.initializers to initialize each parameter.
        grad_modifier: float, default=1,
            A value that scales the gradients for the covariances matrices.
    """

    def __init__(
        self,
        alpha,
        n_clusters,
        dissimilarity=Dissimilarities.mahalanobis,
        trainable={"centers": True, "cov": True, "mixers": True},
        initializers={
            "centers": tf.initializers.RandomUniform(-1, 1),
            "cov": tf.initializers.RandomUniform(-1, 1),
            "mixers": tf.keras.initializers.Constant(1.0),
        },
        grad_modifier=1,
        **kwargs
    ):
        self.__alpha = tf.constant(alpha, dtype=tf.float32)
        self.__n_clusters = n_clusters
        self.__dissimilarity = dissimilarity
        self.__trainable = trainable
        self.__initializers = initializers
        self.__grad_modifier = grad_modifier
        super(DissimilarityMixtureEncoderCov, self).__init__(**kwargs)

    @tf.custom_gradient
    def __psd_matrix(self, X):
        """Computes a positive semidefinite matrix and modifies its gradients"""

        res = tf.matmul(X, tf.transpose(X, [0, 2, 1]))

        def grad(dy):
            return dy * self.__grad_modifier

        return res, grad

    def call(self, x):
        """Forward pass in DM-Encoder"""

        cov = self.__psd_matrix(self.cov)  # Defining a PSD matrix
        D = self.__dissimilarity(
            x, self.centers, cov
        )  # Compute pairwise dissimilarities
        bias = tf.math.log(
            tf.nn.relu(self.mixers)
        )  # Computes the bias using the mixers
        assigns = tf.nn.softmax(-self.__alpha * D + bias)  # Soft-assignments
        return assigns

    def build(self, input_shape):
        """Defines and initializes each parameter"""

        self.centers = self.add_weight(
            name="centers",
            initializer=self.__initializers["centers"],
            shape=(self.__n_clusters, input_shape[1]),
            trainable=self.__trainable["centers"],
        )
        self.cov = self.add_weight(
            name="cov",
            initializer=self.__initializers["cov"],
            shape=(self.__n_clusters, input_shape[1], input_shape[1]),
            trainable=self.__trainable["cov"],
        )
        self.mixers = self.add_weight(
            name="mixers",
            initializer=self.__initializers["mixers"],
            shape=(1, self.__n_clusters),
            trainable=self.__trainable["mixers"],
        )
        super(DissimilarityMixtureEncoderCov, self).build(input_shape)


class layers:
    def __init__(self):
        self.DissimilarityMixtureAutoencoder = DissimilarityMixtureAutoencoder
        self.DissimilarityMixtureEncoder = DissimilarityMixtureEncoder
        self.DissimilarityMixtureAutoencoderCov = DissimilarityMixtureAutoencoderCov
        self.DissimilarityMixtureEncoderCov = DissimilarityMixtureEncoderCov
