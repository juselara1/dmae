# -*- coding: utf-8 -*-
"""
The :mod:`dmae.layers` module implements the dissimilarity mixture autoencoder (DMAE) layers as
tensorflow keras layers. 
"""

# Author: Juan S. Lara <julara@unal.edu.co>
# License: MIT

import tensorflow as _tf
from tensorflow.keras.layers import Layer as _Layer
from tensorflow.keras.initializers import RandomUniform as _RandomUniform, Constant as _Constant
from dmae import dissimilarities as _dissimilarities

class DissimilarityMixtureAutoencoder(_Layer):
    """
    A :mod:`tf.keras` layer with the Dissimilarity Mixture Autoencoder (DMAE).

    Parameters
    ----------
    alpha : float
        Softmax inverse temperature.
    n_clusters : int
        Number of clusters.
    dissimilarity : function, default = :mod:`dmae.dissimilarities.euclidean`
        A tensorflow function that computes a pairwise dissimilarity function
        between a batch of points and the cluster's parameters.
    trainable : dict, default = {"centers": True, "mixers": True}
        Specifies which parameters are trainable.
    initializers : dict, default = {"centers": :mod:`RandomUniform(-1, 1)`, "mixers": :mod:`Constant(1.0)`}
        Specifies a keras initializer (:mod:`tf.keras.initializers`) for each  parameter.
    regularizers : dict, default = {"centers": None, "mixers": None}
        Specifies a keras regularizer (:mod:`tf.keras.regularizers`) for each parameter.
    """

    def __init__(
            self, alpha, n_clusters,
            dissimilarity=_dissimilarities.euclidean,
            trainable={"centers": True, "mixers": False},
            initializers={
                "centers": _RandomUniform(-1, 1),
                "mixers": _Constant(1.0)
                },
            regularizers={"centers": None, "mixers": None},
            **kwargs
            ):

        self.__alpha = _tf.constant(alpha, dtype=_tf.float32)
        self.__n_clusters = n_clusters
        self.__dissimilarity = dissimilarity
        self.__trainable = trainable
        self.__initializers = initializers
        self.__regularizers = regularizers
        super(DissimilarityMixtureAutoencoder, self).__init__(**kwargs)

    def call(self, x):
        """
        Forward pass in DMAE.

        Parameters
        ----------
        x : array_like
            Input tensor.

        Returns
        -------
        mu_tilde : array_like
            Soft-assigned centroids.
        pi_tilde : array_like
            Soft-assigned mixing coefficients.
        """

        # computes pairwise dissimilarities
        D = self.__dissimilarity(x, self.centers)
        # computes the soft-assignements
        assigns = _tf.nn.softmax(
                -self.__alpha * D +\
                        _tf.math.log(_tf.math.abs(self.mixers))
                )
        # soft-assigned centroids
        mu_tilde = _tf.matmul(assigns, self.centers)

        # soft-assigned mixing coefficients
        pi_tilde = _tf.reduce_sum(
                assigns * self.mixers,
                axis=1
                )

        return mu_tilde, pi_tilde

    def build(self, input_shape):
        """
        Builds the tensorflow variables.

        Parameters
        ----------
        input_shape : tuple
            Input tensor shape.
        """

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

class DissimilarityMixtureEncoder(_Layer):
    """
    A tf.keras layer that implements the dissimilarity mixture encoder (DM-Encoder).
    It computes the soft assignments using a dissimilarity function from
    :mod:`dmae.dissimilarities`.

    Parameters
    ----------
    alpha : float
        Softmax inverse temperature.
    n_clusters : int
        Number of clusters.
    dissimilarity : function, default = :mod:`dmae.dissimilarities.euclidean`
        A tensorflow function that computes a pairwise dissimilarity function
        between a batch of points and the cluster's parameters.
    trainable : dict, default = {"centers": True, "mixers": True}
        Specifies which parameters are trainable.
    initializers : dict, default = {"centers": :mod:`RandomUniform(-1, 1)`, "mixers": :mod:`Constant(1.0)`}
        Specifies a keras initializer (:mod:`tf.keras.initializers`) for each  parameter.
    regularizers : dict, default = {"centers": None, "mixers": None}
        Specifies a keras regularizer (:mod:`tf.keras.regularizers`) for each parameter.
    """


    def __init__(
            self, alpha, n_clusters,
            dissimilarity=_dissimilarities.euclidean,
            trainable={"centers": True, "mixers": False},
            initializers={
                "centers": _RandomUniform(-1, 1),
                "mixers": _Constant(1.0)
                },
            regularizers={"centers": None, "mixers": None},
            **kwargs
            ):
        self.__alpha = _tf.constant(alpha, dtype=_tf.float32)
        self.__n_clusters = n_clusters
        self.__dissimilarity = dissimilarity
        self.__trainable = trainable
        self.__initializers = initializers
        self.__regularizers = regularizers
        super(DissimilarityMixtureEncoder, self).__init__(**kwargs)

    def call(self, x):
        """
        Forward pass in DM-Encoder.

        Parameters
        ----------
        x : array_like
            Input tensor.

        Returns
        -------
        S : array_like
            Soft assignments.    
        """

        # compute pairwise dissimilarities
        D = self.__dissimilarity(x, self.centers)
        # compute the soft assignments
        S = _tf.nn.softmax(
                -self.__alpha * D +\
                        _tf.math.log(_tf.nn.relu(self.mixers))
                        )
        return assigns

    def build(self, input_shape):
        """
        Builds the tensorflow variables.

        Parameters
        ----------
        input_shape : tuple
            Input tensor shape.
        """

        self.centers = self.add_weight(
                name="centers",
                initializer=self.__initializers["centers"],
                shape=(self.__n_clusters, input_shape[1]),
                trainable=self.__trainable["centers"],
                regularizer=self.__regularizers["centers"]
                )

        self.mixers = self.add_weight(
                name="mixers",
                initializer=self.__initializers["mixers"],
                shape=(1, self.__n_clusters),
                trainable=self.__trainable["mixers"],
                regularizer=self.__regularizers["mixers"]
                )

        super(DissimilarityMixtureEncoder, self).build(input_shape)

class DissimilarityMixtureAutoencoderCov(_Layer):
    """
    A :mod:`tf.keras` layer with the Dissimilarity Mixture Autoencoder (DMAE).
    This layer includes a covariance parameter for dissimilarities that allow it.

    Parameters
    ----------
    alpha : float
        Softmax inverse temperature.
    n_clusters : int
        Number of clusters.
    dissimilarity : function, default = :mod:`dmae.dissimilarities.mahalanobis`
        A tensorflow function that computes a pairwise dissimilarity function
        between a batch of points and the cluster's parameters.
    trainable : dict, default = {"centers": True, "cov": True, mixers": True}
        Specifies which parameters are trainable.
    initializers : dict, default = {"centers": :mod:`RandomUniform(-1, 1)`, "cov": :mod:`RandomUniform(-1, 1)`
    "mixers": :mod:`Constant(1.0)`}
        Specifies a keras initializer (:mod:`tf.keras.initializers`) for each  parameter.
    regularizers : dict, default = {"centers": None, "cov": None, "mixers": None}
        Specifies a keras regularizer (:mod:`tf.keras.regularizers`) for each parameter.
    """

    def __init__(
            self, alpha, n_clusters,
            dissimilarity=_dissimilarities.mahalanobis,
            trainable={"centers": True, "cov": True, "mixers": True},
            initializers={
                "centers": _RandomUniform(-1, 1),
                "cov": _RandomUniform(-1, 1),
                "mixers": _Constant(1.0),
            },
            grad_modifier=1,
            regularizers={"centers": None, "cov": None, "mixers": None},
            **kwargs
            ):

        self.__alpha =_tf.constant(alpha, dtype=_tf.float32)
        self.__n_clusters = n_clusters
        self.__dissimilarity = dissimilarity
        self.__trainable = trainable
        self.__initializers = initializers
        self.__regularizers = regularizers
        super(DissimilarityMixtureAutoencoderCov, self).__init__(**kwargs)

    def call(self, x):
        """
        Forward pass in DMAE.

        Parameters
        ----------
        x : array_like
            Input tensor.

        Returns
        -------
        mu_tilde : array_like
            Soft-assigned centroids.
        Cov_hat : array_like
            Soft-assigned covariance matrices.
        pi_tilde : array_like
            Soft-assigned mixing coefficients.
        """

        # compute PSD matrix.
        cov =_tf.matmul(self.cov,_tf.transpose(self.cov, [0, 2, 1]))
        # compute pairwise dissimilarities.
        D = self.__dissimilarity(x, self.centers, cov)
        # compute the soft assignments.
        assigns = _tf.nn.softmax(
                -self.__alpha * D +\
                        _tf.math.log(tf.nn.relu(self.mixers))
                        )
        # soft-assigned centroids
        mu_hat = _tf.matmul(
                assigns, self.centers
                )
        # soft-assigned covariance matrices
        Cov_hat = _tf.tensordot(
                assigns, cov, axes=[[1], [0]]
                ) 
        # soft-assigned mixing coefficients
        pi_tilde = _tf.reduce_sum(
                assigns * self.mixers,
                axis=1
                )

        return mu_hat, Cov_hat, pi_tilde

    def build(self, input_shape):
        """
        Builds the tensorflow variables.

        Parameters
        ----------
        input_shape : tuple
            Input tensor shape.
        """

        self.centers = self.add_weight(
                name="centers",
                initializer=self.__initializers["centers"],
                shape=(self.__n_clusters, input_shape[1]),
                trainable=self.__trainable["centers"],
                regularizer=self.__regularizers["centers"]
                )

        self.cov = self.add_weight(
                name="cov",
                initializer=self.__initializers["cov"],
                shape=(self.__n_clusters, input_shape[1], input_shape[1]),
                trainable=self.__trainable["cov"],
                regularizer=self.__regularizers["cov"]
                )

        self.mixers = self.add_weight(
                name="mixers",
                initializer=self.__initializers["mixers"],
                shape=(1, self.__n_clusters),
                trainable=self.__trainable["mixers"],
                regularizer=self.__regularizers["mixers"]
                )

        super(DissimilarityMixtureAutoencoderCov, self).build(input_shape)

class DissimilarityMixtureEncoderCov(_Layer):
    """
    A tf.keras layer that implements the dissimilarity mixture encoder (DM-Encoder).
    It computes the soft assignments using a dissimilarity function from
    :mod:`dmae.dissimilarities`. This layer includes a covariance parameter for
    dissimilarities that allow it.

    Parameters
    ----------
    alpha : float
        Softmax inverse temperature.
    n_clusters : int
        Number of clusters.
    dissimilarity : function, default = :mod:`dmae.dissimilarities.mahalanobis`
        A tensorflow function that computes a pairwise dissimilarity function
        between a batch of points and the cluster's parameters.
    trainable : dict, default = {"centers": True, "cov": True, mixers": True}
        Specifies which parameters are trainable.
    initializers : dict, default = {"centers": :mod:`RandomUniform(-1, 1)`, "cov": :mod:`RandomUniform(-1, 1)`
    "mixers": :mod:`Constant(1.0)`}
        Specifies a keras initializer (:mod:`tf.keras.initializers`) for each  parameter.
    regularizers : dict, default = {"centers": None, "cov": None, "mixers": None}
        Specifies a keras regularizer (:mod:`tf.keras.regularizers`) for each parameter.
    """

    def __init__(
            self, alpha, n_clusters,
            dissimilarity=_dissimilarities.mahalanobis,
            trainable={"centers": True, "cov": True, "mixers": True},
            initializers={
                "centers": _RandomUniform(-1, 1),
                "cov": _RandomUniform(-1, 1),
                "mixers": _Constant(1.0),
                },
            regularizers={"centers": None, "cov": None, "mixers": None},
            **kwargs
            ):

        self.__alpha = _tf.constant(alpha, dtype=_tf.float32)
        self.__n_clusters = n_clusters
        self.__dissimilarity = dissimilarity
        self.__trainable = trainable
        self.__initializers = initializers
        self.__regularizers = regularizers
        super(DissimilarityMixtureEncoderCov, self).__init__(**kwargs)

    def call(self, x):
        """
        Forward pass in DM-Encoder.

        Parameters
        ----------
        x : array_like
            Input tensor.

        Returns
        -------
        S : array_like
            Soft assignments.    
        """

        # computes PSD matrix
        cov = _tf.matmul(self.cov, _tf.transpose(self.cov, [0, 2, 1]))
        # computes pairwise dissimilarities.
        D = self.__dissimilarity(
                x, self.centers, cov
                )
        # computes the soft assignments.
        assigns = _tf.nn.softmax(
                -self.__alpha * D +\
                        _tf.math.log(_tf.nn.relu(self.mixers))
                        )
        return assigns

    def build(self, input_shape):
        """
        Builds the tensorflow variables.

        Parameters
        ----------
        input_shape : tuple
            Input tensor shape.
        """

        self.centers = self.add_weight(
                name="centers",
                initializer=self.__initializers["centers"],
                shape=(self.__n_clusters, input_shape[1]),
                trainable=self.__trainable["centers"],
                regularizer=self.__regularizers["centers"]
                )

        self.cov = self.add_weight(
                name="cov",
                initializer=self.__initializers["cov"],
                shape=(self.__n_clusters, input_shape[1], input_shape[1]),
                trainable=self.__trainable["cov"],
                regularizer=self.__regularizers["cov"]
                )

        self.mixers = self.add_weight(
                name="mixers",
                initializer=self.__initializers["mixers"],
                shape=(1, self.__n_clusters),
                trainable=self.__trainable["mixers"],
                regularizer=self.__regularizers["mixers"]
                )

        super(DissimilarityMixtureEncoderCov, self).build(input_shape)
