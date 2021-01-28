## -*- coding: utf-8 -*-
"""
Example script of clustering painwheel data with DMAE.
"""
# Author: Juan S. Lara <julara@unal.edu.co>
# License: MIT

from dmae.layers import DissimilarityMixtureAutoencoderCov, DissimilarityMixtureEncoderCov
from dmae.initializers import InitKMeans, InitKMeansCov
from dmae import dissimilarities, losses

import numpy as np
from numpy.random import default_rng

from sklearn.cluster import KMeans

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam

from visutils import decision_region, probability_region, save_pdf

def painwheel(
        rad_std, tan_std,
        n_clusters, rate,
        n_samples,
        random_state=0):
    """

    Generates painwheel-like data, adapted from https://github.com/mattjj/svae
    ----------
    rad_std : float
        Angular standard deviation.
    tan_std : float
        Tangencial standard deviation.
    n_clusters : int
        Number of groups to generate.
    rate : float
        Magnitude of the distortion.
    n_samples : int
        Number of samples.
    random_state : int
        Random seed number.
    """
    rng = default_rng(seed=random_state)
    rads = np.linspace(
            0, 2 * np.pi, 
            n_clusters, endpoint=False
            )
    samples = rng.normal(size=(n_samples, 2)) *\
            np.array([rad_std, tan_std])
    samples[:, 0] += 1
    labels = np.repeat(np.arange(n_clusters), n_samples//n_clusters)

    angles = rads[labels] + rate * np.exp(samples[:, 0])
    cos = np.cos(angles); sin= np.sin(angles)
    rotations = np.stack([cos, -sin, sin, cos])
    rotations = rotations.T.reshape((-1, 2, 2))

    return np.random.permutation(
            np.einsum(
                "ij,ijk->ik",
                samples,
                rotations
                )
            )

if __name__ == '__main__':
    # dataset generation parameters
    rad_std = 0.3
    tan_std = 0.05
    n_clusters = 5
    rate = 0.25
    n_samples = 1000
    random_state = 0 

    # softmax inverse temperature
    alpha = 0.5

    # dmae training parameters
    batch_size = 32
    epochs = 40
    lr = 1e-4

    # generate data
    X = painwheel(
            rad_std, tan_std,
            n_clusters, rate,
            n_samples, random_state
            )

    # pretrainer definition
    pretrainer = KMeans(n_clusters).fit(X)

    # dmae definition
    inp = Input(shape=(2, ))
    theta_tilde = DissimilarityMixtureAutoencoderCov(
            alpha=alpha, n_clusters=n_clusters,
            initializers={
                "centers": InitKMeans(pretrainer),
                "cov": InitKMeansCov(pretrainer, X, n_clusters),
                "mixers": Constant(1.0)
                },
            dissimilarity=dissimilarities.mahalanobis
            )(inp)
    model = Model(
            inputs=[inp],
            outputs=theta_tilde
            )

    # loss function
    loss = losses.mahalanobis_loss(inp, *theta_tilde, alpha)
    model.add_loss(loss)

    model.compile(optimizer=Adam(lr=lr))
    
    # training
    model.fit(
            X, epochs=epochs,
            batch_size=batch_size
            )

    # auxiliar model to obtain the assignments
    inp = Input(shape=(2, ))
    assigns = DissimilarityMixtureEncoderCov(
            alpha=alpha, n_clusters=n_clusters,
            dissimilarity=dissimilarities.mahalanobis
            )(inp)

    assign_model = Model(
            inputs=[inp],
            outputs=[assigns]
            )
    assign_model.layers[-1].set_weights(
            model.layers[1].get_weights()
            )

    # visualize the results
    fig1, ax1 = decision_region(
            encoder_model = assign_model,
            title="Voronoi Regions",
            X=X, batch_size=batch_size,
            figsize=(10, 10)
            )

    fig2, ax2 = probability_region(
            encoder_model = assign_model,
            title="Posterior distributions",
            X=X, batch_size=batch_size,
            n_clusters=n_clusters,
            rows=3, cols=2, figsize=(10, 20)
            )

    # save the results
    save_pdf(
            [fig1, fig2],
            "painwheel.pdf"
            )
