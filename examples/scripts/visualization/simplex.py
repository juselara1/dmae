## -*- coding: utf-8 -*-
"""
Example script of clustering painwheel data with DMAE.
"""
# Author: Juan S. Lara <julara@unal.edu.co>
# License: MIT

from dmae.layers import DissimilarityMixtureAutoencoder, DissimilarityMixtureEncoder
from dmae.initializers import InitPlusPlus
from dmae import dissimilarities, losses

import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam

from visutils import decision_simplex, save_pdf, decision_region

def simplex_blobs(
        n_samples, cluster_std,
        centers):
    X, _ = make_blobs(
            n_samples,
            centers=centers,
            cluster_std=cluster_std
            )
    vals = np.abs(X)
    return vals/vals.sum(axis=1).reshape(-1, 1)
    
if __name__ == '__main__':
    # dataset generation parameters
    n_samples = 1000
    centers = [
            (0.0, 0.2, 0.8), 
            (0.33, 0.33, 0.33),
            (1.0, 0.0, 0.0),
            (0.0, 0.9, 0.1)
            ]
    cluster_std = 0.03

    # softmax inverse temperature
    alpha = 100

    # dmae training parameters
    batch_size = 32
    epochs = 50
    lr = 1e-4
    n_clusters = 4

    # generate data
    X = simplex_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=cluster_std,
            )

    # dissimilarity function
    dis = lambda X, Y: dissimilarities.kullback_leibler(
            X, Y, normalization="softmax_abs"
            )

    # dmae definition
    inp = Input(shape=(3, ))
    theta_tilde = DissimilarityMixtureAutoencoder(
            alpha=alpha, n_clusters=n_clusters,
            initializers={
                "centers": InitPlusPlus(
                    X, n_clusters,
                    dis
                    ), 
                "mixers": Constant(1.0)
                },
            dissimilarity=dis
            )(inp)
    model = Model(
            inputs=[inp],
            outputs=theta_tilde
            )

    # loss function
    loss = losses.kullback_leibler_loss(
            inp, *theta_tilde, 
            alpha 
            )
    model.add_loss(loss)

    model.compile(optimizer=Adam(lr=lr))
    
    # training
    model.fit(
            X, epochs=epochs,
            batch_size=batch_size
            )

    # auxiliar model to obtain the assignments
    inp = Input(shape=(3, ))
    assigns = DissimilarityMixtureEncoder(
            alpha=alpha, n_clusters=n_clusters,
            dissimilarity=dis
            )(inp)

    assign_model = Model(
            inputs=[inp],
            outputs=[assigns]
            )
    assign_model.layers[-1].set_weights(
            model.layers[1].get_weights()
            )

    # visualize the results
    fig1, ax1 = decision_simplex(
            encoder_model = assign_model,
            title="Voronoi Regions",
            X=X, batch_size=batch_size,
            scale=300
            )
    fig1.savefig("simplex.svg")
    # save the results
    save_pdf(
            [fig1],
            "kl_blobs.pdf"
            )
