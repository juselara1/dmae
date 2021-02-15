## -*- coding: utf-8 -*-
"""
Example script of clustering circles data with DMAE.
"""
# Author: Juan S. Lara <julara@unal.edu.co>
# License: MIT

from dmae.layers import DissimilarityMixtureAutoencoderCov, DissimilarityMixtureEncoderCov
from dmae.initializers import InitKMeans, InitIdentityCov
from dmae import dissimilarities, losses

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant

from sklearn.datasets import make_moons
from sklearn.kernel_approximation import RBFSampler
from sklearn.cluster import KMeans

from visutils import decision_region, probability_region, save_pdf

if __name__ == '__main__':
    # dataset generation parameters
    n_samples = 1000
    noise = 0.1
    random_state = 0 

    # number of clusters
    n_clusters = 2
    
    # softmax inverse temperature
    alpha = 1000

    # pretrain parameters
    batch_size = 32
    pretrain_epochs = 250
    pretrain_lr = 1e-4

    # RBFSampler parameters (for pretrain)
    gamma = 30
    n_components = 100

    # dmae training parameters
    cluster_epochs = 100
    cluster_lr = 1e-4

    # loss weights
    lambda_r = 0.5
    lambda_c = 1.0

    # deep autoencoder parameters
    n_units = 256
    activation = "relu"

    # generate dataset
    X, _ = make_moons(
            n_samples, noise=noise,
            random_state=random_state
            )

    # pretrainer
    rbf_feature = RBFSampler(
            gamma=gamma, 
            n_components=n_components,
            random_state=random_state
            )

    X_features = rbf_feature.fit_transform(X)

    # deep encoder
    encoder = Sequential([
        Dense(
            n_units, 
            activation=activation, 
            input_shape=(2, )
            ),
        Dense(
            n_units,
            activation=activation
            ),
        Dense(
            n_components,
            activation="linear"
            )
        ])

    # deep decoder
    decoder = Sequential([
        Dense(
            n_units,
            activation=activation,
            input_shape=(n_components, )
            ),
        Dense(
            n_units,
            activation=activation
            ),
        Dense(
            2,
            activation="linear"
            )
        ])

    # deep autoencoder
    inp = Input(shape=(2, ))
    out = decoder(encoder(inp))
    autoencoder = Model(
            inputs=[inp],
            outputs=[out]
            )

    # encoder pretrain
    print("Pretraining encoder", end="\r")
    encoder.compile(
            loss="mse",
            optimizer=Adam(lr=pretrain_lr)
            )
    encoder.fit(
            X, X_features,
            batch_size=batch_size,
            epochs=pretrain_epochs, 
            verbose=False
            )

    # decoder pretrain
    print("Pretraining decoder", end="\r")
    autoencoder.layers[1].trainable = False
    autoencoder.compile(
            loss="mse",
            optimizer=Adam(lr=pretrain_lr)
            )

    autoencoder.fit(
            X, X, 
            batch_size=batch_size,
            epochs=pretrain_epochs,
            verbose=False
            )

    autoencoder.layers[1].trainable = True

    # pretrainer for DMAE
    X_latent = encoder.predict(X)

    pretrainer = KMeans(
            n_clusters=n_clusters
            ).fit(X_latent)

    # defining dmae model
    inp = Input(shape=(2, ))

    H = encoder(inp)

    theta_tilde = DissimilarityMixtureAutoencoderCov(
            alpha, n_clusters,
            initializers={
                "centers": InitKMeans(pretrainer),
                "cov": InitIdentityCov(X_latent, n_clusters),
                "mixers": Constant(1.0)
                }, 
            dissimilarity=dissimilarities.mahalanobis
            )(H)

    X_tilde = decoder(theta_tilde[0])

    full_model = Model(
            inputs=[inp],
            outputs=[X_tilde]
            )

    # loss function
    loss1 = losses.mahalanobis_loss(H, *theta_tilde, alpha)
    loss2 = tf.losses.mse(inp, X_tilde)

    loss = lambda_c*loss1 + lambda_r*loss2
    
    full_model.add_loss(loss)
    full_model.compile(optimizer=Adam(lr=cluster_lr))

    # training
    full_model.fit(
            X,
            batch_size=batch_size,
            epochs=cluster_epochs,
            )

    # auxiliar model to obtain the assignments
    inp = Input(shape=(2, ))
    H = encoder(inp)
    assigns = DissimilarityMixtureEncoderCov(
            alpha, n_clusters=n_clusters,
            dissimilarity=dissimilarities.mahalanobis
            )(H)

    assign_model = Model(
        inputs=[inp],
        outputs=[assigns]
        )
    assign_model.layers[-1].set_weights(
            full_model.layers[2].get_weights()
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
            rows=1, cols=2, figsize=(15, 8)
            )

    # save the results
    save_pdf(
            [fig1, fig2],
            "moons.pdf"
            )
