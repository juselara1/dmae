from dmae.layers import DissimilarityMixtureAutoencoderCov, DissimilarityMixtureEncoderCov
from dmae.initializers import InitKMeans, InitIdentityCov
from dmae import dissimilarities, losses

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant

from sklearn.datasets import make_circles
from sklearn.kernel_approximation import RBFSampler
from sklearn.cluster import KMeans

from visutils import visualize_regions, visualize_probas
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == '__main__':
    # Dataset generation parameters
    n_samples = 1000
    n_clusters = 2
    noise = 0.1
    factor = 0.1

    # softmax inverse temperature
    alpha = 1000

    # pretrain parameters
    batch_size = 32
    pretrain_epochs = 100
    pretrain_lr = 1e-3

    # RBFSampler parameters (for pretrain)
    gamma = 10
    n_components = 100

    # cluster parameters
    cluster_epochs = 80
    cluster_lr = 1e-5

    # loss weights
    lambda_r = 0.5
    lambda_c = 1.0

    # Deep autoencoder parameters
    n_units = 256
    activation = "sigmoid"

    # generate dataset
    X, _ = make_circles(
            n_samples, noise=noise,
            factor=factor)

    # pretrainer
    rbf_feature = RBFSampler(
            gamma=gamma, 
            n_components=n_components,
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

    # Encoder pretrain
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

    # Decoder pretrain
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

    # Pretrainer for DMAE
    X_latent = encoder.predict(X)

    pretrainer = KMeans(
            n_clusters=n_clusters
            ).fit(X_latent)

    # Defining dmae model
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

    # Loss function
    loss1 = losses.mahalanobis_loss(H, *theta_tilde, alpha)
    loss2 = tf.losses.mse(inp, X_tilde)

    loss = lambda_c*loss1 + lambda_r*loss2
    
    full_model.add_loss(loss)
    full_model.compile(optimizer=Adam(lr=cluster_lr))

    # Training
    full_model.fit(
            X,
            batch_size=batch_size,
            epochs=cluster_epochs,
            )

    # Auxiliar model to obtain the assignments
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

    # Visualize results
    with PdfPages("circles.pdf") as pdf:
        # Regions
        fig, ax = visualize_regions(
                encoder_model = assign_model,
                name="Voronoi Regions",
                X=X, batch_size=batch_size,
                show_clusters=False,
                figsize=(10, 10)
                )
        pdf.savefig(fig)

        # Distributions
        fig, ax = visualize_probas(
                encoder_model = assign_model,
                name="Posterior distributions",
                X=X, batch_size=batch_size,
                n_clusters=n_clusters,
                rows=1, cols=2, figsize=(15, 8)
                )
        pdf.savefig(fig)
