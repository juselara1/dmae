import tensorflow as _tf

from sklearn.cluster import KMeans as _KMeans
from dmae.initializers import InitKMeansCov as _InitKMeansCov
from copy import deepcopy

def pretrain(
        models, datasets,
        params, logger,
        iteration=1
        ):
    #TODOC
    params = deepcopy(params)
    use_cov = params.pop("use_cov")
    n_clusters = params.pop("n_clusters")

    # pretrain encoder and decoder
    models["autoencoder"].fit(
            datasets["pretrain"].repeat(),
            **params
            )

    # pretrain dmae
    X_latent = models["encoder"].predict(
            datasets["test"]
            )

    pretrainer = _KMeans(n_clusters).fit(X_latent)
    weights = models["full_model"].layers[1].\
            layers[2].get_weights()
    weights[0] = pretrainer.cluster_centers_

    if use_cov:
        init_cov = _InitKMeansCov(
                pretrainer,
                X_latent,
                n_clusters
                )

        weights[1] = init_cov(None, _tf.float32)
    
    models["full_model"].layers[1].\
            layers[2].set_weights(weights)

    models["assign_model"].\
            layers[2].set_weights(weights)

    # log results before dmae training
    logger(
            "assign_model", "pretrain", 
            iteration
            )
