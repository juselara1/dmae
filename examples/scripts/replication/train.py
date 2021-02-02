import tensorflow as _tf

from sklearn.cluster import KMeans as _KMeans
from dmae.initializers import InitKMeansCov as _InitKMeansCov
from dmae.initializers import InitPlusPlus as _InitPlusPlus
from copy import deepcopy
from utils import dissimilarities_lut as _dissimilarities_lut

def pretrain(
        models, datasets,
        params, logger,
        iteration=1, 
        dissimilarity="euclidean"
        ):
    #TODOC
    params = deepcopy(params)
    use_cov = params.pop("use_cov")
    n_clusters = params.pop("n_clusters")
    iters = params.pop("iters")

    # pretrain encoder and decoder
    models["autoencoder"].fit(
            datasets["pretrain"],
            **params
            )

    # pretrain dmae
    X_latent = models["encoder"].predict(
            datasets["clustering"], 
            steps=params["steps_per_epoch"]
            )
    weights = models["full_model"].layers[1].\
            layers[2].get_weights()

    if dissimilarity in ["euclidean", "mahalanobis"]:
        pretrainer = _KMeans(n_clusters).fit(X_latent)
        weights[0] = pretrainer.cluster_centers_
    else:
        dis_dict = _dissimilarities_lut(
                {"dissimilarity": dissimilarity}
                )
        init_centers = _InitPlusPlus(
                X_latent, n_clusters,
                **dis_dict,
                iters=iters
                )
        weights[0] = init_centers(None, _tf.float32)

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

def train(
        models, datasets,
        params, logger,
        iteration=1
        ):
    models["full_model"].fit(
            datasets["clustering"],
            **params
            )
    weights = models["full_model"].layers[1].\
            layers[2].get_weights()

    models["assign_model"].\
            layers[2].set_weights(weights)

    # log final results 
    logger(
            "assign_model", "clustering", 
            iteration
            )

