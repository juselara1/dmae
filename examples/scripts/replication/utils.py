import json as _json

from tensorflow.keras.layers import Dense as _Dense, Conv2D as _Conv2D, Conv2DTranspose as _Conv2DTranspose
from tensorflow.keras.layers import Reshape as _Reshape, Flatten as _Flatten
from tensorflow.keras.initializers import VarianceScaling as _VarianceScaling
from tensorflow.keras.optimizers import SGD as _SGD, Adam as _Adam

from dmae import dissimilarities as _dissimilarities

from dmae.layers import DissimilarityMixtureAutoencoder as _DissimilarityMixtureAutoencoder
from dmae.layers import DissimilarityMixtureEncoder as _DissimilarityMixtureEncoder
from dmae.layers import DissimilarityMixtureAutoencoderCov as _DissimilarityMixtureAutoencoderCov
from dmae.layers import DissimilarityMixtureEncoderCov as _DissimilarityMixtureEncoderCov

_dissimilarities_hash = {
        "euclidean": _dissimilarities.euclidean,
        "manhattan": _dissimilarities.manhattan,
        "minkowsky": _dissimilarities.minkowsky,
        "chebyshev": _dissimilarities.chebyshev,
        "cosine": _dissimilarities.cosine,
        "kullback_leibler": _dissimilarities.kullback_leibler,
        "mahalanobis": _dissimilarities.mahalanobis,
        "toroidal_euclidean": _dissimilarities.toroidal_euclidean
        }

_layer_hash = {
        "dense": _Dense,
        "conv": _Conv2D,
        "convtrans": _Conv2DTranspose,
        "reshape": _Reshape,
        "flatten": _Flatten
        }

_dmae_hash = {
        False : (
            _DissimilarityMixtureAutoencoder,
            _DissimilarityMixtureEncoder
            ),
        True: (
            _DissimilarityMixtureAutoencoderCov,
            _DissimilarityMixtureEncoderCov
            )
        }

_initializers_hash = {
        "variance": _VarianceScaling(
            scale=1.0/3,
            mode="fan_in",
            distribution="uniform"
            ),
        }

_optimizer_hash = {
        "sgd": _SGD,
        "adam": _Adam
        }

def layers_lut(params):
    #TODOC
    kind = params.pop("kind")
    return _layer_hash[kind](**params)

def dissimilarities_lut(params):
    #TODOC
    params["dissimilarity"] = _dissimilarities_hash[
            params["dissimilarity"]
            ]
    return params 

def dmae_lut(params):
    #TODOC
    use_cov = params.pop("use_cov")
    dmae_layers = (i(**params) for i in _dmae_hash[use_cov])
    return dmae_layers

def initializer_lut(params):
    params["kernel_initializer"] = _initializers_hash[
            params["kernel_initializer"]
            ]
    return params

def optimizer_lut(params):
    opt_params = params.pop("optimizer_params")
    params["optimizer"] = _optimizer_hash[
            params["optimizer"]
            ](**opt_params)
    return params

def json_arguments(args):
    #TODOC
    arguments = {
            "encoder_params": args.encoder_params, 
            "decoder_params": args.decoder_params,
            "dmae_params": args.dmae_params,
            "experiment_params": args.experiment_params,
            "dataset_params": args.dataset_params,
            "pretrain_params": args.pretrain_params,
            "loss_params": args.loss_params
            }

    parameters = {}
    for name, path in arguments.items():
        with open(path) as f:
            parameters[name] = _json.load(f)
    return parameters
