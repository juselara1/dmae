from argparse import ArgumentParser
import numpy as np

import utils, train, logger, models
from dmae.metrics import unsupervised_classification_accuracy as uacc
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ars

def make_parser():
    parser = ArgumentParser()
    parser.add_argument("--encoder_params", type=str, default="encoder_params.json")
    parser.add_argument("--decoder_params", type=str, default="decoder_params.json")
    parser.add_argument("--dmae_params", type=str, default="dmae_params.json")
    parser.add_argument("--experiment_params", type=str, default="experiment_params.json")
    parser.add_argument("--dataset_params", type=str, default="dataset_params.json")
    parser.add_argument("--pretrain_params", type=str, default="pretrain_params.json")
    parser.add_argument("--train_params", type=str, default="train_params.json")
    parser.add_argument("--loss_params", type=str, default="loss_params.json")
    return parser

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    arguments = utils.json_arguments(args)

    models = models.make_models(
            arguments["encoder_params"],
            arguments["decoder_params"],
            arguments["dmae_params"],
            arguments["loss_params"],
            arguments["dataset_params"]["input_shape"]
            )

    # Make datasets
    # TODO
    datasets = {
            "autoencoder": np.random.uniform(0, 1, (100, 2)),
            "full_model": np.random.uniform(0, 1, (100, 2)),
            "labels" : np.random.randint(0, 2, (100, )),
            "input_shape": (2, )
            }

    metrics = {
            "uACC": lambda y_true, y_pred:\
                    uacc(
                        y_true, np.argmax(
                            y_pred, axis=1
                            )
                        ),
            "NMI": lambda y_true, y_pred:\
                    nmi(
                        y_true, np.argmax(
                            y_pred, axis=1
                            )
                        ),
            "ARS": lambda y_true, y_pred:\
                    ars(
                        y_true, np.argmax(
                            y_pred, axis=1
                            )
                        )
            }

    scorer = logger.Logger(
            models, metrics, 
            datasets, "mnist"
            )

    # Pretrain
    train.pretrain(
            models, datasets,
            arguments["pretrain_params"],
            scorer,
            iteration=1
            )
    scorer.save()
"""
    # Training
    train.training(
            models, datasets,
            training_arguments
            )
"""
