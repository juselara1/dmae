from argparse import ArgumentParser
import numpy as np

import utils, train, logger, models, datasets, metrics
from dmae.metrics import unsupervised_classification_accuracy as uacc
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ars

# GPU VRAM settings:
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def make_parser():
    parser = ArgumentParser()
    parser.add_argument("--encoder_params", type=str, default="hyperparams/encoder_params.json")
    parser.add_argument("--decoder_params", type=str, default="hyperparams/decoder_params.json")
    parser.add_argument("--dmae_params", type=str, default="hyperparams/dmae_params.json")
    parser.add_argument("--dataset_params", type=str, default="hyperparams/dataset_params.json")
    parser.add_argument("--pretrain_params", type=str, default="hyperparams/pretrain_params.json")
    parser.add_argument("--train_params", type=str, default="hyperparams/train_params.json")
    parser.add_argument("--loss_params", type=str, default="hyperparams/loss_params.json")
    parser.add_argument("--iters", type=int, default=5)
    return parser

def make_iteration(arguments, scorer, iteration):
    # make models
    cur_models = models.make_models(
            arguments["encoder_params"],
            arguments["decoder_params"],
            arguments["dmae_params"],
            arguments["loss_params"],
            arguments["dataset_params"]["input_shape"]
            )

    # make datasets
    cur_datasets = datasets.make_datasets(
            **arguments["dataset_params"]
            )

    # make metrics
    cur_metrics = metrics.make_metrics()

    # logger
    scorer.add(cur_models, cur_metrics, cur_datasets)
    
    # Pretrain
    arguments["pretrain_params"]\
            ["steps_per_epoch"] = cur_datasets["steps"]
    arguments["train_params"]\
            ["steps_per_epoch"] = cur_datasets["steps"]

    train.pretrain(
            cur_models, cur_datasets,
            arguments["pretrain_params"],
            scorer, iteration=iteration,
            dissimilarity=arguments["dmae_params"]["dissimilarity"]
            )
    train.train(
            cur_models, cur_datasets,
            arguments["train_params"],
            scorer, iteration=iteration
            )

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    arguments = utils.json_arguments(args)
    scorer = logger.Logger(
            arguments["dataset_params"]["dataset_name"]
            )

    for iteration in range(args.iters):
        make_iteration(arguments, scorer, iteration)
    scorer.save()
