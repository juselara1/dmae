# Importing libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from kmedoids import KMedoids
import argparse, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# sys.path.append("/tf/home/repositorios/DMAE/") # how to use the code in other location
sys.path.append("../../")
import DMAE, datasets, FC_dmae, CNN_dmae


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train a specified model.')
    parser.add_argument("--dataset", type=str, help="Dataset (e.g., mnist, reuters10, fashion)", default="mnist")
    parser.add_argument("--model", type=str, help="Model to be trained (e.g., FC_dmae)", default="FC_dmae")
    parser.add_argument("--trials", type=int, action="store", dest="trials",
                        help="Number of trials to train the model.", default=10)
    parser.add_argument("--dis", type=str, help="Dissimilarity function for DMAE model (e.g., euclidean, cosine, mahalanobis, manhattan)", default="euclidean")
    parser.add_argument("--train_batch", type=int, help="Batch size during training", default=256)
    parser.add_argument("--test_batch", type=int, help="Batch size during testing (recomended N mod test_batch==0)", default=1000)
    parser.add_argument("--da", type=bool, help="Specify if there must be data augmentation", default=False)
    parser.add_argument("--pretrain_epochs", type=int, help="Number of epochs for autoencoder pretrain", default=500)
    parser.add_argument("--cluster_epochs", type=int, help="Number of epochs for clustering", default=500)

    args = parser.parse_args()
    if args.dataset == "mnist":
        augmentation = {"width_shift_range": 0.1, "height_shift_range": 0.1, "rotation_range": 10}
        ds_pretrain, ds_cluster, ds_test, y = datasets.load_mnist(args.train_batch, args.test_batch, args.da,
                                                                  "CNN" in args.model, augmentation)
        # Defining hyperparameters
        N = 70000; n_clusters = 10; latent_dim = 10
        if "CNN" in args.model:
            input_shape = (28, 28, 1)
        else:
            input_shape = (28**2, )
        # Define optimizers
        pretrain_optimizer = {"type": tf.optimizers.SGD, "params": {"lr": 1, "momentum": 0.9}}
        cluster_optimizer = {"type": tf.optimizers.Adam, "params": {"lr": 1e-5}}
    elif args.dataset == "reuters10":
        ds_pretrain, ds_cluster, ds_test, y = datasets.load_reuters10(args.train_batch, args.test_batch)
        # Defining hyperparameters
        N = 10000; n_clusters = 4; latent_dim = 10
        input_shape = (2000, )
        # Define optimizers
        pretrain_optimizer = {"type": tf.optimizers.SGD, "params": {"lr": 1, "momentum": 0.9}}
        cluster_optimizer = {"type": tf.optimizers.Adam, "params": {"lr": 1e-5}}
    elif args.dataset == "usps":
        augmentation = {"width_shift_range": 0.1, "height_shift_range": 0.1, "rotation_range": 10}
        ds_pretrain, ds_cluster, ds_test, y = datasets.load_usps(args.train_batch, args.test_batch, args.da,
                                                                 "CNN" in args.model, augmentation)
        N = 9298; n_clusters = 10; latent_dim = 10
        if "CNN" in args.model:
            input_shape = (16, 16, 1)
        else:
            input_shape = (16**2, )
        # Define optimizers
        pretrain_optimizer = {"type": tf.optimizers.SGD, "params": {"lr": 1, "momentum": 0.9}}
        cluster_optimizer = {"type": tf.optimizers.Adam, "params": {"lr": 1e-5}}
    elif args.dataset == "fashion":
        augmentation = {"width_shift_range": 0.1, "height_shift_range": 0.1, "rotation_range": 10}
        ds_pretrain, ds_cluster, ds_test, y = datasets.load_fashion(args.train_batch, args.test_batch, args.da,
                                                                    "CNN" in args.model, augmentation)
        # Defining hyperparameters
        N = 70000; n_clusters = 10; latent_dim = 10
        if "CNN" in args.model:
            input_shape = (28, 28, 1)
        else:
            input_shape = (28**2, )
        # Define optimizers
        pretrain_optimizer = {"type": tf.optimizers.SGD, "params": {"lr": 1, "momentum": 0.9}}
        cluster_optimizer = {"type": tf.optimizers.Adam, "params": {"lr": 1e-5}}
    else:
        raise Exception(f"Not recognized dataset: {args.dataset}")
    
    if args.dis == "euclidean":
        make_pretrainer = lambda: KMeans(n_clusters=n_clusters)
        dis = DMAE.Dissimilarities.euclidean
        dis_loss = DMAE.Losses.euclidean_loss
        init_dmae = lambda pretrainer:{"centers": DMAE.Initializers.InitKMeans(pretrainer),
                                       "mixers": tf.keras.initializers.Constant(1.0)}
        cov = False
    
    elif args.dis == "cosine":
        make_pretrainer = lambda: KMedoids(n_clusters=n_clusters, metric="cosine")
        dis = DMAE.Dissimilarities.cosine
        dis_loss = DMAE.Losses.cosine_loss
        init_dmae = lambda pretrainer:{"centers": DMAE.Initializers.InitKMeans(pretrainer),
                                       "mixers": tf.keras.initializers.Constant(1.0)}
        cov = False
    
    elif args.dis == "manhattan":
        make_pretrainer = lambda: KMedoids(n_clusters=n_clusters, metric="manhattan")
        dis = DMAE.Dissimilarities.manhattan
        dis_loss = DMAE.Losses.manhattan_loss
        init_dmae = lambda pretrainer:{"centers": DMAE.Initializers.InitKMeans(pretrainer),
                                       "mixers": tf.keras.initializers.Constant(1.0)}
        cov = False
    
    elif args.dis=="mahalanobis":
        make_pretrainer = lambda: KMeans(n_clusters=n_clusters)
        dis = DMAE.Dissimilarities.mahalanobis
        dis_loss = DMAE.Losses.mahalanobis_loss
        init_dmae = lambda pretrainer, X_latent: {"centers": DMAE.Initializers.InitKMeans(pretrainer),
                                                  "cov": DMAE.Initializers.InitKMeansCov(pretrainer, X_latent, n_clusters),
                                                  "mixers": tf.keras.initializers.Constant(1.0)}
        cov = True
    os.system("clear")
    if args.model == "FC_dmae":
        encoder_dims = [500, 500, 2000]
        decoder_dims = [2000, 500, 500, np.prod(input_shape)]
        
        pretrain_params = {"epochs": args.pretrain_epochs, "steps_per_epoch": N//args.train_batch,
                           "verbose": False, "use_multiprocessing": True}
        cluster_params = {"epochs": args.cluster_epochs, "steps_per_epoch": N//args.train_batch,
                          "verbose": False, "use_multiprocessing": True}
        print(f"Training: {args.model}, using dissimilarity: {args.dis}, total trials: {args.trials}, augmentation {args.da}")
        input_layer = tf.keras.layers.Input(shape=input_shape)
        make_autoencoder = lambda: FC_dmae.autoencoder(encoder_dims, decoder_dims, latent_dim, input_layer)
        make_dmae = lambda encoder, decoder, X_latent, make_pretrainer: FC_dmae.deep_dmae(latent_dim, n_clusters, encoder, decoder,
                                                                                          init_dmae, args.train_batch, args.test_batch,
                                                                                          dis, dis_loss, make_pretrainer, input_shape, cov,
                                                                                          X_latent)

        FC_dmae.train(ds_pretrain, ds_cluster, ds_test, y, N, args.test_batch, args.trials,
                      make_autoencoder, make_dmae, pretrain_optimizer, cluster_optimizer,
                      make_pretrainer, pretrain_params, cluster_params)
    elif args.model == "CNN_dmae":
        pretrain_optimizer = {"type": tf.optimizers.Adam, "params": {"lr": 1e-3}}
        cluster_optimizer = {"type": tf.optimizers.Adam, "params": {"lr": 1e-6}}
        encoder_dims = [32, 64, 128, 10]
        decoder_dims = [128, 64, 32]
        pretrain_params = {"epochs": args.pretrain_epochs, "steps_per_epoch": N//args.train_batch,
                           "verbose": False, "use_multiprocessing": True}
        cluster_params = {"epochs": args.cluster_epochs, "steps_per_epoch": N//args.train_batch,
                          "verbose": False, "use_multiprocessing": True}
        print(f"Training: {args.model}, using dissimilarity: {args.dis}, total trials: {args.trials}, augmentation {args.da}")
        input_layer = tf.keras.layers.Input(shape=input_shape)
        make_autoencoder = lambda: CNN_dmae.autoencoder(encoder_dims, decoder_dims, latent_dim, input_layer, input_shape)
        make_dmae = lambda encoder, decoder, X_latent: CNN_dmae.deep_dmae(latent_dim, n_clusters, encoder, decoder,
                                                                          init_dmae, args.train_batch, args.test_batch,
                                                                          dis, dis_loss, pretrainer, input_shape, cov, X_latent)

        CNN_dmae.train(ds_pretrain, ds_cluster, ds_test, y, N, args.test_batch, args.trials,
                       make_autoencoder, make_dmae, pretrain_optimizer, cluster_optimizer,
                       pretrainer, pretrain_params, cluster_params)