# Dissimilarity Mixture Autoencoder

![dmae](https://raw.githubusercontent.com/larajuse/Resources/master/dmae/dmae.svg)
![example](https://raw.githubusercontent.com/larajuse/Resources/fed6276cf237f6b47b816af3d2a32c6508e00f1e/dmae/nonglobular.svg)

Tensorflow implementation of the Dissimilarity Mixture Autoencoder:

* Juan S. Lara and Fabio A. González. ["Dissimilarity Mixture Autoencoder for Deep Clustering"](https://arxiv.org/abs/2006.08177) arXiv preprint arXiv:2006.08177 (2020).

## Abstract

In this paper, we introduce the Dissimilarity Mixture Autoencoder (DMAE), a novel neural network model that uses a dissimilarity function to generalize a family of density estimation and clustering methods. It is formulated in such a way that it internally estimates the parameters of a probability distribution through gradient-based optimization. Also, the proposed model can leverage from deep representation learning due to its straightforward incorporation into deep learning architectures, because, it consists of an encoder-decoder network that computes a probabilistic representation. Experimental evaluation was performed on image and text clustering benchmark datasets showing that the method is competitive in terms of unsupervised classification accuracy and normalized mutual information.

## Requirements

If you have [anaconda](https://www.anaconda.com/) installed, you can create the same environment that we used for the experiments using the following command:

```
conda env create -f dmae_env.yml
```

Then, you must activate the environment:

```
source activate dmae
```

or 

```
conda activate dmae
```

## Usage

This implementation is based on `tf.keras.layers`, therefore, DMAE can be easily used in other deep learning models as an intermediate layer. A replication of the experiments can be found in the folder `examples`, we highly recommend to check the experiments on synthetic data first: `examples/synthetic/`. They provide an interactive experience that is useful to interpret the learned representations of DMAE.

For the real data `examples/real`, you can run `python experiments.py -h` for more information about the possible options.

Some examples are:

* Quick test:

```
python experiments.py --trials 1 --pretrain_epochs 1 --cluster_epochs 1
```

* Replication of the MNIST results using the euclidean dissimilarity:

```
python experiments.py --dataset mnist --trials 10 --pretrain_epochs 500 --cluster_epochs 300 --da True --train_batch 256 --test_batch 1000 --dis euclidean
```

* Replication of the MNIST results using the mahalanobis dissimilarity:

```
python experiments.py --dataset mnist --trials 10 --pretrain_epochs 500 --cluster_epochs 300 --da True --train_batch 256 --test_batch 1000 --dis mahalanobis
```

* Replication of the Reuters experiments:

```
python experiments.py --dataset reuters10 --trials 10 --pretrain_epochs 100 --cluster_epochs 100 --train_batch 256 --test_batch 1000 --dis euclidean
```
