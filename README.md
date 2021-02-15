# Dissimilarity Mixture Autoencoder for Deep Clustering

<a href="https://pypi.python.org/pypi/dmae"><img src="https://img.shields.io/pypi/v/dmae.svg"/></a>
<a href="https://hub.docker.com/repository/docker/juselara/dmae"><img src="https://img.shields.io/badge/docker-v1.1.2-blue"></a>

Tensorflow implementation of the Dissimilarity Mixture Autoencoder:

* Juan S. Lara and Fabio A. González. ["Dissimilarity Mixture Autoencoder for Deep Clustering"](https://arxiv.org/abs/2006.08177) arXiv preprint arXiv:2006.08177 (2020).

## Abstract

The dissimilarity mixture autoencoder (DMAE) is a neural network model for feature-based clustering that incorporates a flexible dissimilarity function and can be integrated into any kind of deep learning architecture. It internally represents a dissimilarity mixture model (DMM) that extends classical methods like Bregman clustering to any convex and differentiable dissimilarity function through the reinterpretation of probabilistic notions as neural network components. Likewise, it leverages from unsupervised representation learning, allowing a simultaneous learning of the clusters and neural network's parameters. Experimental evaluation was performed on image and text clustering benchmark datasets showing that DMAE is competitive in terms of unsupervised classification accuracy and normalized mutual information.

## Usage and Documentation

You can check the official `dmae` [documentation](https://dmae.readthedocs.io/en/latest/index.html).

## Gallery and Examples

* Deep architecture:

    ![dmae](https://raw.githubusercontent.com/larajuse/Resources/master/dmae/dmae.svg)

* Clustering examples:
    ![clustering](https://raw.githubusercontent.com/juselara1/Resources/master/dmae/clustering_examples.svg)

* Probabilistic interpretations:
    ![probabilistic](https://raw.githubusercontent.com/juselara1/Resources/master/dmae/probabilistic.svg)

These examples and the paper replication experiments can be found in the [examples](https://github.com/juselara1/dmae/tree/main/examples) folder.

## Installation

You can install `dmae` from PyPi using `pip`, building from source or pulling a preconfigured docker image.

### PyPi

To install `dmae` using `pip` you can run the following command:

```sh
pip install dmae
```

*(optional) If you have an environment with the nvidia drivers and CUDA, you can instead run:*

```sh
pip install dmae-gpu
```

### Source

You can clone this repository:

```sh
git clone https://github.com/juselara1/dmae.git
```

Install the requirements:

```sh
pip install -r requirements.txt
```

*(optional) If you have an environment with the nvidia drivers and CUDA, you can instead run:*

```sh
pip install -r requiremets-gpu.txt
```

Finally, you can install `dmae` via setuptools:

```sh
pip install --no-deps .
```

### Docker 

You can pull a preconfigured docker image with `dmae` from DockerHub:

```sh
docker pull juselara/dmae:latest
```

*(optional) If you have an environment with the nvidia drivers installed, you can instead run:*

```sh
docker pull juselara/dmae:latest-gpu
```

## Citation

```
@misc{lara2020dissimilarity,
      title={Dissimilarity Mixture Autoencoder for Deep Clustering}, 
      author={Juan S. Lara and Fabio A. González},
      year={2020},
      eprint={2006.08177},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
