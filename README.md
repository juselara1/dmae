# Dissimilarity Mixture Autoencoder

Tensorflow implementation of the Dissimilarity Mixture Autoencoder.

Please install the following requirements:

* tensorflow==2.2.0
* numpy==1.18.1
* matplotlib==3.1.2
* scikit-learn==0.23.1

A replication of the experiments can be found in the folder `examples` (we highly recommend to check the synthetic experiments first, they provide an interactive experience that is useful to interpret the learned representations of DMAE).

* Jupyter notebooks are provided for an interactive visualization and exploration on synthetic data `examples/synthetic/`.
* For the real data `examples/real`, you can run `python experiments.py -h` for more information about its flexible usage on different datasets with different hyperparameters.