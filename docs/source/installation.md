# Installation

You can install `dmae` from PyPI using pip, from the source [Github repository](https://github.com/juselara1/dmae) or pulling a preconfigured docker image.

## PyPI

To install `dmae` using pip you can run the followiing command:

```sh
pip install dmae
```

*(optional) If you have an environment with the nvidia drivers and CUDA, you can instead run:*

```sh
pip install dmae-gpu
```

## Source

You can clone the `dmae` [repository](https://github.com/juselara1/dmae) as follows:

```sh
git clone https://github.com/juselara1/dmae.git
```

You must install the requirements:

```sh
pip install -r requirements.txt
```

*(optional) If you have an environment with the nvidia drivers and CUDA, you can instead run:*

```sh
pip install -r requirements-gpu.txt
```

Finally, you can install `dmae` via setuptools

```sh
pip install --no-deps . 
```

## Docker

You can pull a preconfigured docker image with `dmae` from DockerHub:

```sh
docker pull juselara/dmae:1.1.0
```

*(optional) If you have the nvidia drivers installed, you can pull the following image:*

```sh
docker pull juselara/dmae:1.1.0-gpu
```
