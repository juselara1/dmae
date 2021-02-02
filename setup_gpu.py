from setuptools import setup, find_packages
import argparse

# requirements
with open("requirements_gpu.txt", "r") as f:
    requirements = f.readlines()

# readme
with open("README.md", "r") as f:
    readme = f.read()

setup(
        name="dmae-gpu",
        version="1.1.2",
        author="Juan S. Lara",
        author_email="julara@unal.edu.co",
        packages=find_packages(),
        install_requires=requirements,
        description="TensorFlow implementation of the dissimilarity mixture autoencoder (DMAE)",
        license="MIT",
        long_description=readme,
        long_description_content_type="text/markdown",
        url="https://dmae.readthedocs.io/en/latest/"
        )
