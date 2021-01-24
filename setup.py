from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

setup(
        name="dmae",
        version="1.1",
        author="Juan S. Lara",
        author_email="julara@unal.edu.co",
        packages=find_packages(),
        install_requires=requirements,
        description="TensorFlow implementation of the dissimilarity mixture autoencoder (DMAE)",
        license="MIT"
)
