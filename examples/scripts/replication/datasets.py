import h5py as _h5py
import numpy as _np

import tensorflow as _tf
from tensorflow import image as _tfi
from tensorflow.data import Dataset as _Dataset
import tensorflow_addons as _tfa
from tensorflow_addons.image.transform_ops import angles_to_projective_transforms as _angle2projective

class _Generator:
    # TODOC
    def __init__(self,
            path, batch_size,
            dataset_name
            ):
        self.__path = path
        self.__batch_size = batch_size
        self.__dataset_name = dataset_name
        self.__batch_sample()

    def __batch_sample(self):
        #TODOC
        def gen():
            self.cont = 0
            with _h5py.File(self.__path, "r") as df:
                n_samples = df[self.__dataset_name].shape[0]
            while True:
                if self.cont + self.__batch_size > n_samples:
                    self.cont = 0
                with _h5py.File(self.__path, "r") as df:
                    X = df[self.__dataset_name][self.cont: self.cont + self.__batch_size]
                    yield X
                self.cont += self.__batch_size
        self.__gen = gen()

    def reset_gen(self):
        self.__batch_sample()

    def __call__(self):
        while True:
            yield next(self.__gen)

def _apply_transform(batch, augmentation): 
    #TODOC
    random_angles = _tf.random.uniform(
            shape = (batch.shape[0], ), 
            **augmentation["random_rotation"]
            )
    batch = _tfa.image.transform(batch,
                                _angle2projective(
                                random_angles, 
                                _tf.cast(batch.shape[1], _tf.float32),
                                _tf.cast(batch.shape[2], _tf.float32)),
                                interpolation="BILINEAR"
                                )
    
    if augmentation["horizontal_flip"]:
        batch = _tfi.random_flip_left_right(batch)
    if augmentation["vertical_flip"]:
        batch = _tfi.random_flip_up_down(batch)

    random_x = _tf.random.uniform(
            shape=(batch.shape[0], 1),
            **augmentation["width_shift_range"]
            )
    random_y = _tf.random.uniform(
            shape=(batch.shape[0], 1),
            **augmentation["height_shift_range"]
            )
    delta = _tf.concat(
            [random_x, random_y],
            axis=1
            )

    batch = _tfa.image.translate(
            batch, translations=delta, 
            interpolation="BILINEAR"
            )

    return batch

def make_datasets(**kwargs):
    gen = _Generator(
            kwargs["path"],
            kwargs["batch_size"],
            kwargs["dataset_name"],
            )

    ds = _Dataset.from_generator(
            gen, 
            output_types = _tf.float32,
            output_shapes = (
                kwargs["batch_size"], *kwargs["input_shape"]
                )
            )

    ds_test = _Generator(
            kwargs["path"],
            kwargs["batch_size"],
            kwargs["dataset_name"],
            )

    # augmentation
    if kwargs["augment_autoencoder"]:
        ds_pretrain = ds.map(
                lambda batch: _apply_transform(
                    batch, kwargs["augmentation"]
                    )
                )
    else: 
        ds_pretrain = ds
    ds_pretrain = ds_pretrain.map(
            lambda batch: (batch, batch)
            )

    if kwargs["augment_clustering"]:
        ds_clustering = ds.map(
                lambda batch: _apply_transform(
                    batch, kwargs["augmentation"]
                    )
                )
    else:
        ds_clustering = ds
    
    # labels
    with _h5py.File(kwargs["path"], "r") as df:
        labels = df[kwargs["dataset_name"]+"_labels"][:]
    steps = labels.shape[0]//kwargs["batch_size"]
    labels = labels[:steps*kwargs["batch_size"]]

    return {
            "pretrain": ds_pretrain,
            "clustering": ds_clustering,
            "test": ds_test,
            "labels": labels,
            "steps": steps
            }
