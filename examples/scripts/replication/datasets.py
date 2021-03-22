import os as _os
import glob as _glob
import h5py as _h5py
import numpy as _np

import tensorflow as _tf
from tensorflow import image as _tfi
from tensorflow.data import Dataset as _Dataset
import tensorflow_addons as _tfa
from tensorflow_addons.image.transform_ops import angles_to_projective_transforms as _angle2projective

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

def _parse_example(record, input_shape):
    features = {
            "x": _tf.io.FixedLenFeature(
                _np.prod(input_shape),
                _tf.float32
                ),
            "y": _tf.io.FixedLenFeature(
                [1], _tf.int64
                )
            }
    decoded = _tf.io.parse_single_example(
            record, features
            )
    return list(decoded.values())

def make_datasets(**kwargs):
    ds = _tf.data.TFRecordDataset(
            _glob.glob(
                _os.path.join(
                    kwargs["path"],
                    kwargs["dataset_name"],
                    "*"
                    )
                )
            )
    ds = ds.map(
            lambda record: 
            _parse_example(
                record, kwargs["input_shape"]
                )
            )

    if len(kwargs["input_shape"])>1:
        ds = ds.map(
                lambda x, y: (
                    _tf.reshape(
                        x,
                        kwargs["input_shape"]
                        ),
                    y
                    )
                )

    ds_test = ds.repeat()

    ds_train = ds_test.map(
            lambda x, y: x
            )

    ds_pretrain = ds_train.batch(
        kwargs["pretrain_batch"],
        drop_remainder=True
        )

    # augmentation
    if kwargs["augment_autoencoder"]:
        ds_pretrain = ds_pretrain.map(
                lambda batch: _apply_transform(
                    batch, kwargs["augmentation"]
                    )
                )
    ds_pretrain = ds_pretrain.map(
            lambda x: (x, x)
            )

    ds_clustering = ds_train.batch(
            kwargs["cluster_batch"],
            drop_remainder=True
            )
    if kwargs["augment_clustering"]:
        ds_clustering = ds_clustering.map(
                lambda batch: _apply_transform(
                    batch, kwargs["augmentation"]
                    )
                )

    return {
            "pretrain": ds_pretrain,
            "clustering": ds_clustering,
            "test": ds_test,
            "pretrain_steps": kwargs["n_samples"]//kwargs["pretrain_batch"],
            "cluster_steps": kwargs["n_samples"]//kwargs["cluster_batch"]
            }
