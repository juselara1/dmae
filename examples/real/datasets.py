import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

def load_mnist(train_batch, test_batch, augmentation=None):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X = np.expand_dims(np.concatenate([X_train, X_test], axis=0)/255.0, axis=-1).astype(np.float32)
    y = np.concatenate([y_train, y_test], axis=0)
    
    ds = tf.data.Dataset.from_tensor_slices((X, ))
    ds = ds.shuffle(X.shape[0]).repeat().batch(train_batch, drop_remainder=True)
    def apply_transform(batch):
        random_angles = tf.random.uniform(shape = (batch.shape[0], ), **augmentation["rotation_range"])
        batch = tfa.image.transform(batch,
                                    tfa.image.transform_ops.angles_to_projective_transforms(
                                        random_angles, tf.cast(batch.shape[1], tf.float32),
                                        tf.cast(batch.shape[2], tf.float32)),
                                    interpolation="BILINEAR")
        random_x = tf.random.uniform(shape=(batch.shape[0], 1), **augmentation["width_shift_range"])
        random_y = tf.random.uniform(shape=(batch.shape[0], 1), **augmentation["height_shift_range"])
        translate = tf.concat([random_x, random_y], axis=1)
        batch = tfa.image.translate(batch, translations=translate, interpolation="BILINEAR")
        return batch
    ds_aug = ds.map(apply_transform)
    ds_aug = ds_aug.map(lambda batch: tf.reshape(batch, [batch.shape[0], tf.math.reduce_prod(batch.shape[1:])]))
    ds_aug = ds_aug.map(lambda batch: (batch, batch))
    ds_cluster = ds.map(lambda batch: tf.reshape(batch, [batch.shape[0], tf.math.reduce_prod(batch.shape[1:])]))
    return ds_aug, ds_cluster, np.reshape(X, (-1, 28**2)), y

def load_fashion(train_batch, test_batch, augmentation=None):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X = np.expand_dims(np.concatenate([X_train, X_test], axis=0)/255.0, axis=-1).astype(np.float32)
    y = np.concatenate([y_train, y_test], axis=0)
    
    ds = tf.data.Dataset.from_tensor_slices((X, ))
    ds = ds.shuffle(X.shape[0]).repeat().batch(train_batch, drop_remainder=True)
    def apply_transform(batch):
        random_angles = tf.random.uniform(shape = (batch.shape[0], ), **augmentation["rotation_range"])
        batch = tfa.image.transform(batch,
                                    tfa.image.transform_ops.angles_to_projective_transforms(
                                        random_angles, tf.cast(batch.shape[1], tf.float32),
                                        tf.cast(batch.shape[2], tf.float32)),
                                    interpolation="BILINEAR")
        random_x = tf.random.uniform(shape=(batch.shape[0], 1), **augmentation["width_shift_range"])
        random_y = tf.random.uniform(shape=(batch.shape[0], 1), **augmentation["height_shift_range"])
        translate = tf.concat([random_x, random_y], axis=1)
        batch = tfa.image.translate(batch, translations=translate, interpolation="BILINEAR")
        return batch
    ds_aug = ds.map(apply_transform)
    ds_aug = ds_aug.map(lambda batch: tf.reshape(batch, [batch.shape[0], tf.math.reduce_prod(batch.shape[1:])]))
    ds_aug = ds_aug.map(lambda batch: (batch, batch))
    ds_cluster = ds.map(lambda batch: tf.reshape(batch, [batch.shape[0], tf.math.reduce_prod(batch.shape[1:])]))
    return ds_aug, ds_cluster, np.reshape(X, (-1, 28**2)), y

def load_reuters10(train_batch, test_batch):
    data = np.load("/tf/reuters/reutersidf10k.npy", allow_pickle=True).item()
    X = data["data"]
    y = data["label"]
    ds = tf.data.Dataset.from_tensor_slices((X, ))
    ds_pretrain = ds.shuffle(X.shape[0]).repeat().batch(train_batch)
    ds_pretrain = ds_pretrain.map(lambda batch: (batch, batch))
    ds_cluster = ds.shuffle(X.shape[0]).repeat().batch(train_batch)
    return ds_pretrain, ds_pretrain, X, y

def load_usps(train_batch, test_batch, augmentation=None):
    data = np.load("/tf/usps/usps.npy", allow_pickle=True).item()
    X = data["data"]
    y = data["label"]
    ds = tf.data.Dataset.from_tensor_slices((X, ))
    ds = ds.shuffle(X.shape[0]).repeat().batch(train_batch, drop_remainder=True)
    def apply_transform(batch):
        random_angles = tf.random.uniform(shape = (batch.shape[0], ), **augmentation["rotation_range"])
        batch = tfa.image.transform(batch,
                                    tfa.image.transform_ops.angles_to_projective_transforms(
                                        random_angles, tf.cast(batch.shape[1], tf.float32),
                                        tf.cast(batch.shape[2], tf.float32)),
                                    interpolation="BILINEAR")
        random_x = tf.random.uniform(shape=(batch.shape[0], 1), **augmentation["width_shift_range"])
        random_y = tf.random.uniform(shape=(batch.shape[0], 1), **augmentation["height_shift_range"])
        translate = tf.concat([random_x, random_y], axis=1)
        batch = tfa.image.translate(batch, translations=translate, interpolation="BILINEAR")
        return batch
    ds_aug = ds.map(apply_transform)
    ds_aug = ds_aug.map(lambda batch: tf.reshape(batch, [batch.shape[0], tf.math.reduce_prod(batch.shape[1:])]))
    ds_aug = ds_aug.map(lambda batch: (batch, batch))
    ds_cluster = ds.map(lambda batch: tf.reshape(batch, [batch.shape[0], tf.math.reduce_prod(batch.shape[1:])]))
    return ds_aug, ds_cluster, np.reshape(X, (-1, 28**2)), y