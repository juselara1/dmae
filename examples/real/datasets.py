import tensorflow as tf
import numpy as np

def load_mnist(train_batch, test_batch, augment=False, conv=False, augmentation=None):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X = np.expand_dims(np.concatenate([X_train, X_test], axis=0)/255.0, axis=-1).astype(np.float32)
    y = np.concatenate([y_train, y_test], axis=0)
    ds = tf.data.Dataset.from_tensor_slices((X, ))
    ds_pretrain = ds.shuffle(X.shape[0]).repeat().batch(train_batch, drop_remainder=True)
    ds_cluster = ds.shuffle(X.shape[0]).repeat().batch(train_batch, drop_remainder=True)
    ds_test = ds.batch(test_batch, drop_remainder=True)
    if augment:
        def make_generator():
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(**augmentation)
            gen = data_gen.flow(X, batch_size=train_batch)
            while True:
                batch = next(gen)
                while batch.shape[0]!=train_batch:
                    batch = next(gen)
                yield batch
        ds_pretrain = tf.data.Dataset.from_generator(make_generator, output_types=tf.float32,
                                                     output_shapes=(train_batch, *X.shape[1:]))
    if not conv:
        ds_pretrain = ds_pretrain.map(lambda batch: tf.reshape(batch, [batch.shape[0], tf.math.reduce_prod(batch.shape[1:])]))
        ds_cluster = ds_cluster.map(lambda batch: tf.reshape(batch, [batch.shape[0], tf.math.reduce_prod(batch.shape[1:])]))
        ds_test = ds_test.map(lambda batch: tf.reshape(batch, [batch.shape[0], tf.math.reduce_prod(batch.shape[1:])]))
    ds_pretrain = ds_pretrain.map(lambda batch: (batch, batch))
    return ds_pretrain, ds_cluster, ds_test, y

def load_fashion(train_batch, test_batch, augment=False, conv=False, augmentation=None):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X = np.expand_dims(np.concatenate([X_train, X_test], axis=0)/255.0, axis=-1).astype(np.float32)
    y = np.concatenate([y_train, y_test], axis=0)
    ds = tf.data.Dataset.from_tensor_slices((X, ))
    ds_pretrain = ds.shuffle(X.shape[0]).repeat().batch(train_batch, drop_remainder=True)
    ds_cluster = ds.shuffle(X.shape[0]).repeat().batch(train_batch, drop_remainder=True)
    ds_test = ds.batch(test_batch, drop_remainder=True)
    if augment:
        def make_generator():
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(**augmentation)
            gen = data_gen.flow(X, batch_size=train_batch)
            while True:
                batch = next(gen)
                while batch.shape[0]!=train_batch:
                    batch = next(gen)
                yield batch
        ds_pretrain = tf.data.Dataset.from_generator(make_generator, output_types=tf.float32,
                                                     output_shapes=(train_batch, *X.shape[1:]))
    if not conv:
        ds_pretrain = ds_pretrain.map(lambda batch: tf.reshape(batch, [batch.shape[0], tf.math.reduce_prod(batch.shape[1:])]))
        ds_cluster = ds_cluster.map(lambda batch: tf.reshape(batch, [batch.shape[0], tf.math.reduce_prod(batch.shape[1:])]))
        ds_test = ds_test.map(lambda batch: tf.reshape(batch, [batch.shape[0], tf.math.reduce_prod(batch.shape[1:])]))
    ds_pretrain = ds_pretrain.map(lambda batch: (batch, batch))
    return ds_pretrain, ds_cluster, ds_test, y

def load_reuters10(train_batch, test_batch):
    data = np.load("/tf/reuters/reutersidf10k.npy", allow_pickle=True).item()
    X = data["data"]
    y = data["label"]
    ds = tf.data.Dataset.from_tensor_slices((X, ))
    ds_pretrain = ds.shuffle(X.shape[0]).repeat().batch(train_batch, drop_remainder=True)
    ds_pretrain = ds_pretrain.map(lambda batch: (batch, batch))
    ds_cluster = ds.shuffle(X.shape[0]).repeat().batch(train_batch, drop_remainder=True)
    ds_test = ds.batch(test_batch, drop_remainder=True)
    return ds_pretrain, ds_cluster, ds_test, y

def load_usps(train_batch, test_batch, augment=False, conv=False, augmentation=None):
    data = np.load("/tf/usps/usps.npy", allow_pickle=True).item()
    X = data["data"]
    y = data["label"]
    ds = tf.data.Dataset.from_tensor_slices((X, ))
    ds_pretrain = ds.shuffle(X.shape[0]).repeat().batch(train_batch, drop_remainder=True)
    ds_cluster = ds.shuffle(X.shape[0]).repeat().batch(train_batch, drop_remainder=True)
    ds_test = ds.batch(test_batch, drop_remainder=True)
    if augment:
        def make_generator():
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(**augmentation)
            gen = data_gen.flow(X, batch_size=train_batch)
            while True:
                batch = next(gen)
                while batch.shape[0]!=train_batch:
                    batch = next(gen)
                yield batch
        ds_pretrain = tf.data.Dataset.from_generator(make_generator, output_types=tf.float32,
                                                     output_shapes=(train_batch, *X.shape[1:]))
    if not conv:
        ds_pretrain = ds_pretrain.map(lambda batch: tf.reshape(batch, [batch.shape[0], tf.math.reduce_prod(batch.shape[1:])]))
        ds_cluster = ds_cluster.map(lambda batch: tf.reshape(batch, [batch.shape[0], tf.math.reduce_prod(batch.shape[1:])]))
        ds_test = ds_test.map(lambda batch: tf.reshape(batch, [batch.shape[0], tf.math.reduce_prod(batch.shape[1:])]))
    ds_pretrain = ds_pretrain.map(lambda batch: (batch, batch))
    return ds_pretrain, ds_cluster, ds_test, y