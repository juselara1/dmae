import os

import tensorflow as tf
import numpy as np 
import h5py

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _serialize_example(x, y):
    feature = {
            "x": _float_feature(x),
            "y": _int64_feature(y)
            }
    example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def save_record(
        shard_len, X, y,
        save_path
        ):
    for i in range(X.shape[0]):
        with tf.io.TFRecordWriter(
                os.path.join(
                    save_path, f"shard{i//shard_len}.tfrecords"
                    )
                ) as writer:
            example = _serialize_example(X[i].flatten(), int(y[i]))
            writer.write(example)

if __name__ == '__main__':
    with h5py.File("data/data.h5", "r") as df:
        X = df["fashion"][:]
        y = df["fashion_labels"][:]

    if not os.path.exists("fashion"):
        os.makedirs("fashion")

    save_record(
            1000, X, y,
            "fashion"
            )


