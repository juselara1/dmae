import os, argparse

import tensorflow as tf
import numpy as np 
import h5py

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.datasets import imdb

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
    shard = 0
    idx = 0
    while idx != y.shape[0]:
        print(f"shard: {shard}", end="\r")
        with tf.io.TFRecordWriter(
            os.path.join(
                save_path,
                f"shard{shard}.tfrecord"
                )
            ) as writer:
            for i in range(shard_len):
                example = _serialize_example(X[idx].flatten(), int(y[idx]))
                writer.write(example)
                idx += 1

        shard += 1

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = imdb.load_data()
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    
    tokenizer = Tokenizer(num_words=2000)
    tokenizer.fit_on_sequences(X)
    X = tokenizer.sequences_to_matrix(X, mode='tfidf')

    if not os.path.exists("data/imdb"):
        os.makedirs("data/imdb")
    save_record(
            1000, X, y,
            "data/imdb"
            )


