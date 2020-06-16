"""
Implementation of: Dissimilarity Mixture Autoencoder (DMAE) for Deep Clustering.

**This package contains a tf.keras.Callback that is a variation of the EarlyStopping that measures the changes in the assigned clusters.**

Author: Juan Sebastián Lara Ramírez <julara@unal.edu.co> <https://github.com/larajuse>
"""

import tensorflow as tf
import numpy as np

class DeltaUACC(tf.keras.callbacks.Callback):
    """
    A tf.keras callback that evaluates differences between assigned clusters. It evaluates the proportion of clusters that change between different iterations.
    
    Arguments:
        encoder_model: tf.keras.Model
            Model that is used to get the assignments.
        ds: tf.data.Dataset
            Dataset from which the assignments are computed.
        N: int
            Number of samples in the dataset.
        tol: float, default: 1e-3
            Minimum admisible tolerance for the propotion
        verbose: bool, default: False
            Specify if the tolerance must be printed.
        batch_size: int, default: 1000
            Batch size that is used to compute the paiwise dissimilarities.
        interval: int, default: 10
            Interval (in batches) to compute the differences.
    """
    
    def __init__(self, encoder_model, ds, N, tol=1e-3, verbose=False, batch_size=1000, interval=10):
        self.__encoder_model = encoder_model
        self.__ds = ds
        self.__N = N
        self.__tol = tol
        self.__verbose = verbose
        self.__batch_size = batch_size
        self.__interval = interval
        self.__batch_cont = 0
        self.__prev_y = np.zeros((N, ))
    
    def on_train_batch_end(self, batch, logs=None):
        """Evaluates the proportion on certain batches"""
        if self.__batch_cont%self.__interval==0:
            self.__batch_cont += 1
            self.__encoder_model.layers[-1].set_weights(self.model.layers[2].get_weights())
            preds = self.__encoder_model.predict(self.__ds, steps=self.__N//self.__batch_size)
            y_pred = np.argmax(preds, axis=1)
            delta = np.float32(np.sum(y_pred!=self.__prev_y))/self.__N
            if delta < self.__tol:
                if self.__verbose: print(f"Achieved a tolerance of {delta}")
                self.model.stop_training = True
            else:
                if self.__verbose: print(f"Current tolerance: {delta}")
                self.__prev_y = y_pred
        else:
            self.__batch_cont += 1
            
            
class callbacks():
    def __init__(self):
        self.DeltaUACC = DeltaUACC