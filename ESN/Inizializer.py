import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class FastReccurrentSR(keras.initializers.Initializer):

    def __init__(self, spectral_radius, units,seed=None):
        self.spectral_radius = spectral_radius
        self.units = units
        self.seed = seed

    def __call__(self, shape, dtype=None):
    # build the recurrent weight matrix
    # uses circular law to determine the values of the recurrent weight matrix
    # rif. paper
    # Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli.
    # "Fast spectral radius initialization for recurrent neural networks."
    # INNS Big Data and Deep Learning conference. Springer, Cham, 2019.

        assert shape[0] == shape[1]
        tf.random.set_seed(self.seed)
        value = (self.spectral_radius / np.sqrt(self.units)) * (6 / np.sqrt(12))
        W = tf.random.uniform(shape=shape, minval=-value, maxval=value)
        return W
