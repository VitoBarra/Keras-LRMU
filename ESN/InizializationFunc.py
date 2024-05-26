import numpy as np
import tensorflow as tf


def BuildRecurentWeight(spectral_radius, units):
    # build the recurrent weight matrix
    # uses circular law to determine the values of the recurrent weight matrix
    # rif. paper
    # Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli.
    # "Fast spectral radius initialization for recurrent neural networks."
    # INNS Big Data and Deep Learning conference. Springer, Cham, 2019.
    value = (spectral_radius / np.sqrt(units)) * (6 / np.sqrt(12))
    W = tf.random.uniform(shape=(units, units), minval=-value, maxval=value)
    return W
