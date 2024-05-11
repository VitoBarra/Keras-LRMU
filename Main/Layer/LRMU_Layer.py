import warnings

import keras
import numpy as np
import tensorflow as tf
from packaging import version

tf_version = version.parse(tf.__version__)
if tf_version < version.parse("2.8.0rc0"):
    from tensorflow.keras.layers import Layer as BaseRandomLayer
elif tf_version < version.parse("2.13.0rc0"):
    from keras.engine.base_layer import BaseRandomLayer
elif tf_version < version.parse("2.16.0rc0"):
    from keras.src.engine.base_layer import BaseRandomLayer
else:
    from keras.layers import Layer as BaseRandomLayer




class LRMU(Main.Layer.keras_lmu.LMU):

    def __init__(self,
                 memory_d,
                 order,
                 theta,
                 hidden_cell,
                 trainable_theta=False,
                 hidden_to_memory=False,
                 memory_to_memory=False,
                 input_to_hidden=False,
                 discretizer="zoh",
                 kernel_initializer="glorot_uniform",
                 recurrent_initializer="orthogonal",
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 use_bias=False,
                 bias_initializer="zeros",
                 bias_regularizer=None,
                 dropout=0,
                 recurrent_dropout=0,
                 seed=None,
                 **kwargs):
        super(LRMU, self).__init__(memory_d=memory_d,
                                   order=order,
                                   theta=theta,
                                   hidden_cell=hidden_cell,
                                   trainable_theta=trainable_theta,
                                   hidden_to_memory=hidden_to_memory,
                                   memory_to_memory=memory_to_memory,
                                   input_to_hidden=input_to_hidden,
                                   discretizer=discretizer,
                                   kernel_initializer=kernel_initializer,
                                   recurrent_initializer=recurrent_initializer,
                                   kernel_regularizer=kernel_regularizer,
                                   recurrent_regularizer=recurrent_regularizer,
                                   use_bias=use_bias,
                                   bias_initializer=bias_initializer,
                                   bias_regularizer=bias_regularizer,
                                   dropout=dropout,
                                   recurrent_dropout=recurrent_dropout,
                                   seed=seed,
                                   **kwargs)

    def _gen_AB(self, x, h_tm1):
        # Generate the A and B matrices
        A = self._gen_A(x)
        B = self._gen_B(x, h_tm1)
        return A, B



