import keras
import tensorflow as tf
import numpy as np
from Main.Layer.ESN.InizializationUtility import *


class ReservoirCell(keras.layers.Layer):

    # builds a reservoir as a hidden dynamical layer for a recurrent neural network
    def __init__(self, units,
                 input_scaling=1.0, bias_scaling=1.0,
                 spectral_radius=0.99,
                 leaky=1, activation=tf.nn.tanh,
                 **kwargs):

        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky  # leaking rate
        self.activation = activation
        self.output_size = units

        super().__init__(**kwargs)

    def build(self, input_shape):

        self.recurrent_kernel = BuildRecurentWeight(self.spectral_radius, self.units)
        # build the input weight matrix
        self.kernel = tf.random.uniform(shape=(input_shape[-1], self.units), minval=-self.input_scaling,
                                        maxval=self.input_scaling)

        # initialize the bias
        self.bias = tf.random.uniform(shape=(self.units,), minval=-self.bias_scaling, maxval=self.bias_scaling)
        self.built = True

    def call(self, inputs, states):

        prev_output = states[0]

        input_part = tf.matmul(inputs, self.kernel)
        state_part = tf.matmul(prev_output, self.recurrent_kernel)

        if self.activation != None:
            output = prev_output * (1 - self.leaky) + self.activation(input_part + self.bias + state_part) * self.leaky
        else:
            output = prev_output * (1 - self.leaky) + (input_part + self.bias + state_part) * self.leaky

        return output, [output]
