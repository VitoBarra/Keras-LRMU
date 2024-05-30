from keras.initializers import *
from ESN.Inizializer import *


class ReservoirCell(keras.layers.Layer):

    # builds a reservoir as a hidden dynamical layer for a recurrent neural network
    def __init__(self, units,
                 input_scaling=1.0, bias_scaling=1.0,
                 spectral_radius=0.99,
                 leaky=1, activation=tf.nn.tanh, seed=0,
                 **kwargs):

        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky  # leaking rate
        self.activation = activation
        self.output_size = units
        self.seed = seed

        super().__init__(**kwargs)

    def build(self, input_shape):

        tf.random.set_seed(self.seed)
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units),
                                                initializer=FastReccurrentSR(self.spectral_radius, self.units, self.seed),
                                                trainable=False)

        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer=RandomUniform(-self.input_scaling, self.input_scaling,
                                                                seed=self.seed),
                                      trainable=False)

        # initialize the bias
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer=RandomUniform(-self.bias_scaling, self.bias_scaling, seed=self.seed),
                                    trainable=False)
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
