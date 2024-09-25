from tensorflow.keras.initializers import *
import tensorflow.keras as keras
from Reservoir.Inizializer import *
import tensorflow as tf


@tf.keras.utils.register_keras_serializable("keras-ReservoirCell")
class ReservoirCell(keras.layers.Layer):

    # builds a reservoir as a hidden dynamical layer for a recurrent neural network
    def __init__(self, units,
                 input_scaling=1.0, useBias=True, bias_scaling=1.0,
                 spectral_radius=0.99,
                 leaky=1, activation=tf.nn.tanh, seed=0,
                 **kwargs):
        self.built = None
        self.bias = None
        self.kernel = None
        self.recurrent_kernel = None
        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.usebias = useBias
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
                                                initializer=FastReccurrentSR(self.spectral_radius, self.units,
                                                                             self.seed),
                                                trainable=False)

        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer=RandomUniform(-self.input_scaling, self.input_scaling,
                                                                seed=self.seed),
                                      trainable=False)

        if self.usebias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=RandomUniform(-self.bias_scaling, self.bias_scaling,
                                                                  seed=self.seed),
                                        trainable=False)
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]

        input_part = tf.matmul(inputs, self.kernel)
        state_part = tf.matmul(prev_output, self.recurrent_kernel)

        value = input_part + state_part
        if self.usebias:
            value += self.bias

        output = (1 - self.leaky) * prev_output
        output += self.leaky * (self.activation(value) if self.activation is not None else value)

        return output, [output]

    def get_config(self):
        """Return config of layer (for serialization during model saving/loading)."""

        config = super().get_config()
        config.update({
            "units": self.units,
            "input_scaling": self.input_scaling,
            "bias_scaling": self.bias_scaling,
            "spectral_radius": self.spectral_radius,
            "leaky": self.leaky,
            "activation": self.activation.__name__,
            "seed": self.seed})

        return config

    @classmethod
    def from_config(cls, config):
        config["activation"] = tf.keras.activations.deserialize(config["activation"])
        return super().from_config(config)
