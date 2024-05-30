import keras.initializers
from packaging import version
from Utility import MathUtility as M
from ESN.layer import *
from ESN.Inizializer import *
import math

tf_version = version.parse(tf.__version__)
if tf_version < version.parse("2.8.0rc0"):
    from tensorflow.keras.layers import Layer as BaseRandomLayer
elif tf_version < version.parse("2.13.0rc0"):
    from keras.engine.base_layer import BaseRandomLayer
elif tf_version < version.parse("2.16.0rc0"):
    from keras.src.engine.base_layer import BaseRandomLayer
else:
    from keras.layers import Layer as BaseRandomLayer


@tf.keras.utils.register_keras_serializable("keras-lrmu")
class LRMUCell(keras.layers.Layer):
    def __init__(self, memoryDimension, order, theta,
                 hiddenUnit=0, spectraRadius=0.99,
                 reservoirMode=True, hiddenCell=None,
                 memoryToMemory=False, hiddenToMemory=False, inputToHiddenCell=False,
                 useBias=False, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.MemoryDim = memoryDimension
        self.Order = order
        self.Theta = theta
        self.SpectraRadius = spectraRadius
        self.MemoryToMemory = memoryToMemory
        self.HiddenToMemory = hiddenToMemory
        self.InputToHiddenCell = inputToHiddenCell
        self.UseBias = useBias
        self.HiddenCell = hiddenCell
        self.ReservoirMode = reservoirMode
        self.Seed = seed

        self.RecurrentMemoryKernel = None
        self.MemoryKernel = None
        self.HiddenUnit = hiddenUnit

        self.Bias = None

        self.A = None
        self.B = None

        if self.HiddenCell is None and self.HiddenUnit != 0:
            self.HiddenCell = ReservoirCell(self.HiddenUnit, spectral_radius=self.SpectraRadius)

        if self.HiddenCell is None:
            if self.HiddenToMemory:
                raise ValueError(
                    "hidden_to_memory must be False if hidden_cell is None"
                )
            self.HiddenOutputSize = self.MemoryDim * self.Order
            self.HiddenStateSize = []
        elif hasattr(self.HiddenCell, "state_size"):
            self.HiddenOutputSize = self.HiddenCell.output_size
            self.HiddenStateSize = self.HiddenCell.state_size
        else:
            # TODO: support layers that don't have the `units` attribute
            self.HiddenOutputSize = self.HiddenCell.units
            self.HiddenStateSize = [self.HiddenCell.units]

        self.state_size = [self.MemoryDim * self.Order] + tf.nest.flatten(self.HiddenStateSize)
        self.output_size = self.HiddenOutputSize

    def createWeight(self, shape, inizializer="glorot_uniform"):
        if self.ReservoirMode:
            initializer = keras.initializers.GlorotUniform(seed=self.Seed)
            return self.add_weight(shape=shape, initializer=initializer, trainable=False)
        else:
            return self.add_weight(shape=shape, initializer=inizializer)

    def build(self, input_shape):
        tf.random.set_seed(self.Seed)
        super().build(input_shape)

        inputDim = input_shape[-1]
        outDim = self.HiddenOutputSize

        kernel_dim = inputDim + outDim if self.HiddenToMemory else inputDim

        self.MemoryKernel = self.createWeight(shape=(kernel_dim, self.MemoryDim))

        if self.MemoryToMemory:
            self.RecurrentMemoryKernel = self.createWeight(shape=(self.Order * self.MemoryDim, self.MemoryDim))

        if self.UseBias:
            self.Bias = self.createWeight(shape=(self.MemoryDim,))

        if self.HiddenCell is not None and not self.HiddenCell.built:
            hidden_input_d = self.MemoryDim * self.Order
            if self.InputToHiddenCell:
                hidden_input_d += input_shape[-1]
            with tf.name_scope(self.HiddenCell.name):
                self.HiddenCell.build((input_shape[0], hidden_input_d), )

        self.GenerateABMatrix()

    def GenerateABMatrix(self):
        """Generates A and B matrices."""
        # compute analog A/B matrices
        Q = np.arange(self.Order, dtype=np.float64)
        R = (2 * Q + 1)[:, None]
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R

        # discretize matrices with ZOH method
        # save the un-discretized matrices for use in .call
        self._base_A = tf.constant(A.T, dtype=self.dtype)
        self._base_B = tf.constant(B.T, dtype=self.dtype)

        self.A, self.B = M._cont2discrete_zoh(
            self._base_A / self.Theta, self._base_B / self.Theta
        )

    def call(self, inputs, states, training=False):
        states_fat = tf.nest.flatten(states)
        memory_state = states_fat[0]
        hidden_state = states_fat[1:]
        # get Previous hidden/Memory State

        concat_input = inputs
        if self.HiddenToMemory:
            concat_input = tf.concat((concat_input, hidden_state[0]), axis=1)

        # Compute the new Memory State
        u = tf.matmul(concat_input, self.MemoryKernel)
        if self.MemoryToMemory:
            u += tf.matmul(memory_state, self.RecurrentMemoryKernel)
        if self.UseBias:
            u += self.Bias

        u = tf.expand_dims(u, -1)
        memory_state = tf.reshape(memory_state, (-1, self.MemoryDim, self.Order))

        new_memory_state = tf.matmul(memory_state, self.A) + tf.matmul(u, self.B)
        new_memory_state = tf.reshape(new_memory_state, (-1, self.MemoryDim * self.Order))
        # Compute the new Hidden State

        hidden_input = (
            new_memory_state if not self.InputToHiddenCell else tf.concat((new_memory_state, inputs), axis=1))

        if self.HiddenCell is None:
            output = hidden_input
            new_hidden_state = []
        elif hasattr(self.HiddenCell, "state_size"):  # the hidden cell is an RNN cell
            output, new_hidden_state = self.HiddenCell(hidden_input, hidden_state, training=training)
        else:
            output = self.HiddenCell(hidden_input, training=training)
            new_hidden_state = [output]

        return output, [new_memory_state] + new_hidden_state

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "memoryDimension": self.MemoryDim,
                "order": self.Order,
                "theta": self.Theta,
                "hiddenUnit": self.HiddenUnit,
                "spectraRadius": self.SpectraRadius,
                "reservoirMode": self.ReservoirMode,
                "hiddenCell": keras.layers.serialize(self.HiddenCell),
                "memoryToMemory": self.MemoryToMemory,
                "hiddenToMemory": self.HiddenToMemory,
                "inputToHiddenCell": self.InputToHiddenCell,
                "useBias": self.UseBias,
                "seed": self.Seed})

        return config

    @classmethod
    def from_config(cls, config):
        """Load model from serialized config."""

        config["hiddenCell"] = (
            None
            if config["hiddenCell"] is None
            else keras.layers.deserialize(config["hiddenCell"])
        )
        return super().from_config(config)


@tf.keras.utils.register_keras_serializable("keras-lrmu")
class LRMU(keras.layers.Layer):

    def __init__(self, memoryDimension, order, theta, hiddenUnit=0, spectraRadius=0.99,
                 reservoirMode=True, hiddenCell=None,
                 memoryToMemory=False, hiddenToMemory=False, inputToHiddenCell=False,
                 useBias=False, seed=0, returnSequences=False, **kwargs):
        super().__init__(**kwargs)
        self.MemoryDim = memoryDimension
        self.Order = order
        self.Theta = theta
        self.HiddenUnit = hiddenUnit
        self.SpectraRadius = spectraRadius
        self.MemoryToMemory = memoryToMemory
        self.HiddenToMemory = hiddenToMemory
        self.InputToHiddenCell = inputToHiddenCell
        self.UseBias = useBias
        self.HiddenCell = hiddenCell
        self.ReservoirMode = reservoirMode
        self.Seed = seed
        self.ReturnSequence = returnSequences

        self.layer = None

    def build(self, input_shape):
        super().build(input_shape)

        self.layer = keras.layers.RNN(
            LRMUCell(self.MemoryDim, self.Order, self.Theta, self.HiddenUnit, self.SpectraRadius,
                     self.ReservoirMode, self.HiddenCell,
                     self.MemoryToMemory, self.HiddenToMemory, self.InputToHiddenCell,
                     self.UseBias, self.Seed), return_sequences=self.ReturnSequence)

        self.layer.build(input_shape)

    def call(self, inputs, training=False):
        return self.layer.call(inputs, training=training)

    def get_config(self):
        """Return config of layer (for serialization during model saving/loading)."""

        config = super().get_config()
        config.update({
            "memoryDimension": self.MemoryDim,
            "order": self.Order,
            "theta": self.Theta,
            "hiddenUnit": self.HiddenUnit,
            "spectraRadius": self.SpectraRadius,
            "reservoirMode": self.ReservoirMode,
            "hiddenCell": keras.layers.serialize(self.HiddenCell),
            "memoryToMemory": self.MemoryToMemory,
            "hiddenToMemory": self.HiddenToMemory,
            "inputToHiddenCell": self.InputToHiddenCell,
            "useBias": self.UseBias,
            "seed": self.Seed,
            "returnSequences": self.ReturnSequence})

        return config

    @classmethod
    def from_config(cls, config):
        """Load model from serialized config."""

        config["hiddenCell"] = (
            None
            if config["hiddenCell"] is None
            else keras.layers.deserialize(config["hiddenCell"])
        )
        return super().from_config(config)
