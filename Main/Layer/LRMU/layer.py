import warnings

import keras
import numpy as np
import tensorflow as tf
from packaging import version
from Main.Util import MathUtility as M
from Main.Layer.ESN.layer import *
from Main.Layer.ESN.InizializationUtility import *
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
    def __init__(self, memoryDimension, order, theta, hiddenUnit=0, spectraRadius=0.99,
                 reservoirMode=True, hiddenCell=None, memoryToMemory=False, hiddenToMemory=False,
                 inputToCell=False, useBias=False, seed=0, **kwargs):
        super().__init__()
        self.MemoryDim = memoryDimension
        self.Order = order
        self._init_theta = theta
        self.SpectraRadius = spectraRadius
        self.MemoryToMemory = memoryToMemory
        self.HiddenToMemory = hiddenToMemory
        self.InputToHiddenCell = inputToCell
        self.UseBias = useBias
        self.HiddenCell = hiddenCell
        self.ReservoirMode = reservoirMode
        self.Seed = seed

        self.MemoryEncoder = None
        self.HiddenEncoder = None
        self.InputEncoder = None
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
            value = 1
            if len(shape) > 1:
                value = math.sqrt(6 / (shape[0] + shape[1]))
            return tf.random.uniform(shape=shape, minval=-value, maxval=value)
        else:
            return self.add_weight(shape=shape, initializer=inizializer)

    def build(self, input_shape):
        tf.random.set_seed(self.Seed)
        super().build(input_shape)

        inputDim = input_shape[-1]
        self.InputEncoder = self.createWeight(shape=(inputDim, self.MemoryDim))

        if self.MemoryToMemory:
            self.MemoryEncoder = self.createWeight(shape=(self.Order * self.MemoryDim, self.MemoryDim))

        OutDim = self.HiddenOutputSize

        if self.HiddenToMemory:
            self.HiddenEncoder = self.createWeight(shape=(OutDim, self.MemoryDim))

        if self.UseBias:
            self.Bias = self.createWeight(shape=(self.MemoryDim,))

        if self.HiddenCell is not None and not self.HiddenCell.built:
            hidden_input_d = self.MemoryDim * self.Order
            if self.InputToHiddenCell:
                hidden_input_d += input_shape[-1]
            with tf.name_scope(self.HiddenCell.name):
                self.HiddenCell.build((input_shape[0], hidden_input_d), )

        self._gen_AB()

    def call(self, inputs, states, training=False):

        # get Previous hidden/Memory State
        states = tf.nest.flatten(states)
        memory_state = states[0]
        hidden_state = states[1:]

        # Compute the new Memory State
        u = tf.matmul(inputs, self.InputEncoder)
        if self.MemoryToMemory:
            u += tf.matmul(memory_state, self.MemoryEncoder)
        if self.HiddenToMemory:
            u += tf.matmul(hidden_state[0], self.HiddenEncoder)
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

    def _gen_AB(self):
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
            self._base_A / self._init_theta, self._base_B / self._init_theta
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "memory_d": self.MemoryDim,
                "order": self.Order,
                "theta": self._init_theta,
                "hidden_cell": keras.layers.serialize(self.HiddenCell),
                "hidden_to_memory": self.HiddenToMemory,
                "memory_to_memory": self.MemoryToMemory,
                "input_to_hidden": self.InputToHiddenCell,
                "use_bias": self.UseBias,
                "seed": self.Seed,
            }
        )

        return config


@tf.keras.utils.register_keras_serializable("keras-lrmu")
class LRMU(keras.layers.Layer):

    def __init__(self, memoryDimension, order, theta, hiddenUnit=0, spectraRadius=0.99,
                 reservoirMode=True, hiddenCell=None, memoryToMemory=False, hiddenToMemory=False,
                 inputToCell=False, useBias=False, seed=0, returnSequences = False, **kwargs):
        super().__init__()
        self.MemoryDim = memoryDimension
        self.Order = order
        self._init_theta = theta
        self.HiddenUnit = hiddenUnit
        self.SpectraRadius = spectraRadius
        self.MemoryToMemory = memoryToMemory
        self.HiddenToMemory = hiddenToMemory
        self.InputToHiddenCell = inputToCell
        self.UseBias = useBias
        self.HiddenCell = hiddenCell
        self.ReservoirMode = reservoirMode
        self.Seed = seed
        self.ReturnSequence = returnSequences

        self.layer = None

    def build(self, input_shape):
        super().build(input_shape)

        self.layer = keras.layers.RNN(
            LRMUCell(self.MemoryDim, self.Order, self._init_theta, self.HiddenUnit, self.SpectraRadius,
                     self.ReservoirMode, self.HiddenCell,
                     self.MemoryToMemory, self.HiddenToMemory, self.InputToHiddenCell,
                     self.UseBias, self.Seed), return_sequences=self.ReturnSequence)

        self.layer.build(input_shape)

    def call(self, inputs, training=False):
        return self.layer.call(inputs, training=training)

    def get_config(self):
        """Return config of layer (for serialization during model saving/loading)."""

        config = super().get_config()
        config.update(
            {
                "memory_d": self.MemoryDim,
                "order": self.Order,
                "theta": self._init_theta,
                "hidden_cell": keras.layers.serialize(self.HiddenCell),
                "hidden_to_memory": self.HiddenToMemory,
                "memory_to_memory": self.MemoryToMemory,
                "input_to_hidden": self.InputToHiddenCell,
                "use_bias": self.UseBias,
                "return_sequences": self.ReturnSequence,
                "seed": self.Seed
            }
        )

        return config