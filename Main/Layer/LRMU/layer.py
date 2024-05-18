import warnings

import keras
import numpy as np
import tensorflow as tf
from packaging import version
from Main.Util import MathUtility as M

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
    def __init__(self, memory_d, order, theta, hiddenCell=None, memoryToMemory=False, hiddenToMemory=False,
                 inputToCell=False, useBias=False, **kwargs):
        super().__init__()
        self.MemoryDim = memory_d
        self.Order = order
        self._init_theta = theta
        self.MemoryToMemory = memoryToMemory
        self.HiddenToMemory = hiddenToMemory
        self.InputToHiddenCell = inputToCell
        self.UseBias = useBias

        self.MemoryEncoder = None
        self.HiddenEncoder = None
        self.InputEncoder = None
        self.HiddenCell = hiddenCell

        self.Bias = None

        self.A = None
        self.B = None

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

    def build(self, input_shape):
        super().build(input_shape)

        inputDim = input_shape[-1]
        self.InputEncoder = self.add_weight(
            shape=(inputDim, self.MemoryDim),
            initializer="glorot_uniform",
            name="InputEncoder",
        )

        if self.MemoryToMemory:
            self.MemoryEncoder = self.add_weight(
                shape=(self.Order * self.MemoryDim, self.MemoryDim),
                initializer="glorot_uniform",
                name="MemoryEncoder",
            )

        OutDim = self.HiddenOutputSize

        if self.HiddenToMemory:
            self.HiddenEncoder = self.add_weight(
                shape=(OutDim, self.MemoryDim),
                initializer="glorot_uniform",
                name="HiddenEncoder",
            )

        if self.UseBias:
            self.Bias = self.add_weight(
                shape=(self.MemoryDim,),
                initializer="zeros",
                name="Bias",
            )

        if self.HiddenCell is not None and not self.HiddenCell.built:
            hidden_input_d = self.MemoryDim * self.Order
            if self.InputToHiddenCell:
                hidden_input_d += input_shape[-1]
            with tf.name_scope(self.HiddenCell.name):
                self.HiddenCell.build((input_shape[0], hidden_input_d) , )

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
            new_memory_state if not self.InputToHiddenCell else tf.concat((new_memory_state, inputs), axis=0))

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
