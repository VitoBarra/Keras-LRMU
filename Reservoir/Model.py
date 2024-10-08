import tensorflow.keras as keras
from Reservoir.layer import ReservoirCell
from sklearn.linear_model import RidgeClassifier
import tensorflow as tf
import numpy as np


class ESN(keras.Model):

    # Implements an Echo State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with ReservoirCell,
    # followed by a trainable dense readout layer for classification
    def __init__(self, units,
                 input_scaling=1., bias_scaling=1.0, spectral_radius=0.9,
                 leaky=1,
                 readout_regularizer=1.0,
                 activation=tf.nn.tanh,
                 features_dim=1,
                 batch_size=None,
                 **kwargs):

        super().__init__(**kwargs)
        if batch_size is not None:
            self.reservoir = keras.Sequential([

                keras.Input(batch_input_shape=(batch_size, None, features_dim)),

                keras.layers.RNN(cell=ReservoirCell(units=units,
                                                    input_scaling=input_scaling,
                                                    bias_scaling=bias_scaling,
                                                    spectral_radius=spectral_radius,
                                                    leaky=leaky, activation=activation),
                                 stateful=True)
            ])

        else:

            self.reservoir = keras.Sequential([
                keras.layers.RNN(cell=ReservoirCell(units=units,
                                                    input_scaling=input_scaling,
                                                    bias_scaling=bias_scaling,
                                                    spectral_radius=spectral_radius,
                                                    leaky=leaky, activation=activation))])

        self.readout = RidgeClassifier(alpha=readout_regularizer, solver='svd')
        self.units = units
        self.features_dim = features_dim
        self.batch_size = batch_size

    def compute_output(self, inputs):
        # calculate the reservoir states and the corresponding output of the model
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)

        return output

    def call(self, inputs):

        # create a numpy version of the input, which has an explicit first dimension given by num_samples

        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states)
        return output

    def fit(self, x, y, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method

        x_train_states = self.reservoir(x)
        self.readout.fit(x_train_states, y)

    def evaluate(self, x, y):
        x_train_states = self.reservoir(x)
        return self.readout.score(x_train_states, y)

    def evaluate_batch(self, x, y):

        # memory_reservoir_states = self.memory(x)
        # concatenated_input = np.concatenate((memory_reservoir_states,x), axis = -1)
        # perform the state computation in the reservoir in batches, to avoid memory issues
        # use a bacth size of batch_size for the reservoir computation

        batch_size = self.batch_size
        num_batches = int(np.ceil(x.shape[0] / batch_size))
        states_all = np.zeros(shape=(x.shape[0], self.units))

        for i in range(num_batches):
            xlocal = x[i * batch_size:(i + 1) * batch_size, :, :]
            self.reservoir.reset_states()
            original_shape = xlocal.shape
            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                x_states = self.reservoir(xlocal[:, t:t + 1, :])

            # x_states = self.reservoir(xlocal)

            states_all[i * batch_size:(i + 1) * batch_size, :] = x_states[:original_shape[0], :]

        return self.readout.score(states_all, y)

    def compute_output_batch(self, inputs):

        # calculate the reservoir states and the corresponding output of the model
        # memory_reservoir_states = self.memory(inputs)
        # concatenated_input = np.concatenate((self.memory(inputs),inputs), axis = -1)
        # perform the state computation in the reservoir in batches, to avoid memory issues
        # use a bacth size of batch_size for the reservoir computation

        batch_size = self.batch_size
        num_batches = int(np.ceil(inputs.shape[0] / batch_size))
        states_all = np.zeros(shape=(inputs.shape[0], self.units))

        for i in range(num_batches):

            self.reservoir.reset_states()
            xlocal = inputs[i * batch_size:(i + 1) * batch_size, :, :]
            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                x_states = self.reservoir(xlocal[:, t:t + 1, :])

            # memory_reservoir_states = self.memory(xlocal)
            # concatenated_input = np.concatenate((memory_reservoir_states,xlocal), axis = -1)
            # x_states = self.reservoir(xlocal)

            states_all[i * batch_size:(i + 1) * batch_size, :] = x_states[:original_shape[0], :]

        output = self.readout.predict(states_all)

        return output

    def fit_batch(self, x, y, **kwargs):

        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method
        # perform the state computation in the reservoir in batches, to avoid memory issues
        # use a bacth size of batch_size for the reservoir computation

        batch_size = self.batch_size
        num_batches = int(np.ceil(x.shape[0] / batch_size))
        x_train_states_all = np.zeros(shape=(x.shape[0], self.units))

        for i in range(num_batches):

            self.reservoir.reset_states()
            xlocal = x[i * batch_size:(i + 1) * batch_size, :, :]
            # memory_reservoir_states = self.memory(xlocal)
            # concatenated_input = np.concatenate((memory_reservoir_states,xlocal), axis = -1)

            original_shape = xlocal.shape
            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                x_states = self.reservoir(xlocal[:, t:t + 1, :])

            # x_train_states = self.reservoir(xlocal)

            x_train_states_all[i * batch_size:(i + 1) * batch_size, :] = x_states[:original_shape[0], :]

        self.readout.fit(x_train_states_all, y)

