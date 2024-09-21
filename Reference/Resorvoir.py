import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
# from legendre import *
# from tqdm import tqdm
# import StandardScaler

from sklearn.preprocessing import StandardScaler


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


class ESN_readoutCV(keras.Model):

    # Implements an Echo State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with ReservoirCell,
    # followed by a trainable dense readout layer for classification
    # The readout layer is trained using a RidgeClassifierCV

    def __init__(self, units,
                 input_scaling=1., bias_scaling=1.0, spectral_radius=0.9,
                 leaky=1,
                 # readout_regularizer = 1.0,
                 activation=tf.nn.tanh,
                 features_dim=1,
                 batch_size=None,
                 # cv = None, lambda_min = 1e-5, lambda_max = 1e1, num_lambdas = 7,
                 cv=None, lambda_min=1e-5, lambda_max=1e4, num_lambdas=10,
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

                                                    leaky=leaky, activation=activation))

            ])

        # self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

        self.readout = RidgeClassifierCV(alphas=np.logspace(lambda_min, lambda_max, num_lambdas), cv=cv)

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

        # use a bacth size of of batch_size for the reservoir computation

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

        # use a bacth size of of batch_size for the reservoir computation

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


class EulerReservoirCell(keras.layers.Layer):
    # Implements the reservoir layer of the Euler State Network
    # - the state transition function is achieved by Euler discretization of an ODE
    # - the recurrent weight matrix is constrained to have an anti-symmetric (i.e., skew-symmetric) structure

    def __init__(self, units,

                 input_scaling=1., bias_scaling=1.0, recurrent_scaling=1,

                 epsilon=0.01, gamma=0.001,

                 activation=tf.nn.tanh,

                 **kwargs):

        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.recurrent_scaling = recurrent_scaling
        self.bias_scaling = bias_scaling
        self.epsilon = epsilon
        self.gamma = gamma
        self.activation = activation

        super().__init__(**kwargs)

    def build(self, input_shape):

        # build the recurrent weight matrix

        I = tf.linalg.eye(self.units)
        W = tf.random.uniform(shape=(self.units, self.units), minval=-self.recurrent_scaling,
                              maxval=self.recurrent_scaling)

        self.recurrent_kernel = (W - tf.transpose(W) - self.gamma * I)
        # build the input weight matrix
        self.kernel = tf.random.uniform(shape=(input_shape[-1], self.units), minval=-self.input_scaling,
                                        maxval=self.input_scaling)
        # bias vector
        self.bias = tf.random.uniform(shape=(self.units,), minval=-self.bias_scaling, maxval=self.bias_scaling)
        self.built = True

    def call(self, inputs, states):

        prev_output = states[0]

        input_part = tf.matmul(inputs, self.kernel)

        state_part = tf.matmul(prev_output, self.recurrent_kernel)

        if self.activation != None:

            output = prev_output + self.epsilon * self.activation(input_part + self.bias + state_part)

        else:

            output = prev_output + self.epsilon * (input_part + self.bias + state_part)

        return output, [output]


class EuSN(keras.Model):

    # Implements an Euler State Network model for time-series classification problems

    #

    # The architecture comprises a recurrent layer with EulerReservoirCell,

    # followed by a trainable dense readout layer for classification

    def __init__(self, units,

                 input_scaling=1., bias_scaling=1.0, recurrent_scaling=1,

                 epsilon=0.01, gamma=0.001,

                 readout_regularizer=1.0,

                 activation=tf.nn.tanh,

                 features_dim=1,

                 batch_size=None,

                 **kwargs):

        super().__init__(**kwargs)

        if batch_size is not None:

            self.reservoir = keras.Sequential([

                keras.Input(batch_input_shape=(batch_size, None, features_dim)),

                keras.layers.RNN(cell=EulerReservoirCell(units=units,

                                                         input_scaling=input_scaling,

                                                         bias_scaling=bias_scaling,

                                                         recurrent_scaling=recurrent_scaling,

                                                         epsilon=epsilon,

                                                         gamma=gamma), stateful=True)

            ])

        else:

            self.reservoir = keras.Sequential([

                keras.layers.RNN(cell=EulerReservoirCell(units=units,

                                                         input_scaling=input_scaling,

                                                         bias_scaling=bias_scaling,

                                                         recurrent_scaling=recurrent_scaling,

                                                         epsilon=epsilon,

                                                         gamma=gamma)),

            ])

        self.readout = RidgeClassifier(alpha=readout_regularizer, solver='svd')

        self.units = units

        self.batch_size = batch_size

    def call(self, inputs):

        reservoir_states = self.reservoir(inputs)

        output = self.readout.predict(reservoir_states)

        return output

    def compute_output(self, inputs):

        # calculate the reservoir states and the corresponding output of the model

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

        # use a bacth size of of batch_size for the reservoir computation

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

        # use a bacth size of of batch_size for the reservoir computation

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

        # use a bacth size of of batch_size for the reservoir computation

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


class EuSN_readoutCV(keras.Model):

    # Implements an Euler State Network model for time-series classification problems

    #

    # The architecture comprises a recurrent layer with EulerReservoirCell,

    # followed by a trainable dense readout layer for classification

    # The readout layer is trained using a RidgeClassifierCV

    def __init__(self, units,

                 input_scaling=1., bias_scaling=1.0, recurrent_scaling=1,

                 epsilon=0.01, gamma=0.001,

                 # readout_regularizer = 1.0,

                 activation=tf.nn.tanh,

                 features_dim=1,

                 batch_size=None,

                 # cv = None, lambda_min = 1e-5, lambda_max = 1e1, num_lambdas = 7,

                 cv=None, lambda_min=1e-5, lambda_max=1e4, num_lambdas=10,

                 **kwargs):

        super().__init__(**kwargs)

        if batch_size is not None:

            self.reservoir = keras.Sequential([

                keras.Input(batch_input_shape=(batch_size, None, features_dim)),

                keras.layers.RNN(cell=EulerReservoirCell(units=units,

                                                         input_scaling=input_scaling,

                                                         bias_scaling=bias_scaling,

                                                         recurrent_scaling=recurrent_scaling,

                                                         epsilon=epsilon,

                                                         gamma=gamma), stateful=True)

            ])

        else:

            self.reservoir = keras.Sequential([

                keras.layers.RNN(cell=EulerReservoirCell(units=units,

                                                         input_scaling=input_scaling,

                                                         bias_scaling=bias_scaling,

                                                         recurrent_scaling=recurrent_scaling,

                                                         epsilon=epsilon,

                                                         gamma=gamma)),

            ])

        self.readout = RidgeClassifierCV(alphas=np.logspace(lambda_min, lambda_max, num_lambdas), cv=cv)

        # (alpha = readout_regularizer, solver = 'svd')

        self.units = units

        self.batch_size = batch_size

    def call(self, inputs):

        reservoir_states = self.reservoir(inputs)

        output = self.readout.predict(reservoir_states)

        return output

    def compute_output(self, inputs):

        # calculate the reservoir states and the corresponding output of the model

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

        # use a bacth size of of batch_size for the reservoir computation

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

        # use a bacth size of of batch_size for the reservoir computation

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

        # use a bacth size of of batch_size for the reservoir computation

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


class ReservoirCell_2(keras.layers.Layer):

    # builds a reservoir as a hidden dynamical layer for a recurrent neural network

    def __init__(self, units,

                 memory_dim=100,

                 input_scaling=1.0, bias_scaling=1.0,

                 memory_scaling=1.0, spectral_radius=0.99,

                 leaky=1, activation=tf.nn.tanh,

                 **kwargs):

        self.units = units  # number of units in the reservoir

        self.memory_d = memory_dim  # this identifies the first components of the input that enters

        # this layer from the output of the memory layer

        self.input_d = 0  # this is the dimension of the external input

        self.state_size = units  # this is the size of the state of the layer

        self.input_scaling = input_scaling  # scaling factor for the input weights

        self.memory_scaling = memory_scaling  # scaling factor for the memory weights

        self.bias_scaling = bias_scaling  # scaling factor for the bias

        self.spectral_radius = spectral_radius  # spectral radius of the recurrent weight matrix

        self.leaky = leaky  # leaking rate

        self.activation = activation

        super().__init__(**kwargs)

    def build(self, input_shape):

        # build the recurrent weight matrix

        # uses circular law to determine the values of the recurrent weight matrix

        # rif. paper

        # Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli.

        # "Fast spectral radius initialization for recurrent neural networks."

        # INNS Big Data and Deep Learning conference. Springer, Cham, 2019.

        value = (self.spectral_radius / np.sqrt(self.units)) * (6 / np.sqrt(12))

        W = tf.random.uniform(shape=(self.units, self.units), minval=-value, maxval=value)

        self.recurrent_kernel = W

        # here it is needed to separate the two inputs that enters this layer: memory + external input

        total_input_d = input_shape[-1]  # this is the total dimension of the input

        self.input_d = total_input_d - self.memory_d  # this is the dimension of the external input

        # build the input weight matrix

        self.kernel = tf.random.uniform(shape=(self.input_d, self.units), minval=-self.input_scaling,
                                        maxval=self.input_scaling)

        # build the memory weight matrix

        self.kernel_memory = tf.random.uniform(shape=(self.memory_d, self.units), minval=-self.memory_scaling,
                                               maxval=self.memory_scaling)

        # initialize the bias

        self.bias = tf.random.uniform(shape=(self.units,), minval=-self.bias_scaling, maxval=self.bias_scaling)

        self.built = True

    def call(self, inputs, states):

        prev_output = states[0]

        # now separate the inputs into activations from the memory layer (the first self.memory_d values)

        # and activations from the external input (the last self.input_d values)

        memory_inputs = inputs[:, :self.memory_d]  # this is the input from the memory layer

        external_inputs = inputs[:, self.memory_d:]  # this is the input from the external input

        input_part1 = tf.matmul(external_inputs, self.kernel)  # this is the input from the external input

        input_part2 = tf.matmul(memory_inputs, self.kernel_memory)  # this is the input from the memory layer

        input_part = input_part1 + input_part2  # this is the total input

        state_part = tf.matmul(prev_output, self.recurrent_kernel)  # this is the state part

        if self.activation != None:

            output = prev_output * (1 - self.leaky) + self.activation(input_part + self.bias + state_part) * self.leaky

        else:

            output = prev_output * (1 - self.leaky) + (input_part + self.bias + state_part) * self.leaky

        return output, [output]


class EulerReservoirCell_2(keras.layers.Layer):

    # builds an euler reservoir as a hidden dynamical layer for a recurrent neural network

    def __init__(self, units,

                 memory_dim=100,

                 input_scaling=1.0, bias_scaling=1.0,

                 memory_scaling=1.0, recurrent_scaling=1.0,

                 epsilon=0.01, gamma=0.001,

                 activation=tf.nn.tanh,

                 **kwargs):

        self.units = units  # number of units in the reservoir

        self.memory_d = memory_dim  # this identifies the first components of the input that enters

        # this layer from the output of the memory layer

        self.input_d = 0  # this is the dimension of the external input

        self.state_size = units  # this is the size of the state of the layer

        self.input_scaling = input_scaling  # scaling factor for the input weights

        self.memory_scaling = memory_scaling  # scaling factor for the memory weights

        self.bias_scaling = bias_scaling  # scaling factor for the bias

        self.recurrent_scaling = recurrent_scaling  # scaling factor for the recurrent weights

        self.epsilon = epsilon  # step size for the Euler discretization

        self.gamma = gamma  # regularization parameter for the skew-symmetric matrix

        self.activation = activation

        super().__init__(**kwargs)

    def build(self, input_shape):

        # build the recurrent weight matrix

        # build the recurrent weight matrix

        I = tf.linalg.eye(self.units)

        W = tf.random.uniform(shape=(self.units, self.units), minval=-self.recurrent_scaling,
                              maxval=self.recurrent_scaling)

        self.recurrent_kernel = (W - tf.transpose(W) - self.gamma * I)

        # here it is needed to separate the two inputs that enters this layer: memory + external input

        total_input_d = input_shape[-1]  # this is the total dimension of the input

        self.input_d = total_input_d - self.memory_d  # this is the dimension of the external input

        # build the input weight matrix

        self.kernel = tf.random.uniform(shape=(self.input_d, self.units), minval=-self.input_scaling,
                                        maxval=self.input_scaling)

        # build the memory weight matrix

        self.kernel_memory = tf.random.uniform(shape=(self.memory_d, self.units), minval=-self.memory_scaling,
                                               maxval=self.memory_scaling)

        # initialize the bias

        self.bias = tf.random.uniform(shape=(self.units,), minval=-self.bias_scaling, maxval=self.bias_scaling)

        self.built = True

    def call(self, inputs, states):

        prev_output = states[0]

        # now separate the inputs into activations from the memory layer (the first self.memory_d values)

        # and activations from the external input (the last self.input_d values)

        memory_inputs = inputs[:, :self.memory_d]  # this is the input from the memory layer

        external_inputs = inputs[:, self.memory_d:]  # this is the input from the external input

        input_part1 = tf.matmul(external_inputs, self.kernel)  # this is the input from the external input

        input_part2 = tf.matmul(memory_inputs, self.kernel_memory)  # this is the input from the memory layer

        input_part = input_part1 + input_part2  # this is the total input

        state_part = tf.matmul(prev_output, self.recurrent_kernel)  # this is the state part

        if self.activation != None:

            output = prev_output + self.epsilon * self.activation(input_part + self.bias + state_part)

        else:

            output = prev_output + self.epsilon * (input_part + self.bias + state_part)

        return output, [output]


class RingReservoirCell(keras.layers.Layer):

    # builds a ring reservoir as a hidden dynamical layer for a recurrent neural network

    # differently from a conventional reservoir layer, in this case the units in the recurrent

    # layer are organized to form a cycle (i.e., a ring)

    def __init__(self, units,

                 input_scaling=1.0, bias_scaling=1.0,

                 spectral_radius=0.99,

                 leaky=1., activation=tf.nn.tanh,

                 **kwargs):

        self.units = units

        self.state_size = units

        self.input_scaling = input_scaling

        self.bias_scaling = bias_scaling

        self.spectral_radius = spectral_radius

        self.leaky = leaky

        self.activation = activation

        super().__init__(**kwargs)

    def build(self, input_shape):

        # build the recurrent weight matrix

        I = tf.linalg.eye(self.units)

        W = self.spectral_radius * tf.concat([I[:, -1:], I[:, 0:-1]], axis=1)

        self.recurrent_kernel = W

        # variant for a moment

        # self.orthogonal,_ = np.linalg.qr(2*np.random.rand(self.units,self.units)-1)

        # self.recurrent_kernel = self.spectral_radius * self.orthogonal #tf.constant(self.orthogonal, dtype = tf.float32)

        # build the input weight matrix

        self.kernel = tf.random.uniform(shape=(input_shape[-1], self.units), minval=-self.input_scaling,
                                        maxval=self.input_scaling)

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


class LegendreReservoirCell(keras.layers.Layer):

    # builds a reservoir as a hidden dynamical layer for a recurrent neural network

    # using a Legendre matrix as the recurrent weight matrix

    def __init__(self, units,

                 input_scaling=1.0, bias_scaling=1.0,

                 theta=1000,

                 leaky=1., activation=tf.nn.tanh,

                 **kwargs):

        self.units = units

        self.state_size = units

        self.input_scaling = input_scaling

        self.bias_scaling = bias_scaling

        self.theta = theta

        self.leaky = leaky

        self.activation = activation

        super().__init__(**kwargs)

    def build(self, input_shape):

        # build the recurrent weight matrix

        A, _ = generate_legendre_matrix(order=self.units, theta=self.theta)

        self.recurrent_kernel = A

        # build the input weight matrix

        self.kernel = tf.random.uniform(shape=(input_shape[-1], self.units), minval=-self.input_scaling,
                                        maxval=self.input_scaling)

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


class RMN(keras.Model):  # Reservoir Memory Network

    # Implements a Reservoir Memory Network for time-series classification problems

    def __init__(self,

                 units,  # number of units in the non-linear reservoir

                 units_m=100,  # number of units in the linear memory component

                 input_scaling=1., bias_scaling=1.0, spectral_radius=0.9, memory_scaling=1.0, leaky=1.0,
                 # parameters of the nonlinear reservoir

                 input_scaling_m=1., bias_scaling_m=0.0, spectral_radius_m=1.0,
                 # parameters of the linear reservoir

                 leaky_m=1.0,

                 readout_regularizer=1.0,

                 activation=tf.nn.tanh,

                 features_dim=1,

                 batch_size=None,

                 **kwargs):

        super().__init__(**kwargs)

        if batch_size is not None:

            self.memory = keras.Sequential([

                # linear memory component

                keras.Input(batch_input_shape=(batch_size, None, features_dim)),

                keras.layers.RNN(cell=RingReservoirCell(units=units_m,

                                                        input_scaling=input_scaling_m,

                                                        bias_scaling=bias_scaling_m,

                                                        spectral_radius=spectral_radius_m,

                                                        leaky=leaky_m,

                                                        activation=None),  # linear layer

                                 return_sequences=True, stateful=True)

            ])

            self.reservoir = keras.Sequential([

                # nonlinear reservoir

                # properly specify the batch input shape

                keras.Input(batch_input_shape=(batch_size, None, units_m + features_dim)),

                keras.layers.RNN(cell=ReservoirCell_2(units=units,

                                                      memory_dim=units_m,

                                                      input_scaling=input_scaling,

                                                      bias_scaling=bias_scaling,

                                                      memory_scaling=memory_scaling,

                                                      spectral_radius=spectral_radius,

                                                      leaky=leaky,

                                                      activation=activation),

                                 stateful=True)

            ])




        else:

            self.memory = keras.Sequential([

                # linear memory component

                keras.layers.RNN(cell=RingReservoirCell(units=units_m,

                                                        input_scaling=input_scaling_m,

                                                        bias_scaling=bias_scaling_m,

                                                        spectral_radius=spectral_radius_m,

                                                        leaky=leaky_m,

                                                        activation=None),  # linear layer

                                 return_sequences=True)

            ])

            self.reservoir = keras.Sequential([

                # nonlinear reservoir

                # properly specify the batch input shape

                keras.layers.RNN(cell=ReservoirCell_2(units=units,

                                                      memory_dim=units_m,

                                                      input_scaling=input_scaling,

                                                      bias_scaling=bias_scaling,

                                                      memory_scaling=memory_scaling,

                                                      spectral_radius=spectral_radius,

                                                      leaky=leaky,

                                                      activation=activation))

            ])

        self.readout = RidgeClassifier(alpha=readout_regularizer, solver='svd')

        self.units = units

        self.features_dim = features_dim

        self.batch_size = batch_size

    def call(self, inputs):

        # memory_reservoir_states = self.memory(inputs)

        # concatenated_input = np.concatenate((self.memory(inputs),inputs), axis = -1)

        reservoir_states = self.reservoir(np.concatenate((self.memory(inputs), inputs), axis=-1))

        output = self.readout.predict(reservoir_states)

        return output

    def compute_output(self, inputs):

        # calculate the reservoir states and the corresponding output of the model

        # memory_reservoir_states = self.memory(inputs)

        # concatenated_input = np.concatenate((self.memory(inputs),inputs), axis = -1)

        reservoir_states = self.reservoir(np.concatenate((self.memory(inputs), inputs), axis=-1))

        output = self.readout.predict(reservoir_states)

        return output

    def fit(self, x, y, **kwargs):

        # For all the RC methods, we avoid doing the same reservoir operations at each epoch

        # To this aim, we pre-compute all the states and then we invoke the readout fit method

        # x_train_states = self.reservoir(x)

        # memory_reservoir_states = self.memory(x)

        # concatenated_input = np.concatenate((self.memory(x),x), axis = -1)

        x_train_states = self.reservoir(np.concatenate((self.memory(x), x), axis=-1))

        self.readout.fit(x_train_states, y)

    def evaluate(self, x, y):

        # memory_reservoir_states = self.memory(x)

        # concatenated_input = np.concatenate((memory_reservoir_states,x), axis = -1)

        x_train_states = self.reservoir(np.concatenate((self.memory(x), x), axis=-1))

        return self.readout.score(x_train_states, y)

    def evaluate_batch(self, x, y):

        # memory_reservoir_states = self.memory(x)

        # concatenated_input = np.concatenate((memory_reservoir_states,x), axis = -1)

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        batch_size = self.batch_size

        num_batches = int(np.ceil(x.shape[0] / batch_size))

        states_all = np.zeros(shape=(x.shape[0], self.units))

        for i in range(num_batches):

            # print('batch * ', i)

            # reset the state of self.memory recurrent layer

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = x[i * batch_size:(i + 1) * batch_size, :, :]

            # if xlocal is smaller tahn the batch_size we need to pad it

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_states = self.reservoir(concatenated_input)

            # now update the states_all variable only with the non-padded values

            states_all[i * batch_size:(i + 1) * batch_size, :] = x_states[:original_shape[0], :]

        return self.readout.score(states_all, y)

    def compute_output_batch(self, inputs):

        # calculate the reservoir states and the corresponding output of the model

        # memory_reservoir_states = self.memory(inputs)

        # concatenated_input = np.concatenate((self.memory(inputs),inputs), axis = -1)

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        batch_size = self.batch_size

        num_batches = int(np.ceil(inputs.shape[0] / batch_size))

        states_all = np.zeros(shape=(inputs.shape[0], self.units))

        for i in range(num_batches):

            # print('batch', i)

            # reset the state of self.memory recurrent layer

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = inputs[i * batch_size:(i + 1) * batch_size, :, :]

            # if xlocal is smaller tahn the batch_size we need to pad it

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_states = self.reservoir(concatenated_input)

            # states_all[i*batch_size:(i+1)*batch_size,:] = x_states

            states_all[i * batch_size:(i + 1) * batch_size, :] = x_states[:original_shape[0], :]

        output = self.readout.predict(states_all)

        return output

    def fit_batch(self, x, y, **kwargs):

        # For all the RC methods, we avoid doing the same reservoir operations at each epoch

        # To this aim, we pre-compute all the states and then we invoke the readout fit method

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        batch_size = self.batch_size

        num_batches = int(np.ceil(x.shape[0] / batch_size))

        x_train_states_all = np.zeros(shape=(x.shape[0], self.units))

        for i in range(num_batches):

            # reset the state of self.memory recurrent layer

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = x[i * batch_size:(i + 1) * batch_size, :, :]

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_train_states = self.reservoir(concatenated_input)

            x_train_states_all[i * batch_size:(i + 1) * batch_size, :] = x_train_states[:original_shape[0], :]

        self.readout.fit(x_train_states_all, y)


class RMN_readoutCV(keras.Model):  # Reservoir Memory Network

    # Implements a Reservoir Memory Network for time-series classification problems

    def __init__(self,

                 units,  # number of units in the non-linear reservoir

                 units_m=100,  # number of units in the linear memory component

                 input_scaling=1., bias_scaling=1.0, spectral_radius=0.9, memory_scaling=1.0, leaky=1.0,
                 # parameters of the nonlinear reservoir

                 input_scaling_m=1., bias_scaling_m=0.0, spectral_radius_m=1.0,
                 # parameters of the linear reservoir

                 leaky_m=1.0,

                 # readout_regularizer = 1.0,

                 activation=tf.nn.tanh,

                 features_dim=1,

                 batch_size=None,

                 # cv = None, lambda_min = 1e-5, lambda_max = 1e1, num_lambdas = 7,

                 cv=None, lambda_min=1e-5, lambda_max=1e4, num_lambdas=10,

                 **kwargs):

        super().__init__(**kwargs)

        if batch_size is not None:

            self.memory = keras.Sequential([

                # linear memory component

                keras.Input(batch_input_shape=(batch_size, None, features_dim)),

                keras.layers.RNN(cell=RingReservoirCell(units=units_m,

                                                        input_scaling=input_scaling_m,

                                                        bias_scaling=bias_scaling_m,

                                                        spectral_radius=spectral_radius_m,

                                                        leaky=leaky_m,

                                                        activation=None),  # linear layer

                                 return_sequences=True, stateful=True)

            ])

            self.reservoir = keras.Sequential([

                # nonlinear reservoir

                # properly specify the batch input shape

                keras.Input(batch_input_shape=(batch_size, None, units_m + features_dim)),

                keras.layers.RNN(cell=ReservoirCell_2(units=units,

                                                      memory_dim=units_m,

                                                      input_scaling=input_scaling,

                                                      bias_scaling=bias_scaling,

                                                      memory_scaling=memory_scaling,

                                                      spectral_radius=spectral_radius,

                                                      leaky=leaky,

                                                      activation=activation),

                                 stateful=True)

            ])




        else:

            self.memory = keras.Sequential([

                # linear memory component

                keras.layers.RNN(cell=RingReservoirCell(units=units_m,

                                                        input_scaling=input_scaling_m,

                                                        bias_scaling=bias_scaling_m,

                                                        spectral_radius=spectral_radius_m,

                                                        leaky=leaky_m,

                                                        activation=None),  # linear layer

                                 return_sequences=True)

            ])

            self.reservoir = keras.Sequential([

                # nonlinear reservoir

                # properly specify the batch input shape

                keras.layers.RNN(cell=ReservoirCell_2(units=units,

                                                      memory_dim=units_m,

                                                      input_scaling=input_scaling,

                                                      bias_scaling=bias_scaling,

                                                      memory_scaling=memory_scaling,

                                                      spectral_radius=spectral_radius,

                                                      leaky=leaky,

                                                      activation=activation))

            ])

        self.readout = RidgeClassifierCV(alphas=np.logspace(lambda_min, lambda_max, num_lambdas), cv=cv)

        # self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

        self.units = units

        self.features_dim = features_dim

        self.batch_size = batch_size

        # self.scaler = StandardScaler()

    def call(self, inputs):

        # memory_reservoir_states = self.memory(inputs)

        # concatenated_input = np.concatenate((self.memory(inputs),inputs), axis = -1)

        reservoir_states = self.reservoir(np.concatenate((self.memory(inputs), inputs), axis=-1))

        output = self.readout.predict(reservoir_states)

        return output

    def compute_output(self, inputs):

        # calculate the reservoir states and the corresponding output of the model

        # memory_reservoir_states = self.memory(inputs)

        # concatenated_input = np.concatenate((self.memory(inputs),inputs), axis = -1)

        reservoir_states = self.reservoir(np.concatenate((self.memory(inputs), inputs), axis=-1))

        output = self.readout.predict(reservoir_states)

        return output

    def fit(self, x, y, **kwargs):

        # For all the RC methods, we avoid doing the same reservoir operations at each epoch

        # To this aim, we pre-compute all the states and then we invoke the readout fit method

        # x_train_states = self.reservoir(x)

        # memory_reservoir_states = self.memory(x)

        # concatenated_input = np.concatenate((self.memory(x),x), axis = -1)

        x_train_states = self.reservoir(np.concatenate((self.memory(x), x), axis=-1))

        self.readout.fit(x_train_states, y)

    def evaluate(self, x, y):

        # memory_reservoir_states = self.memory(x)

        # concatenated_input = np.concatenate((memory_reservoir_states,x), axis = -1)

        x_train_states = self.reservoir(np.concatenate((self.memory(x), x), axis=-1))

        return self.readout.score(x_train_states, y)

    def evaluate_batch(self, x, y):

        # memory_reservoir_states = self.memory(x)

        # concatenated_input = np.concatenate((memory_reservoir_states,x), axis = -1)

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        batch_size = self.batch_size

        num_batches = int(np.ceil(x.shape[0] / batch_size))

        states_all = np.zeros(shape=(x.shape[0], self.units))

        for i in range(num_batches):

            # print('batch * ', i)

            # reset the state of self.memory recurrent layer

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = x[i * batch_size:(i + 1) * batch_size, :, :]

            # if xlocal is smaller tahn the batch_size we need to pad it

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_states = self.reservoir(concatenated_input)

            # now update the states_all variable only with the non-padded values

            states_all[i * batch_size:(i + 1) * batch_size, :] = x_states[:original_shape[0], :]

        # apply the scaler

        # states_all = self.scaler.transform(states_all)

        return self.readout.score(states_all, y)

    def compute_output_batch(self, inputs):

        # calculate the reservoir states and the corresponding output of the model

        # memory_reservoir_states = self.memory(inputs)

        # concatenated_input = np.concatenate((self.memory(inputs),inputs), axis = -1)

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        batch_size = self.batch_size

        num_batches = int(np.ceil(inputs.shape[0] / batch_size))

        states_all = np.zeros(shape=(inputs.shape[0], self.units))

        for i in range(num_batches):

            # print('batch', i)

            # reset the state of self.memory recurrent layer

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = inputs[i * batch_size:(i + 1) * batch_size, :, :]

            # if xlocal is smaller tahn the batch_size we need to pad it

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_states = self.reservoir(concatenated_input)

            # states_all[i*batch_size:(i+1)*batch_size,:] = x_states

            states_all[i * batch_size:(i + 1) * batch_size, :] = x_states[:original_shape[0], :]

        # apply the scaler before the output computation

        # states_all = self.scaler.transform(states_all)

        output = self.readout.predict(states_all)

        return output

    def fit_batch(self, x, y, **kwargs):

        # For all the RC methods, we avoid doing the same reservoir operations at each epoch

        # To this aim, we pre-compute all the states and then we invoke the readout fit method

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        batch_size = self.batch_size

        num_batches = int(np.ceil(x.shape[0] / batch_size))

        x_train_states_all = np.zeros(shape=(x.shape[0], self.units))

        progress_bar = tqdm(total=num_batches, desc="Progress", unit="iteration")

        for i in range(num_batches):

            # visualize the progress over the batches

            progress_bar.update(1)

            # reset the state of self.memory recurrent layer

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = x[i * batch_size:(i + 1) * batch_size, :, :]

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_train_states = self.reservoir(concatenated_input)

            x_train_states_all[i * batch_size:(i + 1) * batch_size, :] = x_train_states[:original_shape[0], :]

        progress_bar.close()

        # normalize the states before training the readout

        # using the standard scaler

        # scaler = StandardScaler()

        # x_train_states_all = scaler.fit_transform(x_train_states_all)

        # self.scaler = scaler

        self.readout.fit(x_train_states_all, y)


class RMN_legendre(keras.Model):  # Reservoir Memory Network

    # Implements a Reservoir Memory Network for time-series classification problems

    # using a Legendre matrix as the recurrent weight matrix

    def __init__(self,

                 units,  # number of units in the non-linear reservoir

                 units_m=100,  # number of units in the linear memory component

                 input_scaling=1., bias_scaling=1.0, spectral_radius=0.9, memory_scaling=1.0, leaky=1.0,
                 # parameters of the nonlinear reservoir

                 input_scaling_m=1., bias_scaling_m=1.0, theta=1000,  # parameters of the memory reservoir

                 leaky_m=1.0,

                 readout_regularizer=1.0,

                 activation=tf.nn.tanh,

                 features_dim=1,

                 batch_size=None,

                 **kwargs):

        super().__init__(**kwargs)

        if batch_size is not None:

            self.memory = keras.Sequential([

                # linear memory component

                keras.Input(batch_input_shape=(batch_size, None, features_dim)),

                keras.layers.RNN(cell=LegendreReservoirCell(units=units_m,

                                                            input_scaling=input_scaling_m,

                                                            bias_scaling=bias_scaling_m,

                                                            theta=theta,

                                                            leaky=leaky_m,

                                                            activation=None),  # linear layer

                                 return_sequences=True, stateful=True)

            ])

            self.reservoir = keras.Sequential([

                # nonlinear reservoir

                # properly specify the batch input shape

                keras.Input(batch_input_shape=(batch_size, None, units_m + features_dim)),

                keras.layers.RNN(cell=ReservoirCell_2(units=units,

                                                      memory_dim=units_m,

                                                      input_scaling=input_scaling,

                                                      bias_scaling=bias_scaling,

                                                      memory_scaling=memory_scaling,

                                                      spectral_radius=spectral_radius,

                                                      leaky=leaky,

                                                      activation=activation),

                                 stateful=True)

            ])

        else:

            self.memory = keras.Sequential([

                # linear memory component

                keras.layers.RNN(cell=LegendreReservoirCell(units=units_m,

                                                            input_scaling=input_scaling_m,

                                                            bias_scaling=bias_scaling_m,

                                                            theta=theta,

                                                            leaky=leaky_m,

                                                            activation=None),  # linear layer

                                 return_sequences=True)

            ])

            self.reservoir = keras.Sequential([

                # nonlinear reservoir

                keras.layers.RNN(cell=ReservoirCell_2(units=units,

                                                      memory_dim=units_m,

                                                      input_scaling=input_scaling,

                                                      bias_scaling=bias_scaling,

                                                      memory_scaling=memory_scaling,

                                                      spectral_radius=spectral_radius,

                                                      leaky=leaky,

                                                      activation=activation))

            ])

        self.readout = RidgeClassifier(alpha=readout_regularizer, solver='svd')

        self.units = units

        self.features_dim = features_dim

        self.batch_size = batch_size

    def call(self, inputs):

        memory_reservoir_states = self.memory(inputs)

        concatenated_input = np.concatenate((memory_reservoir_states, inputs), axis=-1)

        reservoir_states = self.reservoir(concatenated_input)

        output = self.readout.predict(reservoir_states)

        return output

    def compute_output(self, inputs):

        # calculate the reservoir states and the corresponding output of the model

        memory_reservoir_states = self.memory(inputs)

        concatenated_input = np.concatenate((memory_reservoir_states, inputs), axis=-1)

        reservoir_states = self.reservoir(concatenated_input)

        output = self.readout.predict(reservoir_states)

        return output

    def fit(self, x, y, **kwargs):

        # For all the RC methods, we avoid doing the same reservoir operations at each epoch

        # To this aim, we pre-compute all the states and then we invoke the readout fit method

        # x_train_states = self.reservoir(x)

        memory_reservoir_states = self.memory(x)

        concatenated_input = np.concatenate((memory_reservoir_states, x), axis=-1)

        x_train_states = self.reservoir(concatenated_input)

        self.readout.fit(x_train_states, y)

    def evaluate(self, x, y):

        memory_reservoir_states = self.memory(x)

        concatenated_input = np.concatenate((memory_reservoir_states, x), axis=-1)

        x_train_states = self.reservoir(concatenated_input)

        return self.readout.score(x_train_states, y)

    def evaluate_batch(self, x, y):

        # memory_reservoir_states = self.memory(x)

        # concatenated_input = np.concatenate((memory_reservoir_states,x), axis = -1)

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        batch_size = self.batch_size

        num_batches = int(np.ceil(x.shape[0] / batch_size))

        states_all = np.zeros(shape=(x.shape[0], self.units))

        for i in range(num_batches):

            # print('batch * ', i)

            # reset the state of self.memory recurrent layer

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = x[i * batch_size:(i + 1) * batch_size, :, :]

            # if xlocal is smaller tahn the batch_size we need to pad it

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_states = self.reservoir(concatenated_input)

            # now update the states_all variable only with the non-padded values

            states_all[i * batch_size:(i + 1) * batch_size, :] = x_states[:original_shape[0], :]

        return self.readout.score(states_all, y)

    def compute_output_batch(self, inputs):

        # calculate the reservoir states and the corresponding output of the model

        # memory_reservoir_states = self.memory(inputs)

        # concatenated_input = np.concatenate((self.memory(inputs),inputs), axis = -1)

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        batch_size = self.batch_size

        num_batches = int(np.ceil(inputs.shape[0] / batch_size))

        states_all = np.zeros(shape=(inputs.shape[0], self.units))

        for i in range(num_batches):

            # print('batch', i)

            # reset the state of self.memory recurrent layer

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = inputs[i * batch_size:(i + 1) * batch_size, :, :]

            # if xlocal is smaller tahn the batch_size we need to pad it

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_states = self.reservoir(concatenated_input)

            # states_all[i*batch_size:(i+1)*batch_size,:] = x_states

            states_all[i * batch_size:(i + 1) * batch_size, :] = x_states[:original_shape[0], :]

        output = self.readout.predict(states_all)

        return output

    def fit_batch(self, x, y, **kwargs):

        # For all the RC methods, we avoid doing the same reservoir operations at each epoch

        # To this aim, we pre-compute all the states and then we invoke the readout fit method

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        batch_size = self.batch_size

        num_batches = int(np.ceil(x.shape[0] / batch_size))

        x_train_states_all = np.zeros(shape=(x.shape[0], self.units))

        for i in range(num_batches):

            # reset the state of self.memory recurrent layer

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = x[i * batch_size:(i + 1) * batch_size, :, :]

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_train_states = self.reservoir(concatenated_input)

            x_train_states_all[i * batch_size:(i + 1) * batch_size, :] = x_train_states[:original_shape[0], :]

        self.readout.fit(x_train_states_all, y)


class RMN_euler(keras.Model):

    # Implements a Reservoir Memory Network for time-series classification problems

    # using an Euler reservoir layer as the memory component

    def __init__(self,

                 units,  # number of units in the non-linear reservoir

                 units_m=100,  # number of units in the linear memory component

                 input_scaling=1., bias_scaling=1.0, spectral_radius=0.9, memory_scaling=1.0, leaky=1.0,
                 # parameters of the nonlinear reservoir

                 input_scaling_m=1., bias_scaling_m=1.0, recurrent_scaling_m=1.0,

                 epsilon=0.01, gamma=0.001,  # parameters of the linear reservoir

                 readout_regularizer=1.0,

                 activation=tf.nn.tanh,

                 **kwargs):

        super().__init__(**kwargs)

        self.memory = keras.Sequential([

            # memory component

            keras.layers.RNN(cell=EulerReservoirCell(units=units_m,

                                                     input_scaling=input_scaling_m,

                                                     bias_scaling=bias_scaling_m,

                                                     recurrent_scaling=recurrent_scaling_m,

                                                     gamma=gamma,

                                                     epsilon=epsilon),

                             return_sequences=True)

        ])

        self.reservoir = keras.Sequential([

            # nonlinear reservoir

            keras.layers.RNN(cell=ReservoirCell_2(units=units,

                                                  memory_dim=units_m,

                                                  input_scaling=input_scaling,

                                                  bias_scaling=bias_scaling,

                                                  memory_scaling=memory_scaling,

                                                  spectral_radius=spectral_radius,

                                                  leaky=leaky,

                                                  activation=activation))

        ])

        self.readout = RidgeClassifier(alpha=readout_regularizer, solver='svd')

        self.units = units

    def call(self, inputs):

        memory_reservoir_states = self.memory(inputs)

        concatenated_input = np.concatenate((memory_reservoir_states, inputs), axis=-1)

        reservoir_states = self.reservoir(concatenated_input)

        output = self.readout.predict(reservoir_states)

        return output

    def compute_output(self, inputs):

        # calculate the reservoir states and the corresponding output of the model

        memory_reservoir_states = self.memory(inputs)

        concatenated_input = np.concatenate((memory_reservoir_states, inputs), axis=-1)

        reservoir_states = self.reservoir(concatenated_input)

        output = self.readout.predict(reservoir_states)

        return output

    def fit(self, x, y, **kwargs):

        # For all the RC methods, we avoid doing the same reservoir operations at each epoch

        # To this aim, we pre-compute all the states and then we invoke the readout fit method

        memory_reservoir_states = self.memory(x)

        concatenated_input = np.concatenate((memory_reservoir_states, x), axis=-1)

        x_train_states = self.reservoir(concatenated_input)

        self.readout.fit(x_train_states, y)

    def evaluate(self, x, y):

        memory_reservoir_states = self.memory(x)

        concatenated_input = np.concatenate((memory_reservoir_states, x), axis=-1)

        x_train_states = self.reservoir(concatenated_input)

        return self.readout.score(x_train_states, y)

    def evaluate_batch(self, x, y, batch_size=128):

        # memory_reservoir_states = self.memory(x)

        # concatenated_input = np.concatenate((memory_reservoir_states,x), axis = -1)

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        num_batches = int(np.ceil(x.shape[0] / batch_size))

        states_all = np.zeros(shape=(x.shape[0], self.units))

        for i in range(num_batches):

            # print('batch * ', i)

            # reset the state of self.memory recurrent layer

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = x[i * batch_size:(i + 1) * batch_size, :, :]

            # if xlocal is smaller tahn the batch_size we need to pad it

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_states = self.reservoir(concatenated_input)

            # now update the states_all variable only with the non-padded values

            states_all[i * batch_size:(i + 1) * batch_size, :] = x_states[:original_shape[0], :]

        return self.readout.score(states_all, y)

    def compute_output_batch(self, inputs, batch_size=128):

        # calculate the reservoir states and the corresponding output of the model

        # memory_reservoir_states = self.memory(inputs)

        # concatenated_input = np.concatenate((self.memory(inputs),inputs), axis = -1)

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        num_batches = int(np.ceil(inputs.shape[0] / batch_size))

        states_all = np.zeros(shape=(inputs.shape[0], self.units))

        for i in range(num_batches):

            # print('batch', i)

            # reset the state of self.memory recurrent layer

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = inputs[i * batch_size:(i + 1) * batch_size, :, :]

            # if xlocal is smaller tahn the batch_size we need to pad it

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_states = self.reservoir(concatenated_input)

            # states_all[i*batch_size:(i+1)*batch_size,:] = x_states

            states_all[i * batch_size:(i + 1) * batch_size, :] = x_states[:original_shape[0], :]

        output = self.readout.predict(states_all)

        return output

    def fit_batch(self, x, y, batch_size=128, **kwargs):

        # For all the RC methods, we avoid doing the same reservoir operations at each epoch

        # To this aim, we pre-compute all the states and then we invoke the readout fit method

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        num_batches = int(np.ceil(x.shape[0] / batch_size))

        x_train_states_all = np.zeros(shape=(x.shape[0], self.units))

        for i in range(num_batches):

            # reset the state of self.memory recurrent layer

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = x[i * batch_size:(i + 1) * batch_size, :, :]

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_train_states = self.reservoir(concatenated_input)

            x_train_states_all[i * batch_size:(i + 1) * batch_size, :] = x_train_states[:original_shape[0], :]

        self.readout.fit(x_train_states_all, y)


class RMN_EuSN(keras.Model):

    # Implements a Reservoir Memory Network for time-series classification problems

    # using a linear orthogonal reservoir as the memeory component

    # and an Euler reservoir as the nonlinear component

    def __init__(self,

                 units,  # number of units in the non-linear reservoir

                 units_m=100,  # number of units in the linear memory component

                 input_scaling=1., bias_scaling=1.0, recurrent_scaling=0.9, memory_scaling=1.0,
                 # parameters of the nonlinear reservoir

                 epsilon=0.01, gamma=0.001,  # parameters of the euler component

                 input_scaling_m=1., bias_scaling_m=1.0, spectral_radius_m=1.0, leaky_m=1.0,
                 # parameters of the linear reservoir

                 readout_regularizer=1.0,

                 activation=tf.nn.tanh,

                 features_dim=1,

                 batch_size=None,

                 **kwargs):

        super().__init__(**kwargs)

        if batch_size is not None:

            self.memory = keras.Sequential([

                # linear memory component

                keras.Input(batch_input_shape=(batch_size, None, features_dim)),

                keras.layers.RNN(cell=RingReservoirCell(units=units_m,

                                                        input_scaling=input_scaling_m,

                                                        bias_scaling=bias_scaling_m,

                                                        spectral_radius=spectral_radius_m,

                                                        leaky=leaky_m,

                                                        activation=None),  # linear layer

                                 return_sequences=True, stateful=True)

            ])

            self.reservoir = keras.Sequential([

                # nonlinear reservoir

                keras.Input(batch_input_shape=(batch_size, None, units_m + features_dim)),

                keras.layers.RNN(cell=EulerReservoirCell_2(units=units,

                                                           memory_dim=units_m,

                                                           input_scaling=input_scaling,

                                                           bias_scaling=bias_scaling,

                                                           memory_scaling=memory_scaling,

                                                           recurrent_scaling=recurrent_scaling,

                                                           gamma=gamma,

                                                           epsilon=epsilon,

                                                           activation=activation),

                                 stateful=True)

            ])

        else:

            self.memory = keras.Sequential([

                # linear memory component

                keras.layers.RNN(cell=RingReservoirCell(units=units_m,

                                                        input_scaling=input_scaling_m,

                                                        bias_scaling=bias_scaling_m,

                                                        spectral_radius=spectral_radius_m,

                                                        leaky=leaky_m,

                                                        activation=None),  # linear layer

                                 return_sequences=True)

            ])

            self.reservoir = keras.Sequential([

                # nonlinear reservoir

                keras.layers.RNN(cell=EulerReservoirCell_2(units=units,

                                                           memory_dim=units_m,

                                                           input_scaling=input_scaling,

                                                           bias_scaling=bias_scaling,

                                                           memory_scaling=memory_scaling,

                                                           recurrent_scaling=recurrent_scaling,

                                                           gamma=gamma,

                                                           epsilon=epsilon,

                                                           activation=activation))

            ])

        self.readout = RidgeClassifier(alpha=readout_regularizer, solver='svd')

        self.units = units

        self.features_dim = features_dim

        self.batch_size = batch_size

    def call(self, inputs):

        memory_reservoir_states = self.memory(inputs)

        concatenated_input = np.concatenate((memory_reservoir_states, inputs), axis=-1)

        reservoir_states = self.reservoir(concatenated_input)

        output = self.readout.predict(reservoir_states)

        return output

    def compute_output(self, inputs):

        # calculate the reservoir states and the corresponding output of the model

        memory_reservoir_states = self.memory(inputs)

        concatenated_input = np.concatenate((memory_reservoir_states, inputs), axis=-1)

        reservoir_states = self.reservoir(concatenated_input)

        output = self.readout.predict(reservoir_states)

        return output

    def fit(self, x, y, **kwargs):

        # For all the RC methods, we avoid doing the same reservoir operations at each epoch

        # To this aim, we pre-compute all the states and then we invoke the readout fit method

        memory_reservoir_states = self.memory(x)

        concatenated_input = np.concatenate((memory_reservoir_states, x), axis=-1)

        x_train_states = self.reservoir(concatenated_input)

        self.readout.fit(x_train_states, y)

    def evaluate(self, x, y):

        memory_reservoir_states = self.memory(x)

        concatenated_input = np.concatenate((memory_reservoir_states, x), axis=-1)

        x_train_states = self.reservoir(concatenated_input)

        return self.readout.score(x_train_states, y)

    def evaluate_batch(self, x, y):

        # memory_reservoir_states = self.memory(x)

        # concatenated_input = np.concatenate((memory_reservoir_states,x), axis = -1)

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        batch_size = self.batch_size

        num_batches = int(np.ceil(x.shape[0] / batch_size))

        states_all = np.zeros(shape=(x.shape[0], self.units))

        progress_bar = tqdm(total=num_batches, desc="Progress", unit="iteration")

        for i in range(num_batches):

            # visualize a progress bar

            progress_bar.update(1)

            # print('batch * ', i)

            # reset the state of self.memory recurrent layer

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = x[i * batch_size:(i + 1) * batch_size, :, :]

            # if xlocal is smaller tahn the batch_size we need to pad it

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_states = self.reservoir(concatenated_input)

            # now update the states_all variable only with the non-padded values

            states_all[i * batch_size:(i + 1) * batch_size, :] = x_states[:original_shape[0], :]

        progress_bar.close()

        return self.readout.score(states_all, y)

    def compute_output_batch(self, inputs):

        # calculate the reservoir states and the corresponding output of the model

        # memory_reservoir_states = self.memory(inputs)

        # concatenated_input = np.concatenate((self.memory(inputs),inputs), axis = -1)

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        batch_size = self.batch_size

        num_batches = int(np.ceil(inputs.shape[0] / batch_size))

        states_all = np.zeros(shape=(inputs.shape[0], self.units))

        for i in range(num_batches):

            # print('batch', i)

            # reset the state of self.memory recurrent layer

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = inputs[i * batch_size:(i + 1) * batch_size, :, :]

            # if xlocal is smaller tahn the batch_size we need to pad it

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_states = self.reservoir(concatenated_input)

            # states_all[i*batch_size:(i+1)*batch_size,:] = x_states

            states_all[i * batch_size:(i + 1) * batch_size, :] = x_states[:original_shape[0], :]

        output = self.readout.predict(states_all)

        return output

    def fit_batch(self, x, y, **kwargs):

        # For all the RC methods, we avoid doing the same reservoir operations at each epoch

        # To this aim, we pre-compute all the states and then we invoke the readout fit method

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        batch_size = self.batch_size

        num_batches = int(np.ceil(x.shape[0] / batch_size))

        x_train_states_all = np.zeros(shape=(x.shape[0], self.units))

        progress_bar = tqdm(total=num_batches, desc="Progress", unit="iteration")

        for i in range(num_batches):

            # visualize the progress over the batches

            progress_bar.update(1)

            # reset the state of self.memory recurrent layer

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = x[i * batch_size:(i + 1) * batch_size, :, :]

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_train_states = self.reservoir(concatenated_input)

            x_train_states_all[i * batch_size:(i + 1) * batch_size, :] = x_train_states[:original_shape[0], :]

        progress_bar.close()

        self.readout.fit(x_train_states_all, y)

        print('model trained')


class RMN_EuSN_readoutCV(keras.Model):

    # Implements a Reservoir Memory Network for time-series classification problems

    # using a linear orthogonal reservoir as the memeory component

    # and an Euler reservoir as the nonlinear component

    # The readout is a RidgeClassifier with cross-validation

    def __init__(self,

                 units,  # number of units in the non-linear reservoir

                 units_m=100,  # number of units in the linear memory component

                 input_scaling=1., bias_scaling=1.0, recurrent_scaling=0.9, memory_scaling=1.0,
                 # parameters of the nonlinear reservoir

                 epsilon=0.01, gamma=0.001,  # parameters of the euler component

                 input_scaling_m=1., bias_scaling_m=1.0, spectral_radius_m=1.0, leaky_m=1.0,
                 # parameters of the linear reservoir

                 # readout_regularizer = 1.0,

                 activation=tf.nn.tanh,

                 features_dim=1,

                 batch_size=None,

                 # cv = None, lambda_min = 1e-5, lambda_max = 1e1, num_lambdas = 7,

                 cv=None, lambda_min=1e-5, lambda_max=1e4, num_lambdas=10,

                 **kwargs):

        super().__init__(**kwargs)

        if batch_size is not None:

            self.memory = keras.Sequential([

                # linear memory component

                keras.Input(batch_input_shape=(batch_size, None, features_dim)),

                keras.layers.RNN(cell=RingReservoirCell(units=units_m,

                                                        input_scaling=input_scaling_m,

                                                        bias_scaling=bias_scaling_m,

                                                        spectral_radius=spectral_radius_m,

                                                        leaky=leaky_m,

                                                        activation=None),  # linear layer

                                 return_sequences=True, stateful=True)

            ])

            self.reservoir = keras.Sequential([

                # nonlinear reservoir

                keras.Input(batch_input_shape=(batch_size, None, units_m + features_dim)),

                keras.layers.RNN(cell=EulerReservoirCell_2(units=units,

                                                           memory_dim=units_m,

                                                           input_scaling=input_scaling,

                                                           bias_scaling=bias_scaling,

                                                           memory_scaling=memory_scaling,

                                                           recurrent_scaling=recurrent_scaling,

                                                           gamma=gamma,

                                                           epsilon=epsilon,

                                                           activation=activation),

                                 stateful=True)

            ])

        else:

            self.memory = keras.Sequential([

                # linear memory component

                keras.layers.RNN(cell=RingReservoirCell(units=units_m,

                                                        input_scaling=input_scaling_m,

                                                        bias_scaling=bias_scaling_m,

                                                        spectral_radius=spectral_radius_m,

                                                        leaky=leaky_m,

                                                        activation=None),  # linear layer

                                 return_sequences=True)

            ])

            self.reservoir = keras.Sequential([

                # nonlinear reservoir

                keras.layers.RNN(cell=EulerReservoirCell_2(units=units,

                                                           memory_dim=units_m,

                                                           input_scaling=input_scaling,

                                                           bias_scaling=bias_scaling,

                                                           memory_scaling=memory_scaling,

                                                           recurrent_scaling=recurrent_scaling,

                                                           gamma=gamma,

                                                           epsilon=epsilon,

                                                           activation=activation))

            ])

        self.readout = RidgeClassifierCV(alphas=np.logspace(lambda_min, lambda_max, num_lambdas), cv=cv)

        # self.readout = RidgeClassifier(alpha = readout_regularizer, solver = 'svd')

        self.units = units

        self.features_dim = features_dim

        self.batch_size = batch_size

    def call(self, inputs):

        memory_reservoir_states = self.memory(inputs)

        concatenated_input = np.concatenate((memory_reservoir_states, inputs), axis=-1)

        reservoir_states = self.reservoir(concatenated_input)

        output = self.readout.predict(reservoir_states)

        return output

    def compute_output(self, inputs):

        # calculate the reservoir states and the corresponding output of the model

        memory_reservoir_states = self.memory(inputs)

        concatenated_input = np.concatenate((memory_reservoir_states, inputs), axis=-1)

        reservoir_states = self.reservoir(concatenated_input)

        output = self.readout.predict(reservoir_states)

        return output

    def fit(self, x, y, **kwargs):

        # For all the RC methods, we avoid doing the same reservoir operations at each epoch

        # To this aim, we pre-compute all the states and then we invoke the readout fit method

        memory_reservoir_states = self.memory(x)

        concatenated_input = np.concatenate((memory_reservoir_states, x), axis=-1)

        x_train_states = self.reservoir(concatenated_input)

        self.readout.fit(x_train_states, y)

    def evaluate(self, x, y):

        memory_reservoir_states = self.memory(x)

        concatenated_input = np.concatenate((memory_reservoir_states, x), axis=-1)

        x_train_states = self.reservoir(concatenated_input)

        return self.readout.score(x_train_states, y)

    def evaluate_batch(self, x, y):

        # memory_reservoir_states = self.memory(x)

        # concatenated_input = np.concatenate((memory_reservoir_states,x), axis = -1)

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        batch_size = self.batch_size

        num_batches = int(np.ceil(x.shape[0] / batch_size))

        states_all = np.zeros(shape=(x.shape[0], self.units))

        progress_bar = tqdm(total=num_batches, desc="Progress", unit="iteration")

        for i in range(num_batches):

            # visualize a progress bar

            progress_bar.update(1)

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = x[i * batch_size:(i + 1) * batch_size, :, :]

            # if xlocal is smaller tahn the batch_size we need to pad it

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_states = self.reservoir(concatenated_input)

            # now update the states_all variable only with the non-padded values

            states_all[i * batch_size:(i + 1) * batch_size, :] = x_states[:original_shape[0], :]

        progress_bar.close()

        return self.readout.score(states_all, y)

    def compute_output_batch(self, inputs):

        # calculate the reservoir states and the corresponding output of the model

        # memory_reservoir_states = self.memory(inputs)

        # concatenated_input = np.concatenate((self.memory(inputs),inputs), axis = -1)

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        batch_size = self.batch_size

        num_batches = int(np.ceil(inputs.shape[0] / batch_size))

        states_all = np.zeros(shape=(inputs.shape[0], self.units))

        for i in range(num_batches):

            # print('batch', i)

            # reset the state of self.memory recurrent layer

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = inputs[i * batch_size:(i + 1) * batch_size, :, :]

            # if xlocal is smaller tahn the batch_size we need to pad it

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_states = self.reservoir(concatenated_input)

            # states_all[i*batch_size:(i+1)*batch_size,:] = x_states

            states_all[i * batch_size:(i + 1) * batch_size, :] = x_states[:original_shape[0], :]

        output = self.readout.predict(states_all)

        return output

    def fit_batch(self, x, y, **kwargs):

        # For all the RC methods, we avoid doing the same reservoir operations at each epoch

        # To this aim, we pre-compute all the states and then we invoke the readout fit method

        # perform the state computation in the reservoir in batches, to avoid memory issues

        # use a bacth size of of batch_size for the reservoir computation

        batch_size = self.batch_size

        num_batches = int(np.ceil(x.shape[0] / batch_size))

        x_train_states_all = np.zeros(shape=(x.shape[0], self.units))

        progress_bar = tqdm(total=num_batches, desc="Progress", unit="iteration")

        for i in range(num_batches):

            # visualize the progress over the batches

            progress_bar.update(1)

            # reset the state of self.memory recurrent layer

            self.memory.reset_states()

            self.reservoir.reset_states()

            xlocal = x[i * batch_size:(i + 1) * batch_size, :, :]

            original_shape = xlocal.shape

            if xlocal.shape[0] < batch_size:
                xlocal = np.concatenate(
                    (xlocal, np.zeros((batch_size - xlocal.shape[0], xlocal.shape[1], xlocal.shape[2]))), axis=0)

            for t in range(xlocal.shape[1]):
                memory_reservoir_states = self.memory(xlocal[:, t:t + 1, :])

                concatenated_input = np.concatenate((memory_reservoir_states, xlocal[:, t:t + 1, :]), axis=-1)

                x_train_states = self.reservoir(concatenated_input)

            x_train_states_all[i * batch_size:(i + 1) * batch_size, :] = x_train_states[:original_shape[0], :]

        progress_bar.close()

        self.readout.fit(x_train_states_all, y)

        print('model trained')
