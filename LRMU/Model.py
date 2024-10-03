import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from Reservoir.layer import ReservoirCell
from sklearn.linear_model import RidgeClassifier, Ridge
from LRMU.layer import LRMU
from LRMU.utility import ModelType


class LRMU_ESN_Ridge(keras.Model):
    def __init__(self, ModelType, sequenceLenght, memoryDimension, order, theta,
                 hiddenToMemory=False, memoryToMemory=False, inputToHiddenCell=False, useBias=False,
                 hiddenEncoderScaler=None, memoryEncoderScaler=None, InputEncoderScaler=None, biasScaler=None,
                 units=1, activation=tf.nn.tanh, spectral_radius=0.99, leaky=1,
                 input_scaling=1.0, bias_scaling=1.0,
                 readout_regularizer=1.0, features_dim=1, batch_size=None, seed=0,
                 **kwargs):

        super().__init__(**kwargs)
        self.metric_cust = None

        if batch_size is not None:
            self.reservoir = keras.Sequential([
                keras.Input(shape=(batch_size, None, features_dim)),
                LRMU(memoryDimension, order, theta, ReservoirCell(units=units,
                                                                  input_scaling=input_scaling,
                                                                  bias_scaling=bias_scaling,
                                                                  spectral_radius=spectral_radius,
                                                                  leaky=leaky, activation=activation),
                     hiddenToMemory, memoryToMemory, inputToHiddenCell, useBias,
                     hiddenEncoderScaler, memoryEncoderScaler, InputEncoderScaler, biasScaler, seed)
            ])

        else:
            self.reservoir = keras.Sequential([
                keras.Input(shape=( sequenceLenght, 1)),
                LRMU(memoryDimension, order, theta, ReservoirCell(units=units,
                                                                  input_scaling=input_scaling,
                                                                  bias_scaling=bias_scaling,
                                                                  spectral_radius=spectral_radius,
                                                                  leaky=leaky, activation=activation),
                     hiddenToMemory, memoryToMemory, inputToHiddenCell, useBias,
                     hiddenEncoderScaler, memoryEncoderScaler, InputEncoderScaler, biasScaler, seed)
            ])
        if ModelType == ModelType.Classification:
            self.readout = RidgeClassifier(alpha=readout_regularizer, solver='svd')
        elif ModelType == ModelType.Prediction:
            self.readout = Ridge(alpha=readout_regularizer, solver='svd')
        self.modelType = ModelType
        self.units = units
        self.features_dim = features_dim
        self.batch_size = batch_size

    def custom_compile(self, metric_cust):
        self.metric_cust = metric_cust
        return self

    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout.predict(reservoir_states.numpy())
        return output

    def fit(self, x, y, validation_data, callbacks=None, **kwargs):
        # For all the RC methods, we avoid doing the same reservoir operations at each epoch
        # To this aim, we pre-compute all the states and then we invoke the readout fit method

        flatten_y = y.reshape(y.shape[0], -1)

        if callbacks is not None:
            for callback in callbacks:
                callback.set_model(self)
                callback.on_train_begin()
                callback.on_epoch_begin(1)

        x_train_states = self.reservoir(x)
        self.readout.fit(x_train_states,flatten_y)
        prediction = self.readout.predict(x_train_states)

        metric_results = {}
        for metric in self.metric_cust:
            metric_results[metric.name] = [metric(flatten_y, prediction)]

        for metric in self.metric_cust:
            if validation_data is not None:
                val_x_state = self.reservoir(validation_data[0])
                val_pred = self.readout.predict(val_x_state)
                metric_results[f"val_{metric.name}"] = [metric(validation_data[1], val_pred)]

        if callbacks is not None:
            for callback in callbacks:
                callback.on_epoch_end(1)
                callback.on_train_end()

        history = keras.callbacks.History()
        history.history = metric_results
        return history

    def evaluate(self, x, y, **kwargs):
        x_states = self.reservoir(x)
        y_prediction = self.readout.predict(x_states)

        print(f"readout shape: {self.readout.coef_.shape[0]}")

        return [self.metric_cust[0](y, y_prediction), self.metric_cust[1](y, y_prediction)]
