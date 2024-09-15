import tensorflow.keras as ks
from tensorflow.keras.initializers import *
from LRMU import LRMU
from LMU import LMU
import tensorflow as tf


def rms(x, axis=-1, keepdims=False):
    print("here2")
    return tf.reduce_mean(tf.square(x), axis=axis, keepdims=keepdims)


def nrmse(actual, target, **kwargs):
    print(tf.sqrt(rms(actual - target) / rms(target)).shape)
    return tf.sqrt(rms(actual - target) / rms(target))


def nrmse_metric(y_true, y_pred):
    return nrmse(y_pred, y_true)


class ModelBuilder:

    def __init__(self, problemName, modelVariant, seed=0):
        self.Input = None
        self.Feature = None
        self.Outputs = None
        self.Model = None
        self.ProblemName = problemName
        self.ModelVariant = modelVariant
        self.Seed = seed

    def inputLayer(self, sequenceLenght):
        self.Input = ks.Input(shape=(sequenceLenght, 1), name=f"{self.ModelVariant}_Input")
        return self

    def LRMU(self, memoryDim, order, theta, hiddenCell,
             hiddenToMemory, memoryToMemory, inputToHiddenCell, useBias,
             memoryEncoderScaler, hiddenEncoderScaler, inputEncoderScaler, biasScaler, layerN):
        feature = LRMU(memoryDim, order, theta, hiddenCell, hiddenToMemory, memoryToMemory, inputToHiddenCell, useBias,
                       hiddenEncoderScaler, memoryEncoderScaler, inputEncoderScaler, biasScaler, self.Seed,
                       returnSequences=layerN > 1)(self.Input)
        for i in range(layerN - 1):
            feature = LRMU(memoryDim, order, theta, hiddenCell, hiddenToMemory, memoryToMemory, inputToHiddenCell,
                           useBias, hiddenEncoderScaler, memoryEncoderScaler, inputEncoderScaler, biasScaler, self.Seed,
                           returnSequences=i != layerN - 2)(feature)
        self.Feature = feature
        return self

    def LMU(self, memoryDim, order, theta, hiddenCell, trainableTheta,
            hiddenToMemory, memoryToMemory, inputToHiddenCell, useBias,
            layerN):
        feature = LMU(memoryDim, order, theta, hiddenCell, trainableTheta,
                      hiddenToMemory, memoryToMemory, inputToHiddenCell, use_bias=useBias
                      , return_sequences=layerN > 1)(self.Input)
        for i in range(layerN - 1):
            feature = LMU(memoryDim, order, theta, hiddenCell, trainableTheta,
                          hiddenToMemory, memoryToMemory, inputToHiddenCell, use_bias=useBias
                          , return_sequences=i != layerN - 2)(feature)
        self.Feature = feature
        return self

    def BuildClassification(self, classNuber):
        self.Outputs = ks.layers.Dense(classNuber, activation="softmax", kernel_initializer=GlorotUniform(self.Seed))(
            self.Feature)
        self.__ComposeModel()
        return self.__Compile("adam", "categorical_crossentropy", ["accuracy"])

    def BuildPrediction(self, unit, activation):
        self.Outputs = ks.layers.Dense(unit, activation=activation, kernel_initializer=GlorotUniform(self.Seed))(
            self.Feature)
        self.__ComposeModel()
        return self.__Compile("adam", "mse", ["mae"])

    def __ComposeModel(self):
        self.Model = ks.Model(inputs=self.Input, outputs=self.Outputs,
                              name=f"{self.ProblemName}_{self.ModelVariant}_Model")
        self.Model.summary()
        return self

    def __Compile(self, optimizer, loss, metrics):
        self.Model.compile(optimizer, loss, metrics=metrics)
        return self.Model
