import tensorflow.keras as ks
from tensorflow.keras.initializers import *
from LRMU.layer import LRMU
from LMU import LMU
import tensorflow as tf
from LRMU.Model import LRMU_ESN_Ridge


def rms(x, axis=-1, keepdims=False):
    print("here2")
    return tf.reduce_mean(tf.square(x), axis=axis, keepdims=keepdims)


def nrmse(actual, target, **kwargs):
    print(tf.sqrt(rms(actual - target) / rms(target)).shape)
    return tf.sqrt(rms(actual - target) / rms(target))


def nrmse_metric(y_true, y_pred):
    return nrmse(y_pred, y_true)


class ModelBuilder:

    def __init__(self, modelVariant, problemName, extraName=None, seed=0):
        self.Input = None
        self.Feature = None
        self.Outputs = None
        self.Model = None
        self.ProblemName = problemName
        self.ModelVariant = modelVariant

        self.ModelName = f"{self.ModelVariant}_{self.ProblemName}"
        if extraName is not None:
            self.ModelName = f"{self.ModelName}_{extraName}"

        self.Seed = seed

    def inputLayer(self, sequenceLength, featureNumber=1, Flatten=False):
        if not Flatten:
            self.Input = ks.Input(shape=(sequenceLength, featureNumber), name=f"Input")
        else:
            self.Input = ks.Input(shape=(sequenceLength * featureNumber,), name=f"Input-Flatten")
        return self

    def LRMU(self, memoryDim, order, theta, hiddenCell,
             hiddenToMemory, memoryToMemory, inputToHiddenCell, useBias,
             hiddenEncoderScaler, memoryEncoderScaler, inputEncoderScaler, biasScaler, layerN):
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

    def FF_Baseline(self):
        self.Feature = self.Input

    def LRMU_ESN_Ridge(self, modelType, sequenceLength, memoryDim, order, theta,
                       hiddenToMemory, memoryToMemory, inputToHiddenCell, useBias,
                       hiddenEncoderScaler, memoryEncoderScaler, InputEncoderScaler, biasEncoderScaler,
                       hiddenUnit, spectraRadius, leaky, inputScaler, biasScaler,
                       redoutRegulizer):
        return LRMU_ESN_Ridge(modelType, sequenceLength, memoryDim, order, theta,
                              hiddenToMemory, memoryToMemory, inputToHiddenCell, useBias,
                              hiddenEncoderScaler, memoryEncoderScaler, InputEncoderScaler, biasEncoderScaler,
                              hiddenUnit, tf.nn.tanh, spectraRadius, leaky, inputScaler, biasScaler, redoutRegulizer,
                              seed=self.Seed)

    def BuildClassification(self, classNuber, IsCategorical):
        self.Outputs = ks.layers.Dense(classNuber, activation="softmax", kernel_initializer=GlorotUniform(self.Seed),name="Output_Class")(self.Feature)
        self.__ComposeModel()
        if IsCategorical:
            return self.__Compile("adam", "categorical_crossentropy", ["categorical_accuracy"])
        else:
            return self.__Compile("adam", "sparse_categorical_crossentropy", ["accuracy"])

    def BuildPrediction(self, featureDimension, activation):
        self.Outputs = ks.layers.Dense(featureDimension, activation, kernel_initializer=GlorotUniform(self.Seed),name="Output_Pred")(self.Feature)
        self.__ComposeModel()
        return self.__Compile("adam", "mse", ["mae"])

    def __ComposeModel(self):
        self.Model = ks.Model(inputs=self.Input, outputs=self.Outputs, name=self.ModelName)
        self.Model.summary()
        return self

    def __Compile(self, optimizer, loss, metrics):
        self.Model.compile(optimizer, loss, metrics=metrics)
        return self.Model
