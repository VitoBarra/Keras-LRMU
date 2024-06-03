import tensorflow as tf
import keras as ks
import keras_tuner
from keras.src.initializers import GlorotUniform

from Utility.ArffFormatUtill import *
from Utility.DataUtil import *
import LMU as LMULayer
import LRMU as LRMULayer
import os

from Utility.Debug import PrintAvailableGPU
from Utility.LRMU_utility import GenerateLRMUFeatureLayer
from Utility.ModelUtil import TrainAndTestModel_OBJ, TunerTraining
from Utility.PlotUtil import *

PROBLEM_NAME = "ECG5000"
SEQUENCE_LENGHT = 140
CLASS_NUMBER = 5


def ModelFFBaseline():
    inputs = ks.Input(shape=(SEQUENCE_LENGHT,), name="ECG5000_FF_Input")
    feature = ks.layers.Dense(200)(inputs)
    feature = ks.layers.Dense(200)(feature)
    outputs = ks.layers.Dense(CLASS_NUMBER, activation="softmax")(feature)
    model = ks.Model(inputs=inputs, outputs=outputs, name="ECG5000Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def ModelLSTM():
    inputs = ks.Input(shape=(SEQUENCE_LENGHT, 1), name="ECG5000_Input")
    feature = ks.layers.LSTM(52, return_sequences=True)(inputs)
    feature = ks.layers.LSTM(52, return_sequences=False)(feature)
    outputs = ks.layers.Dense(CLASS_NUMBER, activation="softmax")(feature)
    model = ks.Model(inputs=inputs, outputs=outputs, name="ECG5000Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def ModelLRMU(memoryDim, order, theta, hiddenUnit, spectraRadius, leaky, reservoirMode, hiddenCell,
              memoryToMemory, hiddenToMemory, inputToCell, useBias, seed, layerN=1):
    inputs = ks.Input(shape=(SEQUENCE_LENGHT, 1), name="ECG5000_Input_LRMU")
    feature = GenerateLRMUFeatureLayer(inputs,
                                       memoryDim, order, theta,
                                       hiddenUnit, spectraRadius, leaky, reservoirMode,
                                       hiddenCell, memoryToMemory, hiddenToMemory, inputToCell, useBias,
                                       seed, layerN)
    outputs = ks.layers.Dense(CLASS_NUMBER, activation="softmax")(feature)
    model = ks.Model(inputs=inputs, outputs=outputs, name="ECG5000Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def ModelLRMU_ESN_Tuning(hp):
    memoryDim = hp.Int("memoryDim", min_value=10, max_value=100, step=5)
    order = hp.Int("order", min_value=32, max_value=512, step=32)
    theta = SEQUENCE_LENGHT

    hiddenUnit = hp.Int("hiddenUnit", min_value=50, max_value=1500, step=50)
    spectraRadius = hp.Float("spectraRadius", min_value=0.8, max_value=1.25, step=0.05)
    leaky = hp.Float("leaky", 0.5, 1, 0.1)

    reservoirMode = True
    hiddenCell = None

    memoryToMemory = hp.Boolean("memoryToMemory")
    hiddenToMemory = hp.Boolean("hiddenToMemory")
    inputToHiddenCell = hp.Boolean("inputToCell")
    useBias = hp.Boolean("useBias")

    seed = 0

    return ModelLRMU(memoryDim, order, theta,
                     hiddenUnit, spectraRadius, leaky,
                     reservoirMode, hiddenCell,
                     memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                     seed)


def ModelLRMU_SimpleRNN_Tuning(hp):
    memoryDim = hp.Int("memoryDim", min_value=10, max_value=100, step=5)
    order = hp.Int("order", min_value=32, max_value=512, step=32)
    theta = SEQUENCE_LENGHT

    hiddenUnit = hp.Int("hiddenUnit", min_value=50, max_value=1500, step=50)
    spectraRadius = None
    leaky = None

    reservoirMode = True
    hiddenCell = ks.layers.SimpleRNNCell(hiddenUnit, kernel_initializer=GlorotUniform(seed),
                                         recurrent_initializer=GlorotUniform(seed))

    memoryToMemory = hp.Boolean("memoryToMemory")
    hiddenToMemory = hp.Boolean("hiddenToMemory")
    inputToHiddenCell = hp.Boolean("inputToCell")
    useBias = hp.Boolean("useBias")

    seed = 0

    return ModelLRMU(memoryDim, order, theta,
                     hiddenUnit, spectraRadius, leaky,
                     reservoirMode, hiddenCell,
                     memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                     seed)


def ModelLRMU_P():
    return ModelLRMU(15, 64, 1050, 0.8, True, None, True, False, True, False)


def FullTraining(training, validation, test):
    history, result = TrainAndTestModel_OBJ(ModelLRMU_P, training, validation, test, 64, 10)

    print('Test loss:', result[0])
    print('Test accuracy:', result[1])
    PlotModelAccuracy(history, "Problems LRMU", f"./plots/{PROBLEM_NAME}", f"{PROBLEM_NAME}_LRMU_ESN")


def Run(fullTraining=True):
    path = "Data/"
    Data, Label = ReadFromCSVToKeras(path + "ECG5000_ALL.csv")
    Label -= 1
    training, validation, test = SplitDataset(Data, Label, 0.15, 0.1)

    if fullTraining:
        FullTraining(training, validation, test)
    else:
        TunerTraining(ModelLRMU_ESN_Tuning, "LRMU_ESN_Tuning", PROBLEM_NAME, training, validation, 10, 100, False)
        TunerTraining(ModelLRMU_SimpleRNN_Tuning, "LRMU_RNN_Tuning", PROBLEM_NAME, training, validation, 10, 100, False)
