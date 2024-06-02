import collections

import keras_tuner
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

import LRMU as lrmu

from keras.initializers import *

from Utility.DataUtil import SplitDataset
from Utility.ModelUtil import *
from Utility.PlotUtil import *
import Model.MackeyGlass.DataGeneration as dg

PROBLEM_NAME = "MackeyGlass"
SEQUENCE_LENGHT = 5000


def ModelLRMU_SelectedHP():
    # return ModelLRMU(2, 16, 32,
    #                  416, 1.05, 1,
    #                  True, None,
    #                  False, True, False, False, 0)
    return ModelLRMU(2, 4, 160,
                     384, 1.05, 0.6,
                     True, None,
                     False, True, False, False, 0)


def ModelLRMU(memoryDim, order, theta, hiddenUnit, spectraRadius, leaky, reservoirMode, hiddenCell,
              memoryToMemory, hiddenToMemory, inputToCell, useBias, seed):
    inputs = ks.Input(shape=(SEQUENCE_LENGHT, 1), name="Mackey-Glass_Input_LRMU")
    feature = lrmu.LRMU(memoryDimension=memoryDim, order=order, theta=theta,
                        hiddenUnit=hiddenUnit, spectraRadius=spectraRadius, leaky=leaky,
                        reservoirMode=reservoirMode, hiddenCell=hiddenCell,
                        memoryToMemory=memoryToMemory, hiddenToMemory=hiddenToMemory, inputToHiddenCell=inputToCell,
                        useBias=useBias,
                        seed=seed, returnSequences=True)(inputs)
    feature = lrmu.LRMU(memoryDimension=memoryDim, order=order, theta=theta,
                        hiddenUnit=hiddenUnit, spectraRadius=spectraRadius, leaky=leaky,
                        reservoirMode=reservoirMode, hiddenCell=hiddenCell,
                        memoryToMemory=memoryToMemory, hiddenToMemory=hiddenToMemory, inputToHiddenCell=inputToCell,
                        useBias=useBias,
                        seed=seed, returnSequences=True)(feature)
    feature = lrmu.LRMU(memoryDimension=memoryDim, order=order, theta=theta,
                        hiddenUnit=hiddenUnit, spectraRadius=spectraRadius, leaky=leaky,
                        reservoirMode=reservoirMode, hiddenCell=hiddenCell,
                        memoryToMemory=memoryToMemory, hiddenToMemory=hiddenToMemory, inputToHiddenCell=inputToCell,
                        useBias=useBias,
                        seed=seed, returnSequences=False)(feature)
    outputs = ks.layers.Dense(1, activation="linear", kernel_initializer=GlorotUniform(seed))(feature)
    model = ks.Model(inputs=inputs, outputs=outputs, name="Mackey-Glass_LRMU_Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="mse",
                  metrics=["mse"])
    return model


def ModelLRMUWhitTuning(hp):
    memoryDim = hp.Choice("memoryDim", [2, 4, 8, 16, 32])
    order = hp.Choice("order", [4, 8, 16, 32, 64])
    theta = hp.Int("theta", 16, 256, 16)

    hiddenUnit = hp.Int("hiddenUnit", 128, 256, 64)
    spectraRadius = 0
    leaky = 0

    memoryToMemory = False
    hiddenToMemory = True
    inputToHiddenCell = False
    useBias = False

    reservoirMode = True
    seed = 0
    hiddenCell = ks.layers.SimpleRNNCell(hiddenUnit, kernel_initializer=GlorotUniform(seed),
                                         recurrent_initializer=GlorotUniform(seed))
    return ModelLRMU(memoryDim, order, theta,
                     hiddenUnit, spectraRadius, leaky,
                     reservoirMode, hiddenCell,
                     memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                     seed)


def FullTraining(training, validation, test):
    history, result = TrainAndTestModel_OBJ(ModelLRMU_SelectedHP, training, validation, test, 64, 15, "val_loss")

    print("test loss:", result[0])
    print("test mse:", result[1])
    PlotModelLoss(history, "Model LRMU", f"./plots/{PROBLEM_NAME}", f"{PROBLEM_NAME}_LRMU_ESN")


def Run(fullTraining=True):
    data, label = dg.generate_data(128, SEQUENCE_LENGHT)
    training, validation, test = SplitDataset(data, label, 0.1, 0.1)
    print(data.shape, label.shape)

    if fullTraining:
        FullTraining(training, validation, test)
    else:
        TunerTraining(ModelLRMUWhitTuning, "LRMU_ESN_Tuning", PROBLEM_NAME, training, validation, False)
