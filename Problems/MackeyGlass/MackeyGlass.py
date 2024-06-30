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
from Utility.LRMU_utility import *
import Problems.MackeyGlass.DataGeneration as dg

PROBLEM_NAME = "Mackey-Glass"
SEQUENCE_LENGTH = 5000


def SelectCell(hp, ESN, hiddenUnit, seed):
    if ESN:
        hiddenCell = None
        spectraRadius = hp.Float("spectraRadius", min_value=0.8, max_value=1.3, step=0.05)
        leaky = 1  # task step invariant so no need to change this parameter
    else:
        hiddenCell = ks.layers.SimpleRNNCell(hiddenUnit, kernel_initializer=GlorotUniform(seed))
        spectraRadius = -1
        leaky = -1
    return hiddenCell, spectraRadius, leaky


def LMU_Par(hp, useESN):
    seed = 0
    layerN = 1
    memoryDim = hp.Choice("memoryDim", values=[1, 2, 4, 8, 16, 32, 48, 64])
    order = hp.Choice("memoryDim", values=[1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64])
    theta = hp.Int("theta", min_value=16, max_value=258, step=16)
    hiddenUnit = hp.Int("hiddenUnit", min_value=32, max_value=32 * 15, step=32)
    return seed, layerN, memoryDim, order, theta, hiddenUnit, SelectCell(hp, useESN, hiddenUnit, seed)


def SelectScaler(hp, reservoirMode, memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias):
    memoryToMemoryScaler = -1
    hiddenToMemoryScaler = -1
    inputToHiddenCellScaler = -1
    biasScaler = -1
    if reservoirMode:
        if memoryToMemory:
            memoryToMemoryScaler = hp.Float("memoryToMemoryScaler", min_value=0.5, max_value=2, step=0.25)
        if hiddenToMemory:
            hiddenToMemoryScaler = hp.Float("hiddenToMemoryScaler", min_value=0.5, max_value=2, step=0.25)
        if inputToHiddenCell:
            inputToHiddenCellScaler = hp.Float("inputToHiddenCellScaler", min_value=0.5, max_value=2, step=0.25)
        if useBias:
            biasScaler = hp.Float("biasScaler", min_value=0.5, max_value=2, step=0.25)

    return memoryToMemoryScaler, hiddenToMemoryScaler, inputToHiddenCellScaler, biasScaler


def SelectConnection(hp, searchConnection, reservoirMode):
    if searchConnection:
        memoryToMemory = hp.Boolean("memoryToMemory")
        hiddenToMemory = hp.Boolean("hiddenToMemory")
        inputToHiddenCell = hp.Boolean("inputToHiddenCell")
        useBias = hp.Boolean("useBias")
    else:
        memoryToMemory = False
        hiddenToMemory = True
        inputToHiddenCell = False
        useBias = False
    return memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias, SelectScaler(hp, reservoirMode, memoryToMemory,
                                                                                    hiddenToMemory, inputToHiddenCell,
                                                                                    useBias)


def ConstructHyperModel(hp, modelName, useESN, reservoirMode, searchConnection):
    seed, layerN, memoryDim, order, theta, hiddenUnit, cellData = LMU_Par(hp, useESN)
    (hiddenCell, spectraRadius, leaky) = cellData

    memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias, scaler = SelectConnection(hp, searchConnection,
                                                                                          reservoirMode)
    (memoryToMemoryScaler, hiddenToMemoryScaler, inputToHiddenCellScaler, biasScaler) = scaler

    return Model_LRMU_Prediction(PROBLEM_NAME, modelName, SEQUENCE_LENGTH,
                                 memoryDim, order, theta,
                                 hiddenUnit, spectraRadius, leaky,
                                 reservoirMode, hiddenCell,
                                 memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                                 memoryToMemoryScaler, hiddenToMemoryScaler, inputToHiddenCellScaler, biasScaler,
                                 seed, layerN)


def Model_LMU_ESN_Tuning(hp):
    return ConstructHyperModel(hp, "LMU-ESN", True, False, True)


def Model_LMU_RE_Tuning(hp):
    return ConstructHyperModel(hp, "LMU-RE", False, True, True)


def Model_LRMU_Tuning(hp):
    return ConstructHyperModel(hp, "LRMU", True, True, True)


def ModelLRMU_SelectedHP():
    return Model_LRMU_Prediction(PROBLEM_NAME, "LRMU", SEQUENCE_LENGTH,
                                 2, 4, 160, 450,
                                 1.05, 0.6, True, None,
                                 False, True, False, False,
                                 1, 1, 1, 1, 0, 1)


def SingleTraining(training, validation, test):
    history, result = TrainAndTestModel_OBJ(ModelLRMU_SelectedHP, training, validation, test, 64, 15, "val_loss")

    print("test loss:", result[0])
    print("test mse:", result[1])
    PlotModelLoss(history, "Problems LRMU", f"./plots/{PROBLEM_NAME}", f"{PROBLEM_NAME}_LRMU_ESN")


def Run(singleTraining=True):
    data, label = dg.generate_data(128, SEQUENCE_LENGTH)
    training, validation, test = SplitDataset(data, label, 0.1, 0.1)
    print(data.shape, label.shape)

    if singleTraining:
        SingleTraining(training, validation, test)
    else:
        TunerTraining(Model_LMU_ESN_Tuning, "LMU_ESN_Tuning_5k", PROBLEM_NAME, training, validation, 5, 150, True)
        TunerTraining(Model_LMU_RE_Tuning, "LMU_RE_Tuning_5k", PROBLEM_NAME, training, validation, 5, 150, True)
        TunerTraining(Model_LRMU_Tuning, "LRMU_Tuning_5k", PROBLEM_NAME, training, validation, 5, 150, True)
