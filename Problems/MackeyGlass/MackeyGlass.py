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





def Model_LMU_AB_Tuning(hp):
    seed = 0
    layerN = 1
    memoryDim = hp.Choice("memoryDim", values=[1, 2, 4, 8, 16, 32, 48, 64])
    order = hp.Int("order", min_value=128, max_value=512, step=32)
    theta = hp.Int("theta", min_value=16, max_value=512, step=16)

    stepSize = 32
    hiddenUnit = hp.Int("hiddenUnit", min_value=stepSize, max_value=stepSize * 16, step=stepSize)
    spectraRadius = -1
    leaky = -1  # task step invariant so no need to change this parameter
    trainableAB = True

    hiddenCell = ks.layers.SimpleRNNCell(hiddenUnit, kernel_initializer=GlorotUniform(seed),
                                         recurrent_initializer=GlorotUniform(seed))

    memoryToMemory = hp.Boolean("memoryToMemory")
    hiddenToMemory = hp.Boolean("hiddenToMemory")
    inputToHiddenCell = hp.Boolean("inputToHiddenCell")
    useBias = hp.Boolean("useBias")

    reservoirMode = False
    memoryToMemoryScaler = -1
    hiddenToMemoryScaler = -1
    inputToHiddenCellScaler = -1
    biasScaler = -1

    return Model_LRMU_Prediction(PROBLEM_NAME, "LMU-AB", SEQUENCE_LENGTH,
                                 memoryDim, order, theta, trainableAB,
                                 hiddenUnit, spectraRadius, leaky,
                                 reservoirMode, hiddenCell,
                                 memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                                 memoryToMemoryScaler, hiddenToMemoryScaler, inputToHiddenCellScaler, biasScaler,
                                 seed, layerN)


def Model_LMU_ESN_Tuning(hp):
    seed = 0
    layerN = 1
    memoryDim = hp.Choice("memoryDim", values=[1, 2, 4, 8, 16, 32, 48, 64])
    order = hp.Int("order", min_value=128, max_value=512, step=32)
    theta = hp.Int("theta", min_value=16, max_value=512, step=16)

    stepSize = 32
    hiddenUnit = hp.Int("hiddenUnit", min_value=stepSize, max_value=stepSize * 16, step=stepSize)
    spectraRadius = hp.Float("spectraRadius", min_value=0.8, max_value=1.3, step=0.05)
    leaky = 1  # task step invariant so no need to change this parameter
    trainableAB = False

    hiddenCell = None

    memoryToMemory = hp.Boolean("memoryToMemory")
    hiddenToMemory = hp.Boolean("hiddenToMemory")
    inputToHiddenCell = hp.Boolean("inputToHiddenCell")
    useBias = hp.Boolean("useBias")

    reservoirMode = False
    memoryToMemoryScaler = -1
    hiddenToMemoryScaler = -1
    inputToHiddenCellScaler = -1
    biasScaler = -1

    return Model_LRMU_Prediction(PROBLEM_NAME, "LMU_ESN", SEQUENCE_LENGTH,
                                 memoryDim, order, theta, trainableAB,
                                 hiddenUnit, spectraRadius, leaky,
                                 reservoirMode, hiddenCell,
                                 memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                                 memoryToMemoryScaler, hiddenToMemoryScaler, inputToHiddenCellScaler, biasScaler,
                                 seed, layerN)



def Model_LMU_RE_Tuning(hp):
    seed = 0
    layerN = 1
    memoryDim = hp.Choice("memoryDim", values=[1, 2, 4, 8, 16, 32, 48, 64])
    order = hp.Int("order", min_value=128, max_value=512, step=32)
    theta = hp.Int("theta", min_value=16, max_value=512, step=16)

    hidden_unit_stepSize = 32
    hiddenUnit = hp.Int("hiddenUnit", min_value=hidden_unit_stepSize, max_value=hidden_unit_stepSize * 16, step=hidden_unit_stepSize)
    spectraRadius = -1
    leaky = -1  # task step invariant so no need to change this parameter
    trainableAB = False

    hiddenCell = None

    memoryToMemory = hp.Boolean("memoryToMemory")
    hiddenToMemory = hp.Boolean("hiddenToMemory")
    inputToHiddenCell = hp.Boolean("inputToHiddenCell")
    useBias = hp.Boolean("useBias")

    reservoirMode = True
    memoryToMemoryScaler = hp.Float("memoryToMemoryScaler", min_value=0.5, max_value=2, step=0.25)
    hiddenToMemoryScaler = hp.Float("hiddenToMemoryScaler", min_value=0.5, max_value=2, step=0.25)
    inputToHiddenCellScaler = hp.Float("inputToHiddenCellScaler", min_value=0.5, max_value=2, step=0.25)
    biasScaler = hp.Float("biasScaler", min_value=0.5, max_value=2, step=0.25)


    return Model_LRMU_Prediction(PROBLEM_NAME, "LMU-RE", SEQUENCE_LENGTH,
                                 memoryDim, order, theta, trainableAB,
                                 hiddenUnit, spectraRadius, leaky,
                                 reservoirMode, hiddenCell,
                                 memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                                 memoryToMemoryScaler, hiddenToMemoryScaler, inputToHiddenCellScaler, biasScaler,
                                 seed, layerN)

def Model_LRMU_Tuning(hp):
    seed = 0
    layerN = 1
    memoryDim = hp.Choice("memoryDim", values=[1, 2, 4, 8, 16, 32, 48, 64])
    order = hp.Int("order", min_value=128, max_value=512, step=32)
    theta = hp.Int("theta", min_value=16, max_value=512, step=16)

    hidden_unit_stepSize = 32
    hiddenUnit = hp.Int("hiddenUnit", min_value=hidden_unit_stepSize, max_value=hidden_unit_stepSize * 16, step=hidden_unit_stepSize)
    spectraRadius = hp.Float("spectraRadius", min_value=0.8, max_value=1.3, step=0.05)
    leaky = 1  # task step invariant so no need to change this parameter
    trainableAB = False

    hiddenCell = None

    memoryToMemory = hp.Boolean("memoryToMemory")
    hiddenToMemory = hp.Boolean("hiddenToMemory")
    inputToHiddenCell = hp.Boolean("inputToHiddenCell")
    useBias = hp.Boolean("useBias")

    reservoirMode = True
    memoryToMemoryScaler = hp.Float("memoryToMemoryScaler", min_value=0.5, max_value=2, step=0.25)
    hiddenToMemoryScaler = hp.Float("hiddenToMemoryScaler", min_value=0.5, max_value=2, step=0.25)
    inputToHiddenCellScaler = hp.Float("inputToHiddenCellScaler", min_value=0.5, max_value=2, step=0.25)
    biasScaler = hp.Float("biasScaler", min_value=0.5, max_value=2, step=0.25)


    return Model_LRMU_Prediction(PROBLEM_NAME, "LRMU", SEQUENCE_LENGTH,
                                 memoryDim, order, theta, trainableAB,
                                 hiddenUnit, spectraRadius, leaky,
                                 reservoirMode, hiddenCell,
                                 memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                                 memoryToMemoryScaler, hiddenToMemoryScaler, inputToHiddenCellScaler, biasScaler,
                                 seed, layerN)

def ModelLRMU_SelectedHP():
    return Model_LRMU_Prediction(PROBLEM_NAME, "LRMU", SEQUENCE_LENGTH,
                                 2, 4, 160, 450, False,
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
        #TunerTraining(Model_LMU_AB_Tuning, "LRMU_LMU_AB_Tuning_5k", PROBLEM_NAME, training, validation, 5, 150, False)
        TunerTraining(Model_LMU_RE_Tuning, "LMU_RE_Tuning_5k", PROBLEM_NAME, training, validation, 5, 150,True)
        TunerTraining(Model_LMU_ESN_Tuning, "LMU_ESN_Tuning_5k", PROBLEM_NAME, training, validation, 5,150,True)
        TunerTraining(Model_LRMU_Tuning, "LRMU_Tuning_5k", PROBLEM_NAME, training, validation,5, 150,True)


