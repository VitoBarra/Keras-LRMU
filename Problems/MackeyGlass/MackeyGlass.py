import collections

import keras_tuner
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

import LRMU as lrmu

from keras.initializers import *

from Utility.DataUtil import SplitDataset
from Utility.LRMU_utility import GenerateLRMUFeatureLayer
from Utility.ModelUtil import *
from Utility.PlotUtil import *
import Problems.MackeyGlass.DataGeneration as dg

PROBLEM_NAME = "MackeyGlass"
SEQUENCE_LENGHT = 5000




def ModelLRMU(memoryDim, order, theta, hiddenUnit, spectraRadius, leaky, reservoirMode, hiddenCell,
              memoryToMemory, hiddenToMemory, inputToCell, useBias, seed, layerN=1):
    inputs = ks.Input(shape=(SEQUENCE_LENGHT, 1), name="Mackey-Glass_Input_LRMU")
    feature = feature = GenerateLRMUFeatureLayer(inputs,
                                                 memoryDim, order, theta,
                                                 hiddenUnit, spectraRadius, leaky,
                                                 reservoirMode, hiddenCell,
                                                 memoryToMemory, hiddenToMemory, inputToCell, useBias,
                                                 seed, layerN)
    outputs = ks.layers.Dense(1, activation="linear", kernel_initializer=GlorotUniform(seed))(feature)
    model = ks.Model(inputs=inputs, outputs=outputs, name="Mackey-Glass_LRMU_Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="mse",
                  metrics=["mse"])
    return model


def ModelLRMU_ESN_Tuning(hp):
    seed = 0
    layerN=1
    memoryDim = hp.Choice("memoryDim", [2, 4, 8, 16, 32])
    order = hp.Choice("order", [4, 8, 16, 32, 64])
    theta = hp.Int("theta", 16, 256, 16)

    hiddenUnit = hp.Int("hiddenUnit", 256, 512, 64)
    spectraRadius = hp.Float("spectraRadius", min_value=0.8, max_value=1.3, step=0.05)
    leaky = hp.Float("leaky", 0.5, 1, 0.05)

    reservoirMode = True
    hiddenCell = ks.layers.SimpleRNNCell(hiddenUnit, kernel_initializer=GlorotUniform(seed),
                                         recurrent_initializer=GlorotUniform(seed))

    memoryToMemory = False
    hiddenToMemory = True
    inputToHiddenCell = False
    useBias = False

    return ModelLRMU(memoryDim, order, theta,
                     hiddenUnit, spectraRadius, leaky,
                     reservoirMode, hiddenCell,
                     memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                     seed, layerN)


def ModelLRMU_SimpleRNN_Tuning(hp):
    seed = 0
    layerN =1

    memoryDim = hp.Choice("memoryDim", values=[1, 2, 4, 8, 16])
    order = hp.Int("order", min_value=4, max_value=64, step=4)
    theta = hp.Int("theta", 4, 128, 4)

    hiddenUnit = hp.Int("hiddenUnit", 256, 480, 32)
    spectraRadius = None
    leaky = None

    reservoirMode = True
    hiddenCell = ks.layers.SimpleRNNCell(hiddenUnit, kernel_initializer=GlorotUniform(seed),recurrent_initializer=GlorotUniform(seed))

    memoryToMemory = False
    hiddenToMemory = True
    inputToHiddenCell = False
    useBias = False

    return ModelLRMU(memoryDim, order, theta,
                     hiddenUnit, spectraRadius, leaky,
                     reservoirMode, hiddenCell,
                     memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                     seed,layerN)

def ModelLRMU_ESN_stack_Tuning(hp):
    seed = 0
    layerN = hp.Choice("layerN",[2,3,4,5])

    memoryDim = hp.Choice("memoryDim", [2, 4, 8, 16, 32])
    order = hp.Choice("order", [4, 8, 16, 32, 64])
    theta = hp.Int("theta", 16, 256, 16)

    hiddenUnit = hp.Int("hiddenUnit", 256, 512, 64)
    spectraRadius = hp.Float("spectraRadius", min_value=0.8, max_value=1.3, step=0.05)
    leaky = hp.Float("leaky", 0.5, 1, 0.05)

    hiddenCell = None

    memoryToMemory = False
    hiddenToMemory = True
    inputToHiddenCell = False
    useBias = False

    reservoirMode = True
    return ModelLRMU(memoryDim, order, theta,
                     hiddenUnit, spectraRadius, leaky,
                     reservoirMode, hiddenCell,
                     memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                     seed,layerN)


def ModelLRMU_SimpleRNN_stack_Tuning(hp):
    seed = 0
    layerN = hp.Choice("layerN",[2,3,4,5])

    memoryDim = hp.Choice("memoryDim", values=[1, 2, 4, 8, 16])
    order = hp.Int("order", min_value=4, max_value=64, step=4)
    theta = hp.Int("theta", 4, 128, 4)

    hiddenUnit = hp.Int("hiddenUnit", 256, 512, 64)
    spectraRadius = None
    leaky = None

    reservoirMode = True
    hiddenCell = ks.layers.SimpleRNNCell(hiddenUnit, kernel_initializer=GlorotUniform(seed),recurrent_initializer=GlorotUniform(seed))

    memoryToMemory = False
    hiddenToMemory = True
    inputToHiddenCell = False
    useBias = False

    return ModelLRMU(memoryDim, order, theta,
                     hiddenUnit, spectraRadius, leaky,
                     reservoirMode, hiddenCell,
                     memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                     seed, layerN)


def ModelLRMU_SelectedHP():
    # return ModelLRMU(2, 16, 32,
    #                  416, 1.05, 1,
    #                  True, None,
    #                  False, True, False, False, 0)
    # return ModelLRMU(2, 4, 160,
    #                  384, 1.05, 0.6,
    #                  True, None,
    #                  False, True, False, False, 0)
    return ModelLRMU(2,4,160,
                     450,1.05,0.6,
                     True,None,False,
                     True,False,False,
                     0,1)


def SingleTraining(training, validation, test):
    history, result = TrainAndTestModel_OBJ(ModelLRMU_SelectedHP, training, validation, test, 64, 15, "val_loss")

    print("test loss:", result[0])
    print("test mse:", result[1])
    PlotModelLoss(history, "Problems LRMU", f"./plots/{PROBLEM_NAME}", f"{PROBLEM_NAME}_LRMU_ESN")


def Run(singleTraining=True):
    data, label = dg.generate_data(128, SEQUENCE_LENGHT)
    training, validation, test = SplitDataset(data, label, 0.1, 0.1)
    print(data.shape, label.shape)

    if singleTraining:
        SingleTraining(training, validation, test)
    else:
        #TunerTraining(ModelLRMU_ESN_Tuning, "LRMU_ESN_Tuning", PROBLEM_NAME, training, validation, 10, 100, False)
        TunerTraining(ModelLRMU_SimpleRNN_Tuning, "LRMU_RNN_Tuning", PROBLEM_NAME, training, validation, 10, 100, False)
        TunerTraining(ModelLRMU_ESN_stack_Tuning, "LRMU_ESN_Stack_Tuning", PROBLEM_NAME, training, validation, 10, 100, False)
        TunerTraining(ModelLRMU_SimpleRNN_stack_Tuning, "LRMU_RNN_Stack_Tuning", PROBLEM_NAME, training, validation, 10, 100, False)


