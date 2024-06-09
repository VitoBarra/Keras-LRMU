import numpy as np
import numpy.random as rng
from keras.src.initializers import GlorotUniform

from Utility.DataUtil import SplitDataset
from Utility.PlotUtil import *
from Utility.LRMU_utility import *
from Utility.ModelUtil import *

PROBLEM_NAME = "psMNIST"
CLASS_NUMBER = 10
SEQUENCE_LENGTH = 784


def ModelLRMU(memoryDim, order, theta, hiddenUnit, spectraRadius, leaky, reservoirMode, hiddenCell,
              memoryToMemory, hiddenToMemory, inputToCell, useBias, seed, layerN=1):
    inputs = ks.Input(shape=(SEQUENCE_LENGTH, 1), name=f"{PROBLEM_NAME}_LRMU_Input")
    feature = GenerateLRMUFeatureLayer(inputs,
                                       memoryDim, order, theta, hiddenUnit,
                                       spectraRadius, leaky,
                                       reservoirMode, hiddenCell,
                                       memoryToMemory, hiddenToMemory, inputToCell, useBias,
                                       seed, layerN)
    outputs = ks.layers.Dense(CLASS_NUMBER, activation="softmax")(feature)
    model = ks.Model(inputs=inputs, outputs=outputs, name=f"{PROBLEM_NAME}_LRMU_Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def ModelLRMU_ESN_Tuning(hp):
    seed = 0
    layerN = 1
    memoryDim = hp.Choice("memoryDim", values=[1, 2, 4, 8])
    order = hp.Int("order", min_value=128, max_value=512, step=32)
    theta = SEQUENCE_LENGTH

    hiddenUnit = hp.Int("HiddenUnit", min_value=64, max_value=512, step=64)
    spectraRadius = hp.Float("spectraRadius", min_value=0.8, max_value=1.3, step=0.05)
    leaky = 1  # task step invariant so no need to change this parameter

    reservoirMode = True
    hiddenCell = None

    memoryToMemory = False
    hiddenToMemory = True
    inputToHiddenCell = False
    useBias = False

    return ModelLRMU(memoryDim, order, theta, hiddenUnit, spectraRadius, leaky, reservoirMode, hiddenCell,
                     memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias, seed, layerN)


def ModelLRMU_SimpleRNN_Tuning(hp):
    seed = 0
    layerN = 1

    memoryDim = hp.Choice("memoryDim", values=[1, 2, 4, 8])
    order = hp.Int("order", min_value=128, max_value=512, step=32)
    theta = SEQUENCE_LENGTH

    hiddenUnit = hp.Int("hiddenUnit", min_value=64, max_value=512, step=64)
    spectraRadius = None
    leaky = None

    reservoirMode = True
    hiddenCell = ks.layers.SimpleRNNCell(hiddenUnit, kernel_initializer=GlorotUniform(seed), recurrent_initializer=GlorotUniform(seed))

    memoryToMemory = False
    hiddenToMemory = True
    inputToHiddenCell = False
    useBias = False

    return ModelLRMU(memoryDim, order, theta, hiddenUnit, spectraRadius, leaky, reservoirMode, hiddenCell,
                     memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias, seed)


def ModelLRMU_ESN_stack_Tuning(hp):
    seed = 0
    layerN = hp.Choice("layerN", [2, 3, 4, 5])

    memoryDim = hp.Choice("memoryDim", [2, 4, 8, 16, 32])
    order = hp.Choice("order", [4, 8, 16, 32, 64])
    theta = hp.Int("theta", 16, 256, 16)

    hiddenUnit = hp.Int("hiddenUnit", 128, 256, 64)
    spectraRadius = hp.Float("spectraRadius", min_value=0.8, max_value=1.3, step=0.05)
    leaky = hp.Float("leaky", 0.5, 1, 0.05)

    hiddenCell = ks.layers.SimpleRNNCell(hiddenUnit, kernel_initializer=GlorotUniform(seed),
                                         recurrent_initializer=GlorotUniform(seed))

    memoryToMemory = False
    hiddenToMemory = True
    inputToHiddenCell = False
    useBias = False

    reservoirMode = True
    return ModelLRMU(memoryDim, order, theta,
                     hiddenUnit, spectraRadius, leaky,
                     reservoirMode, hiddenCell,
                     memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                     seed, layerN)


def ModelLRMU_SimpleRNN_stack_Tuning(hp):
    seed = 0
    layerN = hp.Choice("layerN", [2, 3, 4, 5])

    memoryDim = hp.Choice("memoryDim", values=[1, 2, 4, 8, 16])
    order = hp.Int("order", min_value=4, max_value=64, step=4)
    hiddenUnit = hp.Int("hiddenUnit", min_value=32, max_value=256, step=32)
    theta = hp.Int("theta", 4, 128, 4)

    spectraRadius = None
    leaky = None

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


# 75, 352, 500, 1.15,True, None, False, True, False, True = 92.22% accuracy on test set
# 75, 352, 500, 1.18,True, None, False, True, False, True = 91.88% accuracy on test set
# 256, 128, 212, 1.15,True, None, False, True, False, True = 87.4% accuracy on test set
# 256, 128, 212, 0.99,True, None, False, True, False, True = 87.4% accuracy on test set
# 256, 256, 212, 0.99,True, None, False, True, False, True = 88.3% accuracy on test set
# 256, 256, 212, 1.18,True, None, False, True, False, True = 88.55%  accuracy on test set
def ModelLRMU_SelectedHP():
    return ModelLRMU(1, 256, SEQUENCE_LENGTH,
                     212, 1.18, 1,
                     True, None,
                     False, True, False, False,
                     0)


def SingleTraining(training, validation, test):
    history, result = TrainAndTestModel_OBJ(ModelLRMU_SelectedHP, training, validation, test, 64, 10)

    print(f"Test loss: {result[0]}")
    print(f"Test accuracy: {result[1]}")
    PlotModelAccuracy(history, "Problems LRMU", f"./plots/{PROBLEM_NAME}", f"{PROBLEM_NAME}_LRMU")


def Run(singleTraining=True):
    ((train_images, train_labels), (test_images, test_labels)) = ks.datasets.mnist.load_data()

    Data = np.concatenate((train_images, test_images), axis=0)
    Label = np.concatenate((train_labels, test_labels), axis=0)

    Data = Data.reshape(Data.shape[0], -1, 1)
    rng.seed(1509)
    perm = rng.permutation(Data.shape[1])
    Data = Data[:15000, perm]
    Label = Label[:15000]
    training, validation, test = SplitDataset(Data, Label, 0.1, 0.1)

    if singleTraining:
        SingleTraining(training, validation, test)
    else:
        #TunerTraining(ModelLRMU_ESN_Tuning, "LRMU_ESN_tuning_15k", PROBLEM_NAME, training, validation, 5, 100, False)
        TunerTraining(ModelLRMU_SimpleRNN_Tuning, "LRMU_RNN_tuning_15k", PROBLEM_NAME, training, validation, 5, 100,
                      False)
        TunerTraining(ModelLRMU_ESN_stack_Tuning, "LRMU_RNN_Stack_tuning_15k", PROBLEM_NAME, training, validation, 5,
                      100,
                      False)
        TunerTraining(ModelLRMU_SimpleRNN_stack_Tuning, "LRMU_RNN_Stack_tuning_15k", PROBLEM_NAME, training, validation,
                      5, 100,
                      False)
