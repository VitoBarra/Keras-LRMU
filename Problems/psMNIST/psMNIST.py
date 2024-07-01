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

def SelectCell(hp, ESN, hiddenUnit, seed):
    if ESN:
        hiddenCell = None
        spectraRadius = hp.Float("spectraRadius", min_value=0.8, max_value=1.3, step=0.05)
        leaky = 1  # task step invariant so no need to change this parameter
        ESNinputScaler = hp.Float("ESNinputScaler", min_value=0.5, max_value=2, step=0.25)
    else:
        hiddenCell = ks.layers.SimpleRNNCell(hiddenUnit, kernel_initializer=GlorotUniform(seed))
        spectraRadius = None
        leaky = None
        ESNinputScaler = None
    return hiddenCell, spectraRadius, leaky , ESNinputScaler


def LMU_Par(hp, useESN):
    seed = 0
    layerN = 1
    memoryDim = hp.Choice("memoryDim", values=[1, 2, 4, 8, 16, 32])
    order = hp.Choice("order", values=[1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64])
    theta = hp.Int("theta", min_value=16, max_value=258, step=16)
    hiddenUnit = hp.Int("hiddenUnit", min_value=16, max_value=16 * 20, step=16)
    return seed, layerN, memoryDim, order, theta, hiddenUnit, SelectCell(hp, useESN, hiddenUnit, seed)


def SelectScaler(hp, reservoirMode, memoryToMemory, hiddenToMemory, useBias):
    InputToMemoryScaler = hp.Float("InputToMemoryScaler", min_value=0.5, max_value=2, step=0.25)
    memoryToMemoryScaler = None
    hiddenToMemoryScaler = None
    biasScaler = None
    if reservoirMode:
        if memoryToMemory:
            memoryToMemoryScaler = hp.Float("memoryToMemoryScaler", min_value=0.5, max_value=2, step=0.25)
        if hiddenToMemory:
            hiddenToMemoryScaler = hp.Float("hiddenToMemoryScaler", min_value=0.5, max_value=2, step=0.25)
        if useBias:
            biasScaler = hp.Float("biasScaler", min_value=0.5, max_value=2, step=0.25)

    return memoryToMemoryScaler, hiddenToMemoryScaler, InputToMemoryScaler, biasScaler


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
        useBias = True
    return memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias, SelectScaler(hp, reservoirMode, memoryToMemory, hiddenToMemory, useBias)

def ConstructHyperModel(hp, modelName, useESN, reservoirMode, searchConnection):
    seed, layerN, memoryDim, order, theta, hiddenUnit, cellData = LMU_Par(hp, useESN)
    (hiddenCell, spectraRadius, leaky,ESNInputScaler) = cellData

    memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias, scaler = SelectConnection(hp, searchConnection,
                                                                                          reservoirMode)
    (memoryToMemoryScaler, hiddenToMemoryScaler, InputToMemoryScaler, biasScaler) = scaler

    return Model_LRMU_Prediction(PROBLEM_NAME, modelName, SEQUENCE_LENGTH,
                                 memoryDim, order, theta,
                                 hiddenUnit, spectraRadius, leaky, ESNInputScaler,
                                 reservoirMode, hiddenCell,
                                 memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                                 memoryToMemoryScaler, hiddenToMemoryScaler, InputToMemoryScaler, biasScaler,
                                 seed, layerN)


def Model_LMU_ESN_Tuning(hp):
    return ConstructHyperModel(hp, "LMU-ESN", True, False, True)


def Model_LMU_RE_Tuning(hp):
    return ConstructHyperModel(hp, "LMU-RE", False, True, True)


def Model_LRMU_Tuning(hp):
    return ConstructHyperModel(hp, "LRMU", True, True, True)


# 75, 352, 500, 1.15,True, None, False, True, False, True = 92.22% accuracy on test set xavierInitializer
# 75, 352, 500, 1.18,True, None, False, True, False, True = 91.88% accuracy on test set xavierInitializer
# 256, 128, 212, 1.15,True, None, False, True, False, True = 87.4% accuracy on test set xavierInitializer
# 256, 128, 212, 0.99,True, None, False, True, False, True = 87.4% accuracy on test set xavierInitializer
# 256, 256, 212, 0.99,True, None, False, True, False, True = 88.3% accuracy on test set xavierInitializer
# 256, 256, 212, 1.18,True, None, False, True, False, True = 88.55%  accuracy on test set xavierInitializer
def ModelLRMU_SelectedHP():
    return Model_LRMU_Classification(PROBLEM_NAME, "LRMU", SEQUENCE_LENGTH, 10,
                                     256, 212, 1.18, 1,
                                      1, 1, True, None,
                                     True, False, False, False,
                                     1,1, 1, 1, 1, 0)


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
        #TunerTraining(Model_LMU_AB_Tuning, "LRMU_LMU_AB_Tuning_15k", PROBLEM_NAME, training, validation, 5, 150, False)
        TunerTraining(Model_LMU_RE_Tuning, "LMU_RE_Tuning_15k", PROBLEM_NAME, training, validation, 5, 150,True)
        TunerTraining(Model_LMU_ESN_Tuning, "LMU_ESN_Tuning_15k", PROBLEM_NAME, training, validation, 5,150,True)
        TunerTraining(Model_LRMU_Tuning, "LRMU_Tuning_15k", PROBLEM_NAME, training, validation,5, 150,True)
