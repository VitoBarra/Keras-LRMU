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


def Model_LMU_AB_Tuning(hp):
    seed = 0
    layerN = 1
    memoryDim = hp.Choice("memoryDim", values=[1, 2, 4, 8, 16, 32, 48, 64])
    order = hp.Int("order", min_value=128, max_value=512, step=32)
    theta = SEQUENCE_LENGTH

    stepSize = 32
    hiddenUnit = hp.Int("hiddenUnit", min_value=stepSize, max_value=stepSize * 16, step=stepSize)
    spectraRadius = -1
    leaky = -1  # task step invariant so no need to change this parameter
    trainableAB = True

    reservoirMode = False
    hiddenCell = ks.layers.SimpleRNNCell(hiddenUnit, kernel_initializer=GlorotUniform(seed),
                                         recurrent_initializer=GlorotUniform(seed))

    memoryToMemory = hp.Boolean("memoryToMemory")
    hiddenToMemory = hp.Boolean("hiddenToMemory")
    inputToHiddenCell = hp.Boolean("inputToHiddenCell")
    useBias = hp.Boolean("useBias")

    memoryToMemoryScaler = -1
    hiddenToMemoryScaler = -1
    inputToHiddenCellScaler = -1
    biasScaler = -1

    return Model_LRMU_Classification(PROBLEM_NAME, "LMU-AB", SEQUENCE_LENGTH, CLASS_NUMBER,
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
    theta = SEQUENCE_LENGTH

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

    return Model_LRMU_Classification(PROBLEM_NAME, "LMU_ESN", SEQUENCE_LENGTH, CLASS_NUMBER,
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
    theta = SEQUENCE_LENGTH

    hidden_unit_stepSize = 32
    hiddenUnit = hp.Int("hiddenUnit", min_value=hidden_unit_stepSize, max_value=hidden_unit_stepSize * 16, step=hidden_unit_stepSize)
    spectraRadius = -1
    leaky = -1  # task step invariant so no need to change this parameter
    trainableAB = False

    reservoirMode = True
    hiddenCell = None

    memoryToMemory = hp.Boolean("memoryToMemory")
    hiddenToMemory = hp.Boolean("hiddenToMemory")
    inputToHiddenCell = hp.Boolean("inputToHiddenCell")
    useBias = hp.Boolean("useBias")

    memoryToMemoryScaler = hp.Float("memoryToMemoryScaler", min_value=0.5, max_value=2, step=0.25)
    hiddenToMemoryScaler = hp.Float("hiddenToMemoryScaler", min_value=0.5, max_value=2, step=0.25)
    inputToHiddenCellScaler = hp.Float("inputToHiddenCellScaler", min_value=0.5, max_value=2, step=0.25)
    biasScaler = hp.Float("biasScaler", min_value=0.5, max_value=2, step=0.25)

    return Model_LRMU_Classification(PROBLEM_NAME, "LMU-RE", SEQUENCE_LENGTH, CLASS_NUMBER,
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
    theta = SEQUENCE_LENGTH

    hidden_unit_stepSize = 32
    hiddenUnit = hp.Int("hiddenUnit", min_value=hidden_unit_stepSize, max_value=hidden_unit_stepSize * 16, step=hidden_unit_stepSize)
    spectraRadius = hp.Float("spectraRadius", min_value=0.8, max_value=1.3, step=0.05)
    leaky = 1  # task step invariant so no need to change this parameter
    trainableAB = False

    reservoirMode = True
    hiddenCell = None

    memoryToMemory = hp.Boolean("memoryToMemory")
    hiddenToMemory = hp.Boolean("hiddenToMemory")
    inputToHiddenCell = hp.Boolean("inputToHiddenCell")
    useBias = hp.Boolean("useBias")


    memoryToMemoryScaler = hp.Float("memoryToMemoryScaler", min_value=0.5, max_value=2, step=0.25)
    hiddenToMemoryScaler = hp.Float("hiddenToMemoryScaler", min_value=0.5, max_value=2, step=0.25)
    inputToHiddenCellScaler = hp.Float("inputToHiddenCellScaler", min_value=0.5, max_value=2, step=0.25)
    biasScaler = hp.Float("biasScaler", min_value=0.5, max_value=2, step=0.25)

    return Model_LRMU_Classification(PROBLEM_NAME, "LRMU", SEQUENCE_LENGTH, CLASS_NUMBER,
                                 memoryDim, order, theta, trainableAB,
                                 hiddenUnit, spectraRadius, leaky,
                                 reservoirMode, hiddenCell,
                                 memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                                 memoryToMemoryScaler, hiddenToMemoryScaler, inputToHiddenCellScaler, biasScaler,
                                 seed, layerN)


# 75, 352, 500, 1.15,True, None, False, True, False, True = 92.22% accuracy on test set xavierInitializer
# 75, 352, 500, 1.18,True, None, False, True, False, True = 91.88% accuracy on test set xavierInitializer
# 256, 128, 212, 1.15,True, None, False, True, False, True = 87.4% accuracy on test set xavierInitializer
# 256, 128, 212, 0.99,True, None, False, True, False, True = 87.4% accuracy on test set xavierInitializer
# 256, 256, 212, 0.99,True, None, False, True, False, True = 88.3% accuracy on test set xavierInitializer
# 256, 256, 212, 1.18,True, None, False, True, False, True = 88.55%  accuracy on test set xavierInitializer
def ModelLRMU_SelectedHP():
    return Model_LRMU_Classification(PROBLEM_NAME, "LRMU", SEQUENCE_LENGTH, 10,
                                     256, 212, 1.18, 1,
                                     False, 1, 1, True, None,
                                     True, False, False, False, True,
                                     1, 1, 1, 1, 0)


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
