from Utility.PlotUtil import *
from Utility.ModelUtil import *
from ESN.layer import *
from Utility.LRMUHyperModel import LRMUHyperModel
from Utility.LRMUModelBuilder import LRMUModelBuilder
from tensorflow.keras.layers import SimpleRNNCell
from Problems.psMNIST.DataGeneration import psMNISTDataset

PROBLEM_NAME = "psMNIST"
CLASS_NUMBER = 10
SEQUENCE_LENGTH = 784


def LMU_ESN_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, "LMU_ESN")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(32, 24, 192, False,
                             ReservoirCell(160, spectral_radius=1.05, leaky=0.5),
                             False, True, False, False,
                             1, 1, 1, 1, 1)
    return LRMUBuilder.outputClassification(CLASS_NUMBER).composeModel().buildClassification()


def LMU_RE_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, "LMU_RE")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(2, 48, 256, True,
                             SimpleRNNCell(288, kernel_initializer=keras.initializers.GlorotUniform),
                             False, True, True, False,
                             2.0, 0.75, 2.0, 1.75, 1)
    return LRMUBuilder.outputClassification(CLASS_NUMBER).composeModel().buildClassification()


def LRMU_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, "LRMU")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(8, 64, 64, True,
                             ReservoirCell(320, spectral_radius=1.1, leaky=0.70),
                             False, True, True, False,
                             1.75, 2, 1.0, 1.75, 1)
    return LRMUBuilder.outputClassification(CLASS_NUMBER).composeModel().buildClassification()


def ModelEvaluation(model, testName, training, test):
    try:
        history, result = EvaluateModel(model, testName, training, test, 64, 10, "val_accuracy")
    except:
        return
    print(f"Test loss: {result[0]}")
    print(f"Test accuracy: {result[1]}")
    PlotModelAccuracy(history, f"{PROBLEM_NAME}_{testName}", f"./plots/{PROBLEM_NAME}", f"{testName}")


def RunEvaluation():
    training, validation, test = psMNISTDataset(True, 0)

    ModelEvaluation(LMU_ESN_BestModel, "LMU_ESN", training, test)
    ModelEvaluation(LMU_RE_BestModel, "LMU_RE", training, test)
    ModelEvaluation(LRMU_BestModel, "LRMU", training, test)


def RunTuning(dataPartition=10000):
    training, validation, test = psMNISTDataset(True, 0.1, dataPartition)

    hyperModels = LRMUHyperModel("psMNIST-hyperModel", PROBLEM_NAME, SEQUENCE_LENGTH, CLASS_NUMBER)

    TunerTraining(hyperModels.LMU_ESN(), "LMU_ESN_Tuning_15k", PROBLEM_NAME, training, validation, 5, 150, False)
    TunerTraining(hyperModels.LMU_RE(), "LMU_RE_Tuning_15k", PROBLEM_NAME, training, validation, 5, 150, False)
    TunerTraining(hyperModels.LRMU(), "LRMU_Tuning_15k", PROBLEM_NAME, training, validation, 5, 150, False)
