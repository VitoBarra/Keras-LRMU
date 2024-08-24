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


def ModelEvaluation(model, testName, training, test, batchSize=64, epochs=10):
    try:
        history, result = EvaluateModel(model, testName, training, test, batchSize, epochs, "accuracy")
    except:
        return
    # Serializing json

    print(f"Test loss: {result[0]}")
    print(f"Test accuracy: {result[1]}")
    SaveDataForPlotJson("./plots", PROBLEM_NAME, testName, history, result)


def RunEvaluation(batchSize=64, epochs=10):
    training, validation, test = psMNISTDataset(True, 0)

    ModelEvaluation(LMU_ESN_BestModel, "LMU_ESN_60k", training, test, batchSize, epochs)
    ModelEvaluation(LMU_RE_BestModel, "LMU_RE_60k", training, test, batchSize, epochs)
    ModelEvaluation(LRMU_BestModel, "LRMU_60k", training, test, batchSize, epochs)


def RunTuning(dataPartition=10000, max_trial=50):
    training, validation, test = psMNISTDataset(True, 0.1, dataPartition)

    hyperModels = LRMUHyperModel("psMNIST-hyperModel", PROBLEM_NAME, SEQUENCE_LENGTH, CLASS_NUMBER)

    TunerTraining(hyperModels.LMU_ESN(), "LMU_ESN_Tuning_15k", PROBLEM_NAME, training, validation, 5, max_trial, False)
    TunerTraining(hyperModels.LMU_RE(), "LMU_RE_Tuning_15k", PROBLEM_NAME, training, validation, 5, max_trial, False)
    TunerTraining(hyperModels.LRMU(), "LRMU_Tuning_15k", PROBLEM_NAME, training, validation, 5, max_trial, False)


def PlotAll():
    ReadAndPlot("./plots", PROBLEM_NAME, "LMU_ESN_60k", True)
    ReadAndPlot("./plots", PROBLEM_NAME, "LMU_RE_60k", True)
    ReadAndPlot("./plots", PROBLEM_NAME, "LRMU_60k", True)
