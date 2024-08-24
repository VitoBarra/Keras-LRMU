import numpy as np

from Utility.ModelUtil import *
from Utility.PlotUtil import *
from Problems.MackeyGlass.DataGeneration import MackeyGlassDataset
from ESN.layer import *
from Utility.LRMUHyperModel import LRMUHyperModel
from Utility.LRMUModelBuilder import LRMUModelBuilder
from tensorflow.keras.layers import SimpleRNNCell

PROBLEM_NAME = "Mackey-Glass"
SEQUENCE_LENGTH = 5000


def LMU_ESN_T17_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, "LMU_ESN")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(32, 1, 112, False,
                             ReservoirCell(64, spectral_radius=0.9,leaky=0.9),
                             True, True, False, False,
                             1, 1, 1, 1, 1)
    return LRMUBuilder.outputPrediction(1).composeModel().buildPrediction()


def LMU_RE_T17_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, "LMU_RE")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(4, 8, 240, True,
                             SimpleRNNCell(64, kernel_initializer=keras.initializers.GlorotUniform),
                             False, False, False, False,
                             1.5, 1, 1.5, 0.5, 1)
    return LRMUBuilder.outputPrediction(1).composeModel().buildPrediction()


def LRMU_T17_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, "LRMU")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(4, 1, 144, True,
                             ReservoirCell(320, spectral_radius=1.25,leaky=0.5),
                             True, False, True, True,
                             0.5, 1.5, 0.75, 2, 1)
    return LRMUBuilder.outputPrediction(1).composeModel().buildPrediction()


def LMU_ESN_T30_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, "LMU_ESN")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(1, 4, 64, False,
                             ReservoirCell(80, spectral_radius=0.99),
                             False, True, False, True,
                             1, 1, 1, 1, 1)
    return LRMUBuilder.outputPrediction(1).composeModel().buildPrediction()


def LMU_RE_T30_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, "LMU_RE")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(8, 1, 112, True,
                             SimpleRNNCell(16, kernel_initializer=keras.initializers.GlorotUniform),
                             False, True, True, True,
                             1.25, 2, 1.75, 0.75, 1)
    return LRMUBuilder.outputPrediction(1).composeModel().buildPrediction()


def LRMU_T30_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, "LRMU")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(32, 1, 80, True,
                             ReservoirCell(128, spectral_radius=0.9, leaky=0.8),
                             True, False, False, True,
                             0.5, 1.25, 1.0, 1.25, 1)
    return LRMUBuilder.outputPrediction(1).composeModel().buildPrediction()


def ModelEvaluation(model, testName, training, test,batchSize=64,epochs=15):
    try:
        history, result = EvaluateModel(model, testName, training, test, batchSize, epochs, "loss")
    except:
        print("Error during model evaluation")
        raise
    print("test loss:", result[0])
    print("test mse:", result[1])

    try:
        SaveDataForPlotJson("./plots",PROBLEM_NAME,testName,history,result)
    except:
        print("something went wrong during plot data saving ")
        raise

def RunEvaluation(sample=128, sequenceLength=5000, tau=17,batchSize=64,epochs=15):
    SEQUENCE_LENGTH = sequenceLength
    lengthName = f"{str(sequenceLength)[0]}k"
    training, validation, test = (
        MackeyGlassDataset(0.1, 0.1, sample, SEQUENCE_LENGTH, 15, tau, 0))

    training.Concatenate(validation)
    if tau == 17:
        ModelEvaluation(LMU_ESN_T17_BestModel, f"LMU_ESN_{sample}_{lengthName}_T17", training, test,batchSize,epochs)
        ModelEvaluation(LMU_RE_T17_BestModel, f"LMU_RE_{sample}_{lengthName}_T17", training, test,batchSize,epochs)
        ModelEvaluation(LRMU_T17_BestModel, f"LRMU_{sample}_{lengthName}_T17", training, test,batchSize,epochs)
    elif tau == 30:
        ModelEvaluation(LMU_ESN_T30_BestModel, f"LMU_ESN_{sample}_{lengthName}_T30", training, test,batchSize,epochs)
        ModelEvaluation(LMU_RE_T30_BestModel, f"LMU_RE_{sample}_{lengthName}_T30", training, test,batchSize,epochs)
        ModelEvaluation(LRMU_T30_BestModel, f"LRMU_{sample}_{lengthName}_T30", training, test,batchSize,epochs)


def RunTuning(sample=128, sequenceLength=5000, tau=17,max_trial=50):
    SEQUENCE_LENGTH = sequenceLength
    lengthName = f"{str(sequenceLength)[0]}k"
    training, validation, test = (
        MackeyGlassDataset(0.1, 0.1, sample, SEQUENCE_LENGTH, 15, tau, 0))

    hyperModels = LRMUHyperModel("MackeyGlass-hyperModel", PROBLEM_NAME, SEQUENCE_LENGTH)
    #TunerTraining(hyperModels.LMU_ESN(), f"LMU_ESN_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, training,
    #             validation,5, max_trial, False)
    #TunerTraining(hyperModels.LMU_RE(), f"LMU_RE_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, training,
    #           validation, 5,max_trial, False)
    TunerTraining(hyperModels.LRMU(), f"LRMU_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, training, validation,
                  5,
                  max_trial, True)




def PlotAll(sample=128,sequenceLength=5000,tau=17):
    lengthName = f"{str(sequenceLength)[0]}k"
    ReadAndPlot("./plots", PROBLEM_NAME, f"LMU_ESN_{sample}_{lengthName}_T{tau}", True)
    ReadAndPlot("./plots", PROBLEM_NAME, f"LMU_RE_{sample}_{lengthName}_T{tau}", True)
    ReadAndPlot("./plots", PROBLEM_NAME, f"LRMU_{sample}_{lengthName}_T{tau}", True)
