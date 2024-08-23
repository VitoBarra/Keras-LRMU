from Utility.DataUtil import *
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
    LRMUBuilder.featureLayer(1, 12, 80, False,
                             ReservoirCell(192, spectral_radius=1.1),
                             True, True, True, True,
                             1, 1, 1, 1, 1)
    return LRMUBuilder.outputPrediction(1).composeModel().buildPrediction()


def LMU_RE_T17_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, "LMU_RE")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(4, 4, 160, True,
                             SimpleRNNCell(16, kernel_initializer=keras.initializers.GlorotUniform),
                             False, False, False, True,
                             1.25, 1, 1.5, 0.5, 1)
    return LRMUBuilder.outputPrediction(1).composeModel().buildPrediction()


def LRMU_T17_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, "LRMU")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(1, 1, 256, True,
                             ReservoirCell(144, spectral_radius=0.85),
                             False, False, True, True,
                             0.5, 1, 0.5, 1.75, 1)
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


def ModelEvaluation(model, testName, training, test):
    try:
        history, result = EvaluateModel(model, testName, training, test, 64, 15, "val_loss")
    except:
        return
    print("test loss:", result[0])
    print("test mse:", result[1])

    PlotModelLoss(history, f"{PROBLEM_NAME}_{testName}", f"./plots/{PROBLEM_NAME}", f"{testName}")


def RunEvaluation(sample=128, sequenceLenght=5000, tau=17):
    SEQUENCE_LENGTH = sequenceLenght
    lengthName = f"{str(sequenceLenght)[0]}k"
    training, validation, test = (
        MackeyGlassDataset(0, 0.1, sample, SEQUENCE_LENGTH, 15, tau, 0))

    if tau == 17:
        ModelEvaluation(LMU_ESN_T17_BestModel, f"LMU_ESN_{sample}_{lengthName}_T17", training, test)
        ModelEvaluation(LMU_RE_T17_BestModel, f"LMU_RE_{sample}_{lengthName}_T17", training, test)
        ModelEvaluation(LRMU_T17_BestModel, f"LRMU_{sample}_{lengthName}_T17", training, test)
    elif tau == 30:
        ModelEvaluation(LMU_ESN_T30_BestModel, f"LMU_ESN_{sample}_{lengthName}_T30", training, test)
        ModelEvaluation(LMU_RE_T30_BestModel, f"LMU_RE_{sample}_{lengthName}_T30", training, test)
        ModelEvaluation(LRMU_T30_BestModel, f"LRMU_{sample}_{lengthName}_T30", training, test)


def RunTuning(sample=128, sequenceLenght=5000, tau=17):
    SEQUENCE_LENGTH = sequenceLenght
    lengthName = f"{str(sequenceLenght)[0]}k"
    training, validation, test = (
        MackeyGlassDataset(0.1, 0.1, sample, SEQUENCE_LENGTH, 15, tau, 0))

    hyperModels = LRMUHyperModel("MackeyGlass-hyperModel", PROBLEM_NAME, SEQUENCE_LENGTH)
    TunerTraining(hyperModels.LMU_ESN(), f"LMU_ESN_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, training,
                  validation,
                  5, 150, False)
    TunerTraining(hyperModels.LMU_RE(), f"LMU_RE_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, training,
                  validation, 5,
                  150, False)
    TunerTraining(hyperModels.LRMU(), f"LRMU_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, training, validation,
                  5,
                  150, False)
