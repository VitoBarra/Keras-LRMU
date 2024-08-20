from Utility.DataUtil import *
from Utility.ModelUtil import *
from Utility.PlotUtil import *
import Problems.MackeyGlass.DataGeneration as dg
from ESN.layer import *
from Utility.LRMUHyperModel import LRMUHyperModel
from Utility.LRMUModelBuilder import LRMUModelBuilder
from keras.layers import SimpleRNNCell


PROBLEM_NAME = "Mackey-Glass"
SEQUENCE_LENGTH = 5000


def LMU_ESN_T17_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, f"{PROBLEM_NAME}_LMU_ESN")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(1, 12, 80, False,
                             ReservoirCell(192, spectral_radius=1.1),
                             True, True, True, True,
                             1, 1, 1, 1, 1)
    return LRMUBuilder.outputPrediction(1).composeModel().buildPrediction()


def LMU_RE_T17_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, f"{PROBLEM_NAME}_LMU_RE")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(4, 4, 160, True,
                             SimpleRNNCell(16, kernel_initializer=keras.initializers.GlorotUniform),
                             False, False, False, True,
                             1.25, 1, 1.5, 0.5, 1)
    return LRMUBuilder.outputPrediction(1).composeModel().buildPrediction()


def LRMU_T17_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, f"{PROBLEM_NAME}_LRMU")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(1, 1, 256, True,
                             ReservoirCell(144, spectral_radius=0.85),
                             False, False, True, True,
                             0.5, 1, 0.5, 1.75, 1)
    return LRMUBuilder.outputPrediction(1).composeModel().buildPrediction()

def LMU_ESN_T30_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, f"{PROBLEM_NAME}_LMU_ESN")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(1, 4, 64, False,
                             ReservoirCell(80, spectral_radius=0.99),
                             False, True, False, True,
                             1, 1, 1, 1, 1)
    return LRMUBuilder.outputPrediction(1).composeModel().buildPrediction()


def LMU_RE_T30_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, f"{PROBLEM_NAME}_LMU_RE")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(8, 1, 112, True,
                             SimpleRNNCell(16, kernel_initializer=keras.initializers.GlorotUniform),
                             False, True, True, True,
                             1.25, 2, 1.75, 0.75, 1)
    return LRMUBuilder.outputPrediction(1).composeModel().buildPrediction()


def LRMU_T30_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, f"{PROBLEM_NAME}_LRMU")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(32, 1, 80, True,
                             ReservoirCell(128, spectral_radius=0.9,leaky=0.8),
                             True, False, False, True,
                             0.5, 1.25, 1.0, 1.25, 1)
    return LRMUBuilder.outputPrediction(1).composeModel().buildPrediction()



def ModelEvaluation(model, training, test,testName):
    history, result = EvaluateModel(model, training, test, 64, 15, "val_loss")

    print("test loss:", result[0])
    print("test mse:", result[1])

    PlotModelLoss(history, f"{PROBLEM_NAME}_{testName}", f"./plots/{PROBLEM_NAME}", f"{testName}")


def Run(modelEvaluation=True, tau=17, sequenceLenght=5000):
    SEQUENCE_LENGTH = sequenceLenght

    lengthName = f"{str(sequenceLenght)[0]}k"

    data, label = dg.generate_data(128, SEQUENCE_LENGTH, 0, 15, tau)
    dataset = DataLabel(data, label)
    training, validation, test = dataset.SplitDataset(0.1, 0.1)

    if modelEvaluation:
        training.Concatenate(validation)
        ModelEvaluation(LMU_ESN_BestModel, training, test,"LMU_ESN")
        ModelEvaluation(LMU_RE_BestModel, training, test,"LMU_RE")
        ModelEvaluation(LRMU_BestModel, training, test,"LRMU")
    else:
        hyperModels = LRMUHyperModel("MackeyGlass-hyperModel", PROBLEM_NAME, SEQUENCE_LENGTH)
        TunerTraining(hyperModels.LMU_ESN(), f"LMU_ESN_Tuning_{lengthName}_T{tau}", PROBLEM_NAME, training, validation,
                      5, 150, False)
        TunerTraining(hyperModels.LMU_RE(), f"LMU_RE_Tuning_{lengthName}_T{tau}", PROBLEM_NAME, training, validation, 5,
                      150, False)
        TunerTraining(hyperModels.LRMU(), f"LRMU_Tuning_{lengthName}_T{tau}", PROBLEM_NAME, training, validation, 5,
                      150, True)
