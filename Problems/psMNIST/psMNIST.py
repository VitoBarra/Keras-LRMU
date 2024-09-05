from Problems.psMNIST.Config import *
from Problems.psMNIST.DataGeneration import *
import LMU.layers
from Utility.PlotUtil import *
from Utility.ModelUtil import *
from ESN.layer import *
from Utility.HyperModel import HyperModel
from Utility.ModelBuilder import ModelBuilder
import tensorflow.keras as ks
from tensorflow.keras.layers import SimpleRNNCell, Dense
from GlobalConfig import *


def LMU_Original():
    Builder = ModelBuilder(PROBLEM_NAME, "LMU")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(1, 256, SEQUENCE_LENGTH, SimpleRNNCell(units=212), False,
                False, False, True, False, 1)
    return Builder.BuildClassification(CLASS_NUMBER)


def LMU_ESN_BestModel():
    Builder = ModelBuilder(PROBLEM_NAME, "LMU_ESN")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(32, 24, SEQUENCE_LENGTH, ReservoirCell(160, spectral_radius=1.05, leaky=1), False,
                True, True, False, False, 1)
    return Builder.BuildClassification(CLASS_NUMBER)


def LRMU_BestModel():
    Builder = ModelBuilder(PROBLEM_NAME, "LRMU")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LRMU(2, 48, SEQUENCE_LENGTH, SimpleRNNCell(288, kernel_initializer=keras.initializers.GlorotUniform),
                 False, True, True, False,
                 2.0, 0.75, 2.0, 1.75, 1)
    return Builder.BuildClassification(CLASS_NUMBER)

def LRMU_ESN_BestModel():
    Builder = ModelBuilder(PROBLEM_NAME, "LRMU_ESN")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LRMU(8, 64, SEQUENCE_LENGTH, ReservoirCell(320, spectral_radius=1.1, leaky=1),
                 False, True, True, False,
                 1.75, 2, 1.0, 1.75, 1)
    return Builder.BuildClassification(CLASS_NUMBER)


def LMU_ESN_comp():
    Builder = ModelBuilder(PROBLEM_NAME, "LMU_ESN")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(1, 256, SEQUENCE_LENGTH, ReservoirCell(212, spectral_radius=1.05, leaky=1), False,
                False, False, True, False,
                1)
    return Builder.BuildClassification(CLASS_NUMBER)


def LRMU_comp():
    Builder = ModelBuilder(PROBLEM_NAME, "LRMU")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LRMU(1, 256, SEQUENCE_LENGTH, SimpleRNNCell(212, kernel_initializer=keras.initializers.GlorotUniform),
                 False, False, True, False,
                 1, 1, 1, 1,
                 1)
    return Builder.BuildClassification(CLASS_NUMBER)


def LRMU_ESN_comp():
    Builder = ModelBuilder(PROBLEM_NAME, "LRMU_ESN")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LRMU(1, 256, SEQUENCE_LENGTH, ReservoirCell(212, spectral_radius=1.1, leaky=1),
                 False, False, True, False,
                 1, 1, 1, 1, 1)
    return Builder.BuildClassification(CLASS_NUMBER)


def ModelEvaluation(model, testName, training, test, batchSize=64, epochs=10):
    try:
        history, result = EvaluateModel(model, testName, training, test, batchSize, epochs, "accuracy")
    except Exception as e:
        print(f"exception during evaluation: {e} ")
        return
    print(f"total training time: {sum(history.history['time'])}s", )
    print(f"Test loss: {result[0]}")
    print(f"Test accuracy: {result[1]}")
    SaveDataForPlotJson(PLOTS_DIR, PROBLEM_NAME, testName, history, result)


def RunEvaluationSaved():
    model = ks.models.load_model("./logs/bestModel/LMU_ESN_50k.h5")


def RunEvaluation(batchSize=64, epochs=10):
    training, validation, test = psMNISTDataset(True, 0)
    training.ToCategoricalLabel()
    test.ToCategoricalLabel()

    if validation is not None:
        validation.ToCategoricalLabel()
        training.Concatenate(validation)

    #ModelEvaluation(LMU_Original, "LMU_50k", training, test, batchSize, epochs)

    # ModelEvaluation(LMU_ESN_BestModel, "LMU_ESN_50k", training, test, batchSize, epochs)
    # ModelEvaluation(LRMU_BestModel, "LRMU_50k", training, test, batchSize, epochs)
    # ModelEvaluation(LRMU_ESN_BestModel, "LRMU_ESN_50k", training, test, batchSize, epochs)

    ModelEvaluation(LMU_ESN_comp, "LMU_ESN_50k_comp", training, test, batchSize, epochs)
    ModelEvaluation(LRMU_comp, "LRMU_50k_comp", training, test, batchSize, epochs)
    ModelEvaluation(LRMU_ESN_comp, "LRMU_ESN_50k_comp", training, test, batchSize, epochs)


def RunTuning(dataPartition=10000, max_trial=50):
    lengthName = f"{str(dataPartition)[0:1]}k"
    training, validation, test = psMNISTDataset(True, 0.1, dataPartition)
    training.ToCategoricalLabel()
    validation.ToCategoricalLabel()
    test.ToCategoricalLabel()

    hyperModels = HyperModel("psMNIST-hyperModel", PROBLEM_NAME, SEQUENCE_LENGTH, CLASS_NUMBER, False, False)

    TunerTraining(hyperModels.LMU_ESN(), f"LMU_ESN_Tuning_{lengthName}k", PROBLEM_NAME, training, validation, 5, max_trial, True)
    TunerTraining(hyperModels.LRMU(), f"LRMU_Tuning_{lengthName}k", PROBLEM_NAME, training, validation, 5, max_trial, True)
    TunerTraining(hyperModels.LRMU_ESN(), f"LRMU_ESN_Tuning_{lengthName}k", PROBLEM_NAME, training, validation, 5, max_trial, True)
