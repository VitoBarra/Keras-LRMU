import keras

from Problems.psMNIST.Config import *
from Problems.psMNIST.DataGeneration import *
from Utility.ModelUtil import *
from Reservoir.layer import *
from Utility.HyperModel import HyperModel
from Utility.ModelBuilder import ModelBuilder
from tensorflow.keras.layers import SimpleRNNCell
from GlobalConfig import *
from LRMU.Model import LRMU_ESN_Ridge
from LRMU.utility import ModelType
import tensorflow as tf



def FF_BaseLine(useCategorical):
    Builder = ModelBuilder("FF_Baseline", PROBLEM_NAME)
    Builder.inputLayer(SEQUENCE_LENGTH,Flatten=True)
    Builder.FF_Baseline()
    return Builder.BuildClassification(CLASS_NUMBER, useCategorical)


def LMU(isCategorical):
    Builder = ModelBuilder("LMU", PROBLEM_NAME)
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(1, 256, SEQUENCE_LENGTH,
                SimpleRNNCell(units=212), False,
                False, False, True, False, 1)
    return Builder.BuildClassification(CLASS_NUMBER, isCategorical)


def LMU_ESN(isCategorical):
    Builder = ModelBuilder("LMU_ESN", PROBLEM_NAME)
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(1, 256, SEQUENCE_LENGTH,
                ReservoirCell(212, spectral_radius=0.99, leaky=0.8, input_scaling=1.75, bias_scaling=1.0), False,
                False, False, True, False,
                1)
    return Builder.BuildClassification(CLASS_NUMBER, isCategorical)


def LRMU(isCategorical):
    Builder = ModelBuilder("LRMU", PROBLEM_NAME)
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LRMU(1, 256, SEQUENCE_LENGTH,
                 SimpleRNNCell(212, kernel_initializer=keras.initializers.GlorotUniform),
                 False, False, True, False,
                 None, None, 1.75, None,
                 1)
    return Builder.BuildClassification(CLASS_NUMBER, isCategorical)


def LRMU_ESN(isCategorical):
    Builder = ModelBuilder("LRMU_ESN", PROBLEM_NAME)
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LRMU(1, 256, SEQUENCE_LENGTH,
                 ReservoirCell(212, spectral_radius=0.87, leaky=0.9, input_scaling=1, bias_scaling=1.0),
                 False, False, True, False,
                 None, None, 2.0, None,
                 1)
    return Builder.BuildClassification(CLASS_NUMBER, isCategorical)


def LRMU_ESN_RC(isCategorical):
    model = LRMU_ESN_Ridge(ModelType.Classification, SEQUENCE_LENGTH, 1, 256, SEQUENCE_LENGTH,
                           False, False, True, False,
                           None, None, 2.0, None,
                           212, tf.nn.tanh, 0.87, 0.9, 1, 1.0, )
    model.name = f"LRMU_ESN_RC_{PROBLEM_NAME}"
    model.summary()
    if isCategorical:
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["categorical_accuracy"])
        model.custom_compile([keras.losses.CategoricalCrossentropy(), keras.metrics.CategoricalAccuracy()])
    else:
        model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.custom_compile([keras.losses.SparseCategoricalCrossentropy(), keras.metrics.Accuracy()])

    return model


def RunEvaluation(batchSize=64, epochs=10, isCategorical=True):
    dataSet = psMNISTDataset(True, 0.1)
    if isCategorical:
        dataSet.ToCategoricalLabel()

    dataSet.PrintSplit()
    monitorStat = "val_categorical_accuracy"
    saveDir = f"{DATA_DIR}/{PROBLEM_NAME}/comp"
    ModelEvaluation(LRMU_ESN_RC(isCategorical), f"LRMU_ESN_RC", saveDir, dataSet, batchSize, epochs, monitorStat)
    # ModelEvaluation(LRMU_ESN(isCategorical), "LRMU_ESN", saveDir, dataSet, batchSize, epochs, monitorStat)
    # ModelEvaluation(LMU_ESN(isCategorical), "LMU_ESN", saveDir, dataSet, batchSize, epochs, monitorStat)
    # ModelEvaluation(LRMU(isCategorical), "LRMU", saveDir, dataSet, batchSize, epochs, monitorStat)


def RunTuning(dataPartition, isCategorical, epochs, max_trial):
    if 5000 > dataPartition > 60000:
        raise ValueError("Data partition must be between 5k and 60k")
    lengthName = f"{str(dataPartition)[0:1]}k"
    dataSet = psMNISTDataset(True, 0.1, dataPartition)
    if isCategorical:
        dataSet.ToCategoricalLabel()

    dataSet.PrintSplit()

    hyperModels = HyperModel("psMNIST-hyperModel", PROBLEM_NAME, SEQUENCE_LENGTH, 0).SetUpClassification(CLASS_NUMBER,isCategorical)
    hyperModels.ForceLMUParam(1, 1, 256, SEQUENCE_LENGTH, 212).ForceConnection(False, False, True, False)
    TunerTraining(hyperModels.LRMU_ESN(), f"LRMU_ESN", f"{lengthName}_Final", PROBLEM_NAME, dataSet, epochs,
                  max_trial, True)
    TunerTraining(hyperModels.LMU_ESN(), f"LMU_ESN", f"{lengthName}_Final", PROBLEM_NAME, dataSet, epochs,
                  max_trial, True)
    TunerTraining(hyperModels.LRMU(), f"LRMU", f"{lengthName}_Final", PROBLEM_NAME, dataSet, epochs,
                  max_trial, True)


def PrintSummary(isCategorical=True):
    FF_BaseLine(isCategorical)
    LRMU_ESN_RC(isCategorical)
    LRMU_ESN(isCategorical)
    LMU_ESN(isCategorical)
    LRMU(isCategorical)
    LMU(isCategorical)
