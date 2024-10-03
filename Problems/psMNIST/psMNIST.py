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
    Builder.inputLayer(SEQUENCE_LENGTH, Flatten=True)
    Builder.FF_Baseline()
    return Builder.BuildClassification(CLASS_NUMBER, useCategorical)


def LMU(isCategorical):
    Builder = ModelBuilder("LMU", PROBLEM_NAME)
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(1, 256, SEQUENCE_LENGTH,
                SimpleRNNCell(units=212), False,
                False, False, True, False, 1)
    return Builder.BuildClassification(CLASS_NUMBER, isCategorical)


#Final model score 0.40814894437789917
def LMU_ESN(isCategorical):
    Builder = ModelBuilder("LMU_ESN", PROBLEM_NAME)
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(1, 256, SEQUENCE_LENGTH,
                ReservoirCell(212, spectral_radius=0.8, leaky=0.7, input_scaling=1.75, bias_scaling=0.5), False,
                False, False, True, False,
                1)
    return Builder.BuildClassification(CLASS_NUMBER, isCategorical)


#Final model score 0.21
def LRMU(isCategorical):
    Builder = ModelBuilder("LRMU", PROBLEM_NAME)
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LRMU(1, 256, SEQUENCE_LENGTH,
                 SimpleRNNCell(212, kernel_initializer=keras.initializers.GlorotUniform),
                 False, False, True, False,
                 None, None, 2, None,
                 1)
    return Builder.BuildClassification(CLASS_NUMBER, isCategorical)


#Final model score 0.437406986951828
def LRMU_ESN(isCategorical):
    Builder = ModelBuilder("LRMU_ESN", PROBLEM_NAME)
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LRMU(1, 256, SEQUENCE_LENGTH,
                 ReservoirCell(212, spectral_radius=0.8, leaky=0.9, input_scaling=2.0, bias_scaling=1.75),
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
        model.custom_compile([keras.metrics.Accuracy(), keras.metrics.Accuracy()])

    return model


def RunEvaluation(batchSize: int = 64, epochs: int = 10, isCategorical: bool = True):
    dataSet = psMNISTDataset(True, 0.1)
    if isCategorical:
        dataSet.ToCategoricalLabel()
        monitorStat = "val_categorical_accuracy"
    else:
        monitorStat = "val_accuracy"

    dataSet.PrintSplit()

    saveDir = f"{DATA_DIR}/{PROBLEM_NAME}/Final_3"
    ModelEvaluation(FF_BaseLine(isCategorical), f"FF_Baseline", saveDir, dataSet, batchSize, epochs, monitorStat)
    ModelEvaluation(LRMU_ESN_RC(isCategorical), f"LRMU_ESN_RC", saveDir, dataSet, batchSize, epochs, monitorStat)
    ModelEvaluation(LRMU_ESN(isCategorical), "LRMU_ESN", saveDir, dataSet, batchSize, epochs, monitorStat)
    ModelEvaluation(LMU_ESN(isCategorical), "LMU_ESN", saveDir, dataSet, batchSize, epochs, monitorStat)
    ModelEvaluation(LRMU(isCategorical), "LRMU", saveDir, dataSet, batchSize, epochs, monitorStat)
    ModelEvaluation(LMU(isCategorical), "LMU", saveDir, dataSet, batchSize, epochs, monitorStat)


def RunTuning(dataPartition, isCategorical, epochs, max_trial):
    if 5000 > dataPartition > 60000:
        raise ValueError("Data partition must be between 5k and 60k")
    lengthName = f"{str(dataPartition)[0:1]}k"
    tuningName = f"{lengthName}_Final"
    dataSet = psMNISTDataset(True, 0.1, dataPartition)
    if isCategorical:
        dataSet.ToCategoricalLabel()

    dataSet.PrintSplit()

    hyperModels = HyperModel("psMNIST-hyperModel", PROBLEM_NAME, SEQUENCE_LENGTH, None, 0)
    hyperModels.SetUpClassification(CLASS_NUMBER, isCategorical)
    hyperModels.ForceLMUParam(1, 1, 256, SEQUENCE_LENGTH, 212).ForceConnection(False, False, True, False)
    TunerTraining(hyperModels.LRMU_ESN(), f"LRMU_ESN", tuningName, PROBLEM_NAME, dataSet, epochs,
                  max_trial, True)
    TunerTraining(hyperModels.LMU_ESN(), f"LMU_ESN", tuningName, PROBLEM_NAME, dataSet, epochs,
                  max_trial, True)
    TunerTraining(hyperModels.LRMU(), f"LRMU", tuningName, PROBLEM_NAME, dataSet, epochs,
                  max_trial, True)


def PrintSummary(isCategorical=True):
    FF_BaseLine(isCategorical)
    LRMU_ESN_RC(isCategorical)
    LRMU_ESN(isCategorical)
    LMU_ESN(isCategorical)
    LRMU(isCategorical)
    LMU(isCategorical)
