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


def LMU_ESN_comp():
    Builder = ModelBuilder(PROBLEM_NAME, "LMU_ESN")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(1, 256, SEQUENCE_LENGTH, ReservoirCell(212, spectral_radius=0.89, leaky=0.9,input_scaling=1), False,
                False, False, True, False,
                1)
    return Builder.BuildClassification(CLASS_NUMBER)


def LRMU_comp():
    Builder = ModelBuilder(PROBLEM_NAME, "LRMU")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LRMU(1, 256, SEQUENCE_LENGTH, SimpleRNNCell(212, kernel_initializer=keras.initializers.GlorotUniform),
                 False, False, True, False,
                 None, None, 1, None,
                 1)
    return Builder.BuildClassification(CLASS_NUMBER)


def LRMU_ESN_comp():
    Builder = ModelBuilder(PROBLEM_NAME, "LRMU_ESN")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LRMU(1, 256, SEQUENCE_LENGTH, ReservoirCell(212, spectral_radius=0.87, leaky=0.9,input_scaling=1),
                 False, False, True, False,
                 None, None, 2.0, None,
                 1)
    return Builder.BuildClassification(CLASS_NUMBER)


def RunEvaluation(batchSize=64, epochs=10):
    dataSet = psMNISTDataset(True, 0.1)
    dataSet.ToCategoricalLabel()
    saveDir = f"{DATA_DIR}/{PROBLEM_NAME}/comp"

    ModelEvaluation(LRMU_ESN_comp(), "LRMU_ESN_comp", saveDir, dataSet, batchSize, epochs, "val_accuracy")
    ModelEvaluation(LMU_ESN_comp(), "LMU_ESN_comp", saveDir, dataSet, batchSize, epochs, "val_accuracy")
    ModelEvaluation(LRMU_comp(), "LRMU_comp", saveDir, dataSet, batchSize, epochs, "val_accuracy")


def RunTuning(dataPartition, max_trial, epochs):
    if 10000 > dataPartition > 50000:
        raise ValueError("Data partition must be between 10k and 50k")
    lengthName = f"{str(dataPartition)[0:2]}k"
    dataSet = psMNISTDataset(True, 0.1, dataPartition)
    dataSet.ToCategoricalLabel()

    hyperModels = HyperModel("psMNIST-hyperModel", PROBLEM_NAME, SEQUENCE_LENGTH, 0).SetUpClassification(CLASS_NUMBER)
    hyperModels.ForceLMUParam(1, 1, 256, SEQUENCE_LENGTH, 212).ForceConnection(False, False, True, False)
    # TunerTraining(hyperModels.LRMU_ESN(), f"LRMU_ESN_Tuning_{lengthName}k_parSet", PROBLEM_NAME, dataSet, epochs,
    #               max_trial, True)
    TunerTraining(hyperModels.LMU_ESN(), f"LMU_ESN_Tuning_{lengthName}k_parSet", PROBLEM_NAME, dataSet, epochs,
                  max_trial, True)
    TunerTraining(hyperModels.LRMU(), f"LRMU_Tuning_{lengthName}k_parSet", PROBLEM_NAME, dataSet, epochs,
                  max_trial, True)
