from tensorflow.keras.layers import SimpleRNNCell
from Reservoir.layer import *
from Problems.psMNIST.Config import *
from Utility.ModelBuilder import ModelBuilder


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