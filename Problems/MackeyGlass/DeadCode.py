from Problems.MackeyGlass.Config import *
from Problems.MackeyGlass.DataGeneration import *
from Utility.ModelUtil import *
from Utility.PlotUtil import *
from ESN.layer import *
from Utility.HyperModel import HyperModel
from Utility.ModelBuilder import ModelBuilder
from tensorflow.keras.layers import SimpleRNNCell
from GlobalConfig import *
import tensorflow.keras as keras


def LMU_ESN_BestModel(tau=17, activation="relu"):
    Builder = ModelBuilder(PROBLEM_NAME, f"LMU_ESN_T{tau}")
    Builder.inputLayer(SEQUENCE_LENGTH)
    if tau == 17:
        Builder.LMU(4, 20, 56, ReservoirCell(240, spectral_radius=0.95, leaky=0.7), False,
                    False, True, False, True, 1)
    elif tau == 30:
        Builder.LMU(8, 8, 16, ReservoirCell(288, spectral_radius=0.95, leaky=0.9), False,
                    True, True, False, False, 1)

    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


def LRMU_BestModel(tau=17, activation="relu"):
    Builder = ModelBuilder(PROBLEM_NAME, f"LRMU_T{tau}")
    Builder.inputLayer(SEQUENCE_LENGTH)
    if tau == 17:
        Builder.LRMU(1, 12, 64, SimpleRNNCell(144, kernel_initializer=keras.initializers.GlorotUniform),
                     False, False, True, False,
                     1.5, 1, 1, 1.75, 1)
    elif tau == 30:
        Builder.LRMU(2, 24, 52, SimpleRNNCell(192, kernel_initializer=keras.initializers.GlorotUniform),
                     False, False, True, True,
                     None, None, 1.75, 1.25, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


def LRMU_ESN_BestModel(tau=17, activation="relu"):
    Builder = ModelBuilder(PROBLEM_NAME, f"LRMU_ESN_T{tau}")
    Builder.inputLayer(SEQUENCE_LENGTH)
    if tau == 17:
        Builder.LRMU(4, 1, 144, ReservoirCell(320, spectral_radius=1, leaky=0.5),
                     False, False, True, True,
                     None, None, 0.75, 1.5, 1)
    elif tau == 30:
        Builder.LRMU(1, 24, 48, ReservoirCell(80, spectral_radius=1, leaky=0.5),
                     False, False, True, True,
                     None, None, 1.5, 1.75, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)
