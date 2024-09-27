from tensorflow.keras.layers import SimpleRNNCell
from Reservoir.layer import *
from Problems.MackeyGlass.Config import *
from Utility.ModelBuilder import ModelBuilder

def LMU_BestModel(tau, activation):
    Builder = ModelBuilder(f"LMU_{tau}", PROBLEM_NAME)
    Builder.inputLayer(SEQUENCE_LENGTH)
    if tau == 17:
        Builder.LMU(2, 1, 44, SimpleRNNCell(176), False,
                    True, False, False, False, 1)
    elif tau == 30:
        Builder.LMU(1, 1, 64, SimpleRNNCell(144), False,
                    True, False, False, False, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)

def LMU_ESN_BestModel(tau, activation):
    Builder = ModelBuilder(f"LMU_ESN_T{tau}", PROBLEM_NAME)
    Builder.inputLayer(SEQUENCE_LENGTH)
    if tau == 17:
        Builder.LMU(4, 20, 56, ReservoirCell(240, spectral_radius=0.95, leaky=0.7), False,
                    False, True, False, True, 1)
    elif tau == 30:
        Builder.LMU(8, 8, 16, ReservoirCell(288, spectral_radius=0.95, leaky=0.9), False,
                    True, True, False, False, 1)

    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


def LRMU_BestModel(tau, activation):
    Builder = ModelBuilder(f"LRMU_T{tau}", PROBLEM_NAME)
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


def LRMU_ESN_BestModel(tau, activation):
    Builder = ModelBuilder(f"LRMU_ESN_T{tau}", PROBLEM_NAME)
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
