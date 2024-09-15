from Problems.MackeyGlass.Config import *
from Problems.MackeyGlass.DataGeneration import *
from ESN.layer import *
from Utility.HyperModel import HyperModel
from Utility.ModelBuilder import ModelBuilder
from Utility.ModelUtil import ModelEvaluation, TunerTraining
from tensorflow.keras.layers import SimpleRNNCell
from GlobalConfig import *
import tensorflow.keras as keras


def FF_BaseLine(tau, activation):
    inputs = keras.Input(shape=(SEQUENCE_LENGTH,))
    output = keras.layers.Dense(1, activation=activation, name=f"FF_T{tau}_{activation}_input")(inputs)
    model = keras.Model(inputs=inputs, outputs=output, name=f"FF_T{tau}_{activation}_BaseLine")
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def LMU_Base(tau, activation):
    Builder = ModelBuilder(PROBLEM_NAME, f"LMU_{tau}")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(1, 16, 64, SimpleRNNCell(176), False,
                False, False, True, False, 1)

    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


# comp models
def LMU_ESN_comp(tau, activation):
    Builder = ModelBuilder(PROBLEM_NAME, f"LMU_ESN_T{tau}")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(1, 16, 64, ReservoirCell(176, spectral_radius=0.99, leaky=0.5, input_scaling=1), False,
                False, False, True, False, 1)

    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


def LRMU_comp(tau, activation):
    Builder = ModelBuilder(PROBLEM_NAME, f"LRMU_T{tau}")
    Builder.inputLayer(SEQUENCE_LENGTH)

    Builder.LRMU(1, 16, 64, SimpleRNNCell(176),
                 False, False, True, False,
                 1.5, None, 1, None, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


def LRMU_ESN_comp(tau, activation):
    Builder = ModelBuilder(PROBLEM_NAME, f"LRMU_ESN_T{tau}")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LRMU(1, 16, 64, ReservoirCell(176, spectral_radius=0.99, leaky=0.5, input_scaling=1),
                 False, False, True, False,
                 1.5, None, 1, None, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


def RunEvaluation(sample, tau, activation, batchSize, epochs):
    dataSet = MackeyGlassDataset(0.1, 0.1, sample, SEQUENCE_LENGTH, 15, tau, 0)

    saveDir = f"{DATA_DIR}/{PROBLEM_NAME}/T{tau}_O16"
    monitorStat = "val_mae"

    ModelEvaluation(FF_BaseLine(tau, activation), f"FF_{activation}_BaseLine", saveDir, dataSet, batchSize, epochs,
                    monitorStat)
    ModelEvaluation(LRMU_ESN_comp(tau, activation), f"LRMU_ESN_{activation}_comp", saveDir, dataSet, batchSize, epochs,
                    monitorStat)
    ModelEvaluation(LRMU_comp(tau, activation), f"LRMU_{activation}_comp", saveDir, dataSet, batchSize, epochs,
                    monitorStat)
    ModelEvaluation(LMU_ESN_comp(tau, activation), f"LMU_ESN_{activation}_comp", saveDir, dataSet, batchSize, epochs,
                    monitorStat)
    ModelEvaluation(LMU_Base(tau, activation), f"LMU_{activation}", saveDir, dataSet, batchSize, epochs, monitorStat)


def RunTuning(sample, tau,activation, epoch, max_trial):
    dataSet = MackeyGlassDataset(0.1, 0.1, sample, SEQUENCE_LENGTH, 15, tau, 0)

    hyperModels = HyperModel("MackeyGlass-hyperModel", PROBLEM_NAME, SEQUENCE_LENGTH, 0).SetUpPrediction(1, activation)
    hyperModels.ForceLMUParam(1, 1, 16, 64, 176).ForceConnection(False, False, True, False)
    # TunerTraining(hyperModels.LMU(), f"LMU_Tuning_T{tau}_ParSet_con", PROBLEM_NAME, dataSet, epoch,
    #               max_trial, True)
    TunerTraining(hyperModels.LRMU_ESN(), f"LRMU_ESN_Tuning_T{tau}_ParSet_con", PROBLEM_NAME, dataSet, epoch,
                  max_trial, True)
    TunerTraining(hyperModels.LMU_ESN(), f"LMU_ESN_Tuning_T{tau}_ParSet_con", PROBLEM_NAME, dataSet, epoch,
                  max_trial, True)
    TunerTraining(hyperModels.LRMU(), f"LRMU_Tuning_T{tau}_ParSet_con", PROBLEM_NAME, dataSet, epoch,
                  max_trial, True)
