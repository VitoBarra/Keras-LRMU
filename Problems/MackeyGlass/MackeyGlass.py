from Problems.MackeyGlass.Config import *
from Problems.MackeyGlass.DataGeneration import *
from Reservoir.layer import *
from Utility.HyperModel import HyperModel
from Utility.ModelBuilder import ModelBuilder
from Utility.ModelUtil import ModelEvaluation, TunerTraining
from tensorflow.keras.layers import SimpleRNNCell
from GlobalConfig import *
import tensorflow.keras as keras
from LRMU.Model import LRMU_ESN_Ridge
from LRMU.utility import ModelType


def FF_BaseLine(tau, activation):
    Builder = ModelBuilder("FF_Baseline", PROBLEM_NAME)
    Builder.inputLayer(SEQUENCE_LENGTH, Flatten=True)
    Builder.FF_Baseline()
    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


def LMU(tau, activation):
    Builder = ModelBuilder(f"LMU_T{tau}", PROBLEM_NAME)
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(1, 16, 64,
                SimpleRNNCell(176), False,
                True, True, True, False, 1)

    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


# Final Model
def LMU_ESN(tau, activation):
    Builder = ModelBuilder(f"LMU_ESN_T{tau}", PROBLEM_NAME)
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(1, 16, 64,
                ReservoirCell(176, spectral_radius=0.87, leaky=0.5, input_scaling=1.75, bias_scaling=1.75), False,
                True, True, True, False, 1)

    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


#Final Model score 0.1271391212940216
def LRMU(tau, activation):
    Builder = ModelBuilder(f"LRMU", PROBLEM_NAME, f"T{tau}")
    Builder.inputLayer(SEQUENCE_LENGTH)

    Builder.LRMU(1, 16, 64, SimpleRNNCell(176),
                 True, True, True, False,
                 0.5, 0.5, 1.75, None, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


#Final Model score 0.09625670313835144 con T30
def LRMU_ESN(tau, activation):
    Builder = ModelBuilder(f"LRMU_ESN", PROBLEM_NAME, f"T{tau}")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LRMU(1, 16, 64,
                 ReservoirCell(176, spectral_radius=1.09, leaky=0.7, input_scaling=1.5, bias_scaling=1.5),
                 True, True, True, False,
                 0.5, 0.5, 1.5, None, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


def LRMU_ESN_R(tau, activation):
    model = LRMU_ESN_Ridge(ModelType.Prediction, SEQUENCE_LENGTH, 1, 16, 64,
                           True, True, True, False,
                           0.5, 0.5, 1.5, None,
                           176, tf.nn.tanh, 1.09, 0.7, 1.5, 1.5)
    model.name = f"LRMU_ESN_R_{PROBLEM_NAME}_T{tau}"
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model.custom_compile([keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()], )


def RunEvaluation(sample, tau, activation, batchSize, epochs):
    dataSet = MackeyGlassDataset(0.1, 0.1, sample, SEQUENCE_LENGTH, 15, tau, 0)
    dataSet.PrintSplit()
    # dataSet.FlattenSeriesData()

    saveDir = f"{DATA_DIR}/{PROBLEM_NAME}/T{tau}_Final_TEST"
    monitorStat = "val_mae"
    if tau in [17, 30]:
        ModelEvaluation(FF_BaseLine(tau, activation), f"FF_{activation}_BaseLine", saveDir, dataSet, batchSize, epochs,
                        monitorStat)
    if tau in [17, 30]:
        ModelEvaluation(LRMU_ESN_R(tau, activation), f"LRMU_ESN_R", saveDir, dataSet, batchSize, epochs, monitorStat)
    if tau in [17, 30]:
        ModelEvaluation(LRMU_ESN(tau, activation), f"LRMU_ESN", saveDir, dataSet, batchSize, epochs, monitorStat)
    if tau in [17, 30]:
        ModelEvaluation(LMU_ESN(tau, activation), f"LMU_ESN", saveDir, dataSet, batchSize, epochs, monitorStat)
    if tau in [17, 30]:
        ModelEvaluation(LRMU(tau, activation), f"LRMU", saveDir, dataSet, batchSize, epochs, monitorStat)
    if tau in [17, 30]:
        ModelEvaluation(LMU(tau, activation), f"LMU", saveDir, dataSet, batchSize, epochs, monitorStat)


def RunTuning(sample, tau, activation, epoch, max_trial):
    dataSet = MackeyGlassDataset(0.1, 0.1, sample, SEQUENCE_LENGTH, 15, tau, 0)
    tuningName = f"T{tau}_Final"
    dataSet.PrintSplit()

    hyperModels = HyperModel("MackeyGlass-hyperModel", PROBLEM_NAME, SEQUENCE_LENGTH, f"T{tau}", 0)
    hyperModels.SetUpPrediction(1, activation)
    hyperModels.ForceLMUParam(1, 1, 16, 64, 176).ForceConnection(True, True, True, False)
    TunerTraining(hyperModels.LRMU_ESN_Ridge(), f"LRMU_ESN_R", tuningName, PROBLEM_NAME, dataSet, epoch, max_trial, False)
    TunerTraining(hyperModels.LRMU_ESN(), f"LRMU_ESN", tuningName, PROBLEM_NAME, dataSet, epoch,  max_trial, False)
    TunerTraining(hyperModels.LMU_ESN(), f"LMU_ESN", tuningName, PROBLEM_NAME, dataSet, epoch, max_trial, False)
    TunerTraining(hyperModels.LRMU(), f"LRMU", tuningName, PROBLEM_NAME, dataSet, epoch, max_trial, False)
    TunerTraining(hyperModels.LMU(), f"LMU", tuningName, PROBLEM_NAME, dataSet, epoch, max_trial, False)


def PrintSummary(tau, activation):
    FF_BaseLine(tau, activation)
    LRMU_ESN_R(tau, activation)
    LRMU_ESN(tau, activation)
    LMU_ESN(tau, activation)
    LRMU(tau, activation)
    LMU(tau, activation)
