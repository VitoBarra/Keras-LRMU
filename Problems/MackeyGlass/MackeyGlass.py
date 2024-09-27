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
import tensorflow.keras


def FF_BaseLine(tau, activation):
    Builder = ModelBuilder("FF_Baseline", PROBLEM_NAME)
    Builder.inputLayer(SEQUENCE_LENGTH,Flatten=True)
    Builder.FF_Baseline()
    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)



def LMU_Base(tau, activation):
    Builder = ModelBuilder(f"LMU_T{tau}", PROBLEM_NAME)
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(1, 16, 64,
                SimpleRNNCell(176), False,
                True, True, True, False, 1)

    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


# Final Model
def LMU_ESN_comp(tau, activation):
    Builder = ModelBuilder(f"LMU_ESN_T{tau}", PROBLEM_NAME)
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(1, 16, 64,
                ReservoirCell(176, spectral_radius=0.87, leaky=0.5, input_scaling=1.75, bias_scaling=1.75), False,
                True, True, True, False, 1)

    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


def LRMU_comp(tau, activation):
    Builder = ModelBuilder(f"LRMU", PROBLEM_NAME,f"T{tau}")
    Builder.inputLayer(SEQUENCE_LENGTH)

    Builder.LRMU(1, 16, 64,
                 SimpleRNNCell(176),
                 True, True, True, False,
                 1, 1, 0.5, None, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


def LRMU_ESN_comp(tau, activation):
    Builder = ModelBuilder(f"LRMU_ESN", PROBLEM_NAME,f"T{tau}")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LRMU(1, 16, 64,
                 ReservoirCell(176, spectral_radius=1.1, leaky=0.9, input_scaling=0.5, bias_scaling=2.0),
                 True, True, True, False,
                 1, 1, 1.25, None, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


def LRMU_ESN_R(tau, activation):
    model = LRMU_ESN_Ridge(ModelType.Prediction, SEQUENCE_LENGTH, 1, 16, 64,
                           True, True, True, False, None,
                           None, 1.25, None,
                           176, tf.nn.tanh, 1.1, 0.9, 0.5, 2.0)
    model.name = f"LRMU_ESN_R_{PROBLEM_NAME}_T{tau}"
    model.summary()
    return model.custom_compile([keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()], )


def RunEvaluation(sample, tau, activation, batchSize, epochs):
    dataSet = MackeyGlassDataset(0.1, 0.1, sample, SEQUENCE_LENGTH, 15, tau, 0)
    dataSet.PrintSplit()
    dataSet.FlattenSeriesData()

    saveDir = f"{DATA_DIR}/{PROBLEM_NAME}/T{tau}_Final"
    monitorStat = "val_mae"

    # ModelEvaluation(FF_BaseLine(tau, activation), f"FF_{activation}_BaseLine", saveDir, dataSet, batchSize, epochs,
    #                 monitorStat)
    ModelEvaluation(LRMU_ESN_R(tau, activation), f"LRMU_ESN_RO", saveDir, dataSet, batchSize, epochs,
                    monitorStat)
    # ModelEvaluation(LRMU_ESN_comp(tau, activation), f"LRMU_ESN_{activation}_comp", saveDir, dataSet, batchSize, epochs,
    #                 monitorStat)
    # ModelEvaluation(LMU_ESN_comp(tau, activation), f"LMU_ESN_{activation}_comp", saveDir, dataSet, batchSize, epochs,
    #                 monitorStat)
    # ModelEvaluation(LRMU_comp(tau, activation), f"LRMU_{activation}_comp", saveDir, dataSet, batchSize, epochs,
    #                 monitorStat)
    # ModelEvaluation(LMU_Base(tau, activation), f"LMU_{activation}", saveDir, dataSet, batchSize, epochs, monitorStat)


def RunTuning(sample, tau, activation, epoch, max_trial):
    dataSet = MackeyGlassDataset(0.1, 0.1, sample, SEQUENCE_LENGTH, 15, tau, 0)
    dataSet.PrintSplit()

    hyperModels = HyperModel("MackeyGlass-hyperModel", PROBLEM_NAME, SEQUENCE_LENGTH, 0).SetUpPrediction(1, activation)
    hyperModels.ForceLMUParam(1, 1, 16, 64, 176).ForceConnection(True, True, True, False)
    # TunerTraining(hyperModels.LRMU_ESN_RC(), f"LRMU_ESN_RC", f"T{tau}_Final", PROBLEM_NAME, dataSet, epoch,
    #               max_trial, False)
    TunerTraining(hyperModels.LRMU_ESN(), f"LRMU_ESN", f"T{tau}_Final", PROBLEM_NAME, dataSet, epoch,
                  max_trial, True)
    # TunerTraining(hyperModels.LMU_ESN(), f"LMU_ESN", f"T{tau}_Final", PROBLEM_NAME, dataSet, epoch,
    #               max_trial, True)
    TunerTraining(hyperModels.LRMU(), f"LRMU", f"T{tau}_Final", PROBLEM_NAME, dataSet, epoch,
                  max_trial, True)
    # TunerTraining(hyperModels.LMU(), f"LMU",f"T{tau}_Final", PROBLEM_NAME, dataSet, epoch,
    #               max_trial, True)


def PrintSummary(tau, activation):
    FF_BaseLine(tau, activation)
    LRMU_ESN_R(tau, activation)
    LRMU_ESN_comp(tau, activation)
    LMU_ESN_comp(tau, activation)
    LRMU_comp(tau, activation)
    LMU_Base(tau, activation)
