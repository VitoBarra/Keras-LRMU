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


def FF_BaseLine(tau=17, activation="relu"):
    inputs = keras.Input(shape=(SEQUENCE_LENGTH,))
    output = keras.layers.Dense(1, activation=activation, name=f"FF_T{tau}_{activation}_input")(inputs)
    model = keras.Model(inputs=inputs, outputs=output, name=f"FF_T{tau}_{activation}_BaseLine")
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def LMU_BestModel(tau=17, activation="relu"):
    Builder = ModelBuilder(PROBLEM_NAME, f"LMU_{tau}")
    Builder.inputLayer(SEQUENCE_LENGTH)
    # if tau == 17:
    #     Builder.LMU(2, 1, 44, SimpleRNNCell(176), False,
    #                 True, False, False, False, 1)
    # elif tau == 30:
    #     Builder.LMU(1, 1, 64, SimpleRNNCell(144), False,
    #                 True, False, False, False, 1)

    if tau == 17:
        Builder.LMU(1, 1, 64, SimpleRNNCell(176), False,
                    True, False, False, False, 1)
    elif tau == 30:
        Builder.LMU(1, 1, 64, SimpleRNNCell(176), False,
                    True, False, False, False, 1)

    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


# comp models
def LMU_ESN_comp(tau=17, activation="relu"):
    Builder = ModelBuilder(PROBLEM_NAME, f"LMU_ESN_T{tau}")
    Builder.inputLayer(SEQUENCE_LENGTH)

    if tau == 17:
        Builder.LMU(1, 1, 64, ReservoirCell(176, spectral_radius=0.99, leaky=0.5), False,
                    True, False, False, False, 1)
    elif tau == 30:
        Builder.LMU(1, 1, 64, ReservoirCell(176, spectral_radius=0.99, leaky=0.5), False,
                    True, False, False, False, 1)

    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


def LRMU_comp(tau=17, activation="relu"):
    Builder = ModelBuilder(PROBLEM_NAME, f"LRMU_T{tau}")
    Builder.inputLayer(SEQUENCE_LENGTH)

    if tau == 17:
        Builder.LRMU(1, 1, 64, SimpleRNNCell(176),
                     True, False, False, False,
                     1.5, None, 1, None, 1)
    elif tau == 30:
        Builder.LRMU(1, 1, 64, SimpleRNNCell(176),
                     True, False, False, False,
                     1.5, None, 1, None, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


def LRMU_ESN_comp(tau=17, activation="relu"):
    Builder = ModelBuilder(PROBLEM_NAME, f"LRMU_ESN_T{tau}")
    Builder.inputLayer(SEQUENCE_LENGTH)
    if tau == 17:
        Builder.LRMU(1, 1, 44, ReservoirCell(176, spectral_radius=0.99, leaky=0.5),
                     True, False, False, False,
                     1.5, None, 1, None, 1)
    elif tau == 30:
        Builder.LRMU(1, 1, 64, ReservoirCell(176, spectral_radius=0.99, leaky=0.5),
                     True, False, False, False,
                     1.5, None, 1, None, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


def ModelEvaluation(model, testName, tau, dataset, batchSize=64, epochs=15):
    try:
        history, result = EvaluateModel(model, testName, dataset, batchSize, epochs, "val_mae")
    except:
        print("Error during model evaluation")
        raise
    print(f"total training time:{sum(history.history['time'])}s")
    print(f"test loss:{result[0]}")
    print(f"test mae:{result[1]}")

    try:
        SaveTrainingDataByName(DATA_DIR, f"{PROBLEM_NAME}/T{tau}", testName, history, result)
    except:
        print("something went wrong during plot data saving ")
        raise


def RunEvaluation(sample=128, tau=17, activation="relu", batchSize=64, epochs=25):
    dataSet = MackeyGlassDataset(0.1, 0.1, sample, SEQUENCE_LENGTH, 15, tau, 0)

    # ModelEvaluation(FF_BaseLine(tau, activation), f"FF_T{tau}_{activation}", dataSet, batchSize, epochs)
    # ModelEvaluation(LRMU_ESN_BestModel(tau, activation), f"LRMU_ESN_T{tau}_{activation}", dataSet, batchSize, epochs)
    # ModelEvaluation(LRMU_BestModel(tau, activation), f"LRMU_T{tau}_{activation}", dataSet, batchSize, epochs)
    # ModelEvaluation(LMU_ESN_BestModel(tau, activation), f"LMU_ESN_T{tau}_{activation}", dataSet, batchSize, epochs)
    # ModelEvaluation(LMU_BestModel(tau, activation), f"LMU_T{tau}_{activation}", dataSet, batchSize, epochs)

    ModelEvaluation(FF_BaseLine(tau, activation), f"FF_{activation}_BaseLine", tau, dataSet, batchSize, epochs)
    ModelEvaluation(LRMU_ESN_comp(tau, activation), f"LRMU_ESN_{activation}_comp", tau, dataSet, batchSize, epochs)
    ModelEvaluation(LRMU_comp(tau, activation), f"LRMU_{activation}_comp", tau, dataSet, batchSize, epochs)
    ModelEvaluation(LMU_ESN_comp(tau, activation), f"LMU_ESN_{activation}_comp", tau, dataSet, batchSize, epochs)
    ModelEvaluation(LMU_BestModel(tau, activation), f"LMU_{activation}_comp", tau, dataSet, batchSize, epochs)


def RunTuning(sample=128, sequenceLength=5000, tau=17, epoch=5, max_trial=50):
    lengthName = f"{str(sequenceLength)[0]}k"
    dataSet = MackeyGlassDataset(0.1, 0.1, sample, SEQUENCE_LENGTH, 15, tau, 0)

    hyperModels = HyperModel("MackeyGlass-hyperModel", PROBLEM_NAME, SEQUENCE_LENGTH, None, True, True)
    TunerTraining(hyperModels.LRMU_ESN(), f"LRMU_ESN_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, dataSet, epoch,
                  max_trial, True)
    TunerTraining(hyperModels.LRMU(), f"LRMU_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, dataSet, epoch,
                  max_trial, True)
    TunerTraining(hyperModels.LMU_ESN(), f"LMU_ESN_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, dataSet, epoch,
                  max_trial, True)
    TunerTraining(hyperModels.LMU(), f"LMU_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, dataSet, epoch,
                  max_trial, True)
