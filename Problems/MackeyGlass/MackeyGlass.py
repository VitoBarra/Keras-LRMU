from Problems.MackeyGlass.Config import *
import Problems.MackeyGlass.Config as c
from Problems.MackeyGlass.DataGeneration import *
from Utility.ModelUtil import *
from Utility.PlotUtil import *
from ESN.layer import *
from Utility.HyperModel import HyperModel
from Utility.ModelBuilder import ModelBuilder
from tensorflow.keras.layers import SimpleRNNCell
from GlobalConfig import *
import tensorflow.keras as keras


def FF_BaseLine():
    inputs = keras.Input(shape=(SEQUENCE_LENGTH,))
    output = keras.layers.Dense(1, activation='linear')(inputs)
    model = keras.Model(inputs=inputs, outputs=output)
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def LMU_BestModel(tau=17, activation="relu"):
    #Done
    Builder = ModelBuilder(PROBLEM_NAME, f"LMU_{tau}")
    Builder.inputLayer(SEQUENCE_LENGTH)
    if tau == 17:
        Builder.LMU(2, 1, 44, SimpleRNNCell(176), False,
                    True, False, False, False, 1)
    elif tau == 30:
        Builder.LMU(1, 1, 64, SimpleRNNCell(144), False,
                    True, False, False, False, 1)

    return Builder.BuildPrediction(PREDICTION_DIMENSION, activation)


def LMU_ESN_BestModel(tau=17, activation="relu"):
    #Done
    Builder = ModelBuilder(PROBLEM_NAME, f"LMU_ESN_T{tau}")
    Builder.inputLayer(SEQUENCE_LENGTH)
    if tau == 17:
        Builder.LMU(4, 20, 56, ReservoirCell(240, spectral_radius=0.95, leaky=0.7), False,
                    False, True, False, True, 1)
    elif tau == 30:
        Builder.LMU(8, 8, 16, ReservoirCell(288, spectral_radius=0.975, leaky=0.9), False,
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


def ModelEvaluation(model, testName, training, test, batchSize=64, epochs=15):
    try:
        history, result = EvaluateModel(model, testName, training, test, batchSize, epochs, "loss")
    except:
        print("Error during model evaluation")
        raise
    print(f"total training time:{sum(history.history['time'])}s")
    print(f"test loss:{result[0]}")
    print(f"test mae:{result[1]}")

    try:
        SaveDataForPlotJson(DATA_DIR, PROBLEM_NAME, testName, history, result)
    except:
        print("something went wrong during plot data saving ")
        raise


def RunEvaluation(sample=128, sequenceLength=5000, tau=17, activation="relu", batchSize=64, epochs=15):
    c.SEQUENCE_LENGTH = sequenceLength
    lengthName = f"{str(sequenceLength)[0]}k"
    training, validation, test = MackeyGlassDataset(0.1, 0.1, sample, SEQUENCE_LENGTH, 15, tau, 0)

    training.Concatenate(validation)

    ModelEvaluation(LRMU_ESN_BestModel(tau, activation), f"LRMU_ESN_{sample}_{lengthName}_T{tau}_{activation}",
                    training, test, batchSize, epochs)
    ModelEvaluation(LRMU_BestModel(tau, activation), f"LRMU_{sample}_{lengthName}_T{tau}_{activation}", training, test,
                    batchSize, epochs)
    ModelEvaluation(LMU_ESN_BestModel(tau, activation), f"LMU_ESN_{sample}_{lengthName}_T{tau}_{activation}", training,
                    test, batchSize, epochs)
    ModelEvaluation(LMU_BestModel(tau, activation), f"LMU_{sample}_{lengthName}_T{tau}_{activation}", training, test,
                    batchSize, epochs)

    ModelEvaluation(FF_BaseLine, f"FF_{sample}_{lengthName}", training, test, batchSize, epochs)


def RunTuning(sample=128, sequenceLength=5000, tau=17, epoch=5, max_trial=50):
    lengthName = f"{str(sequenceLength)[0]}k"
    training, validation, test = MackeyGlassDataset(0.1, 0.1, sample, SEQUENCE_LENGTH, 15, tau, 0)

    hyperModels = HyperModel("MackeyGlass-hyperModel", PROBLEM_NAME, SEQUENCE_LENGTH, None, True, True)
    # TunerTraining(hyperModels.LRMU_ESN(), f"LRMU_ESN_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, training,
    #               validation,epoch,max_trial, True)
    # TunerTraining(hyperModels.LRMU(), f"LRMU_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, training,
    #               validation, epoch, max_trial, True)
    TunerTraining(hyperModels.LMU_ESN(), f"LMU_ESN_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, training,
                  validation, epoch, max_trial, True)
    TunerTraining(hyperModels.LMU(), f"LMU_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, training,
                  validation, epoch, max_trial, True)
