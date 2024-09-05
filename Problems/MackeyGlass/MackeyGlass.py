from Problems.MackeyGlass.Config import *
import  Problems.MackeyGlass.Config as c
from Problems.MackeyGlass.DataGeneration import *
from Utility.ModelUtil import *
from Utility.PlotUtil import *
from ESN.layer import *
from Utility.HyperModel import HyperModel
from Utility.ModelBuilder import ModelBuilder
from tensorflow.keras.layers import SimpleRNNCell
from GlobalConfig import *



def LMU_T17_BestModel():
    #Done
    Builder = ModelBuilder(PROBLEM_NAME, "LMU")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(2, 1, 44, SimpleRNNCell(176), False,
                True, False, False, False, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION)
def LMU_ESN_T17_BestModel():
    #Done
    Builder = ModelBuilder(PROBLEM_NAME, "LMU_ESN")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(4, 20, 56, ReservoirCell(240, spectral_radius=0.95, leaky=0.7), False,
                False, True, False, True, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION)


def LRMU_T17_BestModel():
    Builder = ModelBuilder(PROBLEM_NAME, "LRMU")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LRMU(2, 48, 28, SimpleRNNCell(64, kernel_initializer=keras.initializers.GlorotUniform),
                 False, False, False, False,
                 1.5, 1, 1.5, 0.5, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION)


def LRMU_ESN_T17_BestModel():
    Builder = ModelBuilder(PROBLEM_NAME, "LRMU_ESN")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LRMU(4, 1, 144, ReservoirCell(320, spectral_radius=1, leaky=0.5),
                 True, False, True, True,
                 0.5, 1.5, 0.75, 2, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION)



def LMU_T30_BestModel():
    Builder = ModelBuilder(PROBLEM_NAME, "LMU")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(2, 1, 44, SimpleRNNCell(176), False,
                True, False, False, False, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION)
def LMU_ESN_T30_BestModel():
    Builder = ModelBuilder(PROBLEM_NAME, "LMU_ESN")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LMU(1, 4, 64, ReservoirCell(80, spectral_radius=0.99), False,
                False, True, False, True, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION)


def LRMU_T30_BestModel():
    Builder = ModelBuilder(PROBLEM_NAME, "LRMU")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LRMU(8, 1, 112, SimpleRNNCell(16, kernel_initializer=keras.initializers.GlorotUniform),
                 False, True, True, True,
                 1.25, 2, 1.75, 0.75, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION)


def LRMU_ESN_T30_BestModel():
    Builder = ModelBuilder(PROBLEM_NAME, "LRMU_ESN")
    Builder.inputLayer(SEQUENCE_LENGTH)
    Builder.LRMU(32, 1, 80, ReservoirCell(128, spectral_radius=0.99, leaky=0.8),
                 True, False, False, True,
                 0.5, 1.25, 1.0, 1.25, 1)
    return Builder.BuildPrediction(PREDICTION_DIMENSION)


def ModelEvaluation(model, testName, training, test, batchSize=64, epochs=15):
    try:
        history, result = EvaluateModel(model, testName, training, test, batchSize, epochs, "loss")
    except:
        print("Error during model evaluation")
        raise
    print(f"total training time:{sum(history.history['time'])}s")
    print(f"test loss:{result[0]}")
    print(f"test mse:{result[1]}")

    try:
        SaveDataForPlotJson(PLOTS_DIR, PROBLEM_NAME, testName, history, result)
    except:
        print("something went wrong during plot data saving ")
        raise


def RunEvaluation(sample=128, sequenceLength=5000, tau=17, batchSize=64, epochs=15):
    c.SEQUENCE_LENGTH = sequenceLength
    lengthName = f"{str(sequenceLength)[0]}k"
    training, validation, test = MackeyGlassDataset(0.1, 0.1, sample, SEQUENCE_LENGTH, 15, tau, 0)

    training.Concatenate(validation)

    if tau == 17:
        #ModelEvaluation(LMU_T17_Original_1Layer, f"LMU_{sample}_{lengthName}_T17", training, test, batchSize, epochs)
       # ModelEvaluation(LMU_ESN_T17_BestModel, f"LMU_ESN_{sample}_{lengthName}_T17", training, test, batchSize, epochs)
        #ModelEvaluation(LRMU_T17_BestModel, f"LRMU_{sample}_{lengthName}_T17", training, test, batchSize, epochs)
        ModelEvaluation(LRMU_ESN_T17_BestModel, f"LRMU_ESN_{sample}_{lengthName}_T17", training, test, batchSize,
                        epochs)
    elif tau == 30:
        #ModelEvaluation(LMU_T30_Original_1Layer, f"LMU_{sample}_{lengthName}_T30", training, test, batchSize, epochs)
        #ModelEvaluation(LMU_ESN_T30_BestModel, f"LMU_ESN_{sample}_{lengthName}_T30", training, test, batchSize, epochs)
        #ModelEvaluation(LRMU_T30_BestModel, f"LRMU_{sample}_{lengthName}_T30", training, test, batchSize, epochs)
        ModelEvaluation(LRMU_ESN_T30_BestModel, f"LRMU_ESN_{sample}_{lengthName}_T30", training, test, batchSize,
                        epochs)


def RunTuning(sample=128, sequenceLength=5000, tau=17, epoch=5,max_trial=50):
    lengthName = f"{str(sequenceLength)[0]}k"
    training, validation, test = MackeyGlassDataset(0.1, 0.1, sample, SEQUENCE_LENGTH, 15, tau, 0)

    hyperModels = HyperModel("MackeyGlass-hyperModel", PROBLEM_NAME, SEQUENCE_LENGTH, None, True, True)
    # TunerTraining(hyperModels.LMU(), f"LMU_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, training,
    #               validation, epoch, max_trial, True)
    TunerTraining(hyperModels.LMU_ESN(), f"LMU_ESN_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, training,
                  validation,   epoch, max_trial, True)
    print("arrivat qui")
    TunerTraining(hyperModels.LRMU(), f"LRMU_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, training,
                  validation, epoch, max_trial, True)
    TunerTraining(hyperModels.LRMU_ESN(), f"LRMU_ESN_Tuning_{sample}_{lengthName}_T{tau}", PROBLEM_NAME, training,
                  validation,epoch,max_trial, True)
