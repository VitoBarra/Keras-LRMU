import numpy.random as rng
from Utility.DataUtil import *
from Utility.PlotUtil import *
from Utility.ModelUtil import *
from ESN.layer import *
from Utility.LRMUHyperModel import LRMUHyperModel
from Utility.LRMUModelBuilder import LRMUModelBuilder
from keras.layers import SimpleRNNCell

PROBLEM_NAME = "psMNIST"
CLASS_NUMBER = 10
SEQUENCE_LENGTH = 784



def LMU_ESN_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, f"{PROBLEM_NAME}_LMU_ESN")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(32, 24, 192, False,
                             ReservoirCell(160, spectral_radius=1.05,leaky=0.5),
                             False, True, False, False,
                             1, 1, 1, 1, 1)
    return LRMUBuilder.outputClassification(CLASS_NUMBER).composeModel().buildClassification()


def LMU_RE_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, f"{PROBLEM_NAME}_LMU_RE")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(2, 48, 256, True,
                             SimpleRNNCell(288, kernel_initializer=keras.initializers.GlorotUniform),
                             False, True, True, False,
                             2.0, 0.75, 2.0, 1.75, 1)
    return LRMUBuilder.outputClassification(CLASS_NUMBER).composeModel().buildClassification()


def LRMU_BestModel():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, f"{PROBLEM_NAME}_LRMU")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(8, 64, 64, True,
                             ReservoirCell(320,spectral_radius=1.1,leaky=0.70),
                             False, True, True, False,
                             1.75, 2, 1.0, 1.75, 1)
    return LRMUBuilder.outputClassification(CLASS_NUMBER).composeModel().buildClassification()



def ModelEvaluation(model,training, test,testName):
    history, result = EvaluateModel(model, training, test, 64, 10)

    print(f"Test loss: {result[0]}")
    print(f"Test accuracy: {result[1]}")
    PlotModelAccuracy(history, f"{PROBLEM_NAME}_{testName}", f"./plots/{PROBLEM_NAME}", f"{testName}")


def Run(modelEvaluation=True):
    ((train_images, train_labels), (test_images, test_labels)) = ks.datasets.mnist.load_data()

    Data = np.concatenate((train_images, test_images), axis=0)
    Label = np.concatenate((train_labels, test_labels), axis=0)

    Data = Data.reshape(Data.shape[0], -1, 1)
    rng.seed(1509)
    perm = rng.permutation(Data.shape[1])
    Data = Data[:15000, perm]
    Label = Label[:15000]

    dataSet= DataLabel(Data, Label)
    training, validation, test = dataSet.SplitDataset(0.1, 0.1)

    hyperModels = LRMUHyperModel("psMNIST-hyperModel",PROBLEM_NAME, SEQUENCE_LENGTH, CLASS_NUMBER)

    if modelEvaluation:
        training.Concatenate(validation)
        ModelEvaluation(LMU_ESN_BestModel, training, test,"LMU_ESN")
        ModelEvaluation(LMU_RE_BestModel, training, test,"LMU_RE")
        ModelEvaluation(LRMU_BestModel, training, test,"LRMU")
    else:
        TunerTraining(hyperModels.LMU_ESN(), "LMU_ESN_Tuning_15k", PROBLEM_NAME, training, validation, 5, 150, False)
        TunerTraining(hyperModels.LMU_RE(), "LMU_RE_Tuning_15k", PROBLEM_NAME, training, validation, 5, 150, False)
        TunerTraining(hyperModels.LRMU(), "LRMU_Tuning_15k", PROBLEM_NAME, training, validation, 5, 150, True)
