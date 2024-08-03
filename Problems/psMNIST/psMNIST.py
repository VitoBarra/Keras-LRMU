import numpy.random as rng
from Utility.DataUtil import SplitDataset
from Utility.PlotUtil import *
from Utility.ModelUtil import *
from ESN.layer import *
from Utility.LRMUHyperModel import LRMUHyperModel
from Utility.LRMUModelBuilder import LRMUModelBuilder

PROBLEM_NAME = "psMNIST"
CLASS_NUMBER = 10
SEQUENCE_LENGTH = 784


def ModelLRMU_SelectedHP():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME, "manual test")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(10, 256, 212,
                             ReservoirCell(10),
                             True, True, True, False, False,
                             False,
                             1, 1)
    return LRMUBuilder.outputClassification(CLASS_NUMBER).composeModel().buildClassification()


def SingleTraining(training, validation, test):
    history, result = TrainAndTestModel_OBJ(ModelLRMU_SelectedHP, training, validation, test, 64, 10)

    print(f"Test loss: {result[0]}")
    print(f"Test accuracy: {result[1]}")
    PlotModelAccuracy(history, "Problems LRMU", f"./plots/{PROBLEM_NAME}", f"{PROBLEM_NAME}_LRMU")


def Run(singleTraining=True):
    ((train_images, train_labels), (test_images, test_labels)) = ks.datasets.mnist.load_data()

    Data = np.concatenate((train_images, test_images), axis=0)
    Label = np.concatenate((train_labels, test_labels), axis=0)

    Data = Data.reshape(Data.shape[0], -1, 1)
    rng.seed(1509)
    perm = rng.permutation(Data.shape[1])
    Data = Data[:15000, perm]
    Label = Label[:15000]
    training, validation, test = SplitDataset(Data, Label, 0.1, 0.1)

    hyperModels = LRMUHyperModel("psMNIST-hyperModel",PROBLEM_NAME, SEQUENCE_LENGTH, CLASS_NUMBER)

    if singleTraining:
        SingleTraining(training, validation, test)
    else:
        TunerTraining(hyperModels.LMU_ESN(), "LMU_ESN_Tuning_15k", PROBLEM_NAME, training, validation, 5, 150, False)
        TunerTraining(hyperModels.LMU_RE(), "LMU_RE_Tuning_15k", PROBLEM_NAME, training, validation, 5, 150, False)
        TunerTraining(hyperModels.LRMU(), "LRMU_Tuning_15k", PROBLEM_NAME, training, validation, 5, 150, True)
