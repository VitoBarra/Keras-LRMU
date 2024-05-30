import os

import keras_tuner
import tensorflow as tf
import numpy as np
import numpy.random as rng

from LRMU import layer as lrmu
from Utility.DataUtil import SplitDataset
from Utility.PlotUtil import *
from Utility.Debug import *
import tensorflow.keras as ks

from Utility.ModelUtil import TrainAndTestModel_OBJ

PROBLEM_NAME = "pmMNIST"

def ModelLRMU(memoryDim, order, hiddenUnit, spectraRadius, reservoirMode, hiddenCell,
              memoryToMemory, hiddenToMemory, inputToCell, useBias, seed):
    sequence_length = 784
    classNumber = 10
    inputs = ks.Input(shape=(sequence_length, 1), name=f"{PROBLEM_NAME}_Input_LRMU")
    feature = lrmu.LRMU(memoryDimension=memoryDim, order=order, theta=sequence_length, hiddenUnit=hiddenUnit,
                        spectraRadius=spectraRadius, reservoirMode=reservoirMode, hiddenCell=hiddenCell,
                        memoryToMemory=memoryToMemory, hiddenToMemory=hiddenToMemory, inputToHiddenCell=inputToCell,
                        useBias=useBias, seed=seed)(inputs)
    outputs = ks.layers.Dense(classNumber, activation="softmax")(feature)
    model = ks.Model(inputs=inputs, outputs=outputs, name=f"{PROBLEM_NAME}_LRMU_Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def ModelLRMUWhitTuning(hp):
    memoryDim = 1
    order = 256
    hiddenUnit = 212
    spectraRadius = hp.Float("spectraRadius", min_value=0.8, max_value=1.3, step=0.025)
    reservoirMode = True
    hiddenCell = None
    memoryToMemory = False
    hiddenToMemory = True
    inputToHiddenCell = False
    useBias = False
    seed = 0
    return ModelLRMU(memoryDim, order, hiddenUnit, spectraRadius, reservoirMode, hiddenCell,
                     memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias, seed)


# 75, 352, 500, 1.15,True, None, False, True, False, True = 92.22% accuracy on test set
# 75, 352, 500, 1.18,True, None, False, True, False, True = 91.88% accuracy on test set
# 256, 128, 212, 1.15,True, None, False, True, False, True = 87.4% accuracy on test set
# 256, 128, 212, 0.99,True, None, False, True, False, True = 87.4% accuracy on test set
# 256, 256, 212, 0.99,True, None, False, True, False, True = 88.3% accuracy on test set
# 256, 256, 212, 1.18,True, None, False, True, False, True = 88.55%  accuracy on test set
def ModelLRMU_P():
    return ModelLRMU(1, 256, 212, 1.18,
                     True, None, False, True, False, False, 0)


def FullTraining(training, validation, test):
        history, result = TrainAndTestModel_OBJ(ModelLRMU_P, training, validation, test, 64, 10)

        print(f"Test loss: {result[0]}")
        print(f"Test accuracy: {result[1]}")
        PlotModelAccuracy(history, "Model LRMU",f"./plots/{PROBLEM_NAME}", f"{PROBLEM_NAME}_LRMU_ESN")


def TunerTraining(training, validation, test):
    tuner = keras_tuner.GridSearch(
        ModelLRMUWhitTuning,
        project_name=f"{PROBLEM_NAME}",
        executions_per_trial=1,
        # Do not resume the previous search in the same directory.
        overwrite=True,
        objective="val_accuracy",
        # Set a directory to store the intermediate results.
        directory=f"./logs/{PROBLEM_NAME}/tmp",
    )

    tuner.search(
        training.Data,
        training.Label,
        validation_data=(validation.Data, validation.Label),
        epochs=2,
        # Use the TensorBoard callback.
        callbacks=[ks.callbacks.TensorBoard(f"./logs/{PROBLEM_NAME}2")],
    )


def Run(fullTraining = True):
    ((train_images, train_labels), (test_images, test_labels)) = ks.datasets.mnist.load_data()

    Data = np.concatenate((train_images, test_images), axis=0)
    Label = np.concatenate((train_labels, test_labels), axis=0)

    Data = Data.reshape(Data.shape[0], -1, 1)
    rng.seed(1509)
    perm = rng.permutation(Data.shape[1])
    Data = Data[:, perm]
    Label = Label[:]
    training, validation, test = SplitDataset(Data, Label, 0.1, 0.1)

    if fullTraining:
        FullTraining(training, validation, test)
    else:
        TunerTraining(training, validation, test)

