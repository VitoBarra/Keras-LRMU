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


def ModelLRMU(memoryDim, order, hiddenUnit, spectraRadius, reservoirMode, hiddenCell,
              memoryToMemory, hiddenToMemory, inputToCell, useBias):
    sequence_length = 784
    classNumber = 10
    inputs = ks.Input(shape=(sequence_length, 1), name="pmMNIST_Input_LRMU")
    feature = lrmu.LRMU(memoryDimension=memoryDim, order=order, theta=sequence_length, hiddenUnit=hiddenUnit,
                        spectraRadius=spectraRadius, reservoirMode=reservoirMode, hiddenCell=hiddenCell,
                        memoryToMemory=memoryToMemory, hiddenToMemory=hiddenToMemory, inputToHiddenCell=inputToCell,
                        useBias=useBias)(inputs)
    outputs = ks.layers.Dense(classNumber, activation="softmax")(feature)
    model = ks.Model(inputs=inputs, outputs=outputs, name="pmMNIST_LRMU_Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def ModelLRMUWhitTuning(hp):
    memoryDim = hp.Int("memoryDim", min_value=10, max_value=100, step=5)
    order = hp.Int("order", min_value=32, max_value=512, step=32)
    hiddenUnit = hp.Int("hiddenUnit", min_value=50, max_value=500, step=50)
    spectraRadius = hp.Float("spectraRadius", min_value=0.8, max_value=1.25, step=0.05)
    reservoirMode = True
    hiddenCell = None
    memoryToMemory = hp.Boolean("memoryToMemory")
    hiddenToMemory = hp.Boolean("hiddenToMemory")
    inputToCell = hp.Boolean("inputToCell")
    useBias = hp.Boolean("useBias")
    return ModelLRMU(memoryDim, order, hiddenUnit, spectraRadius, reservoirMode, hiddenCell,
                     memoryToMemory, hiddenToMemory, inputToCell, useBias)


#75, 352, 500, 1.15,True, None, False, True, False, True = %92.22 accuracy on test set
#75, 352, 500, 1.18,True, None, False, True, False, True = %91.88 accuracy on test set
#256, 128, 212, 1.15,True, None, False, True, False, True = %87.4. accuracy on test set, hyper parameter  of the orginagl paper
#256, 128, 212, 0.99,True, None, False, True, False, True = % accuracy on test set, hyper parameter  of the orginagl paper
def ModelLRMU_P():
    return ModelLRMU(256, 128, 212, 0.99,
                     True, None, False, True, False, True)


def Run():


    ((train_images, train_labels), (test_images, test_labels)) = ks.datasets.mnist.load_data()

    Data = np.concatenate((train_images, test_images), axis=0)
    Label = np.concatenate((train_labels, test_labels), axis=0)

    Data = Data.reshape(Data.shape[0], -1, 1)
    rng.seed(1509)
    perm = rng.permutation(Data.shape[1])
    Data = Data[:, perm]
    Label = Label[:]
    training, validation, test = SplitDataset(Data, Label, 0.15, 0.1)

    # tuner = keras_tuner.RandomSearch(
    #     ModelLRMUWhitTuning,
    #     max_trials=100,
    #     project_name="pmMNIST",
    #     executions_per_trial=1,
    #     # Do not resume the previous search in the same directory.
    #     overwrite=True,
    #     objective="val_accuracy",
    #     # Set a directory to store the intermediate results.
    #     directory="./logs/pmMNIST/tmp",
    #
    # )

    # tuner.search(
    #     training.Data,
    #     training.Label,
    #     validation_data=(validation.Data, validation.Label),
    #     epochs=2,
    #     # Use the TensorBoard callback.
    #     callbacks=[ks.callbacks.TensorBoard("./logs/pmMNIST")],
    # )

    history, result = TrainAndTestModel_OBJ(ModelLRMU_P, training, validation, test, 64, 10)

    PrintAccuracy(result)
    PlotModel(history, "./Plot/pmMNIST", "pmMNIST_LRMU_ESN")
