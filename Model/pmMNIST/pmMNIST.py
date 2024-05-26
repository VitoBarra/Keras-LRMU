import os
import tensorflow as tf
import numpy as np
import numpy.random as rng

from LRMU import layer as lrmu
from Utility.DataUtil import SplitDataset
from Utility.PlotUtil import *
import tensorflow.keras as ks

from Utility.ModelUtil import TrainAndTestModel_OBJ


def ModelLRMU(memoryDim, order, hiddenUnit, spectraRadius, reservoirMode, hiddenCell,
              memoryToMemory, hiddenToMemory, inputToCell, useBias):
    sequence_length = 784
    classNumber = 10
    inputs = ks.Input(shape=(sequence_length, 1), name="pmMNIST_Input_LRMU")
    feature = lrmu.LRMU(memoryDimension=memoryDim, order=order, theta=sequence_length, hiddenUnit=hiddenUnit,
                        spectraRadius=spectraRadius, reservoirMode=reservoirMode, hiddenCell=hiddenCell,
                        memoryToMemory=memoryToMemory, hiddenToMemory=hiddenToMemory, inputToCell=inputToCell,
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


def ModelLRMU_P():
    return ModelLRMU(80,128,150,1.2,
                     True,None,False,True,False,False)



def Run():
    print(tf.config.list_physical_devices('GPU'))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ((train_images, train_labels), (test_images, test_labels)) = ks.datasets.mnist.load_data()

    Data = np.concatenate((train_images, test_images), axis=0)
    Label = np.concatenate((train_labels, test_labels), axis=0)

    Data = Data.reshape(Data.shape[0], -1, 1)
    perm = rng.permutation(Data.shape[1])
    Data = Data[:1000, perm]
    Label = Label[:1000]
    training, validation, test = SplitDataset(Data, Label, 0.15, 0.1)

    # tuner = keras_tuner.RandomSearch(
    #     ModelLRMUWhitTuning,
    #     max_trials=30,
    #     executions_per_trial=1,
    #     # Do not resume the previous search in the same directory.
    #     overwrite=True,
    #     objective="val_accuracy",
    #     # Set a directory to store the intermediate results.
    #     directory="./tmp/tb",
    #
    # )
    #
    # tuner.search(
    #     training.Data,
    #     training.Label,
    #     validation_data=(validation.Data, validation.Label),
    #     epochs=2,
    #
    #     # Use the TensorBoard callback.
    #     # The logs will be write to "/tmp/tb_logs".
    #     callbacks=[ks.callbacks.TensorBoard("./tmp/tb_logs")],
    # )

    history, result = TrainAndTestModel_OBJ(ModelLRMU_P, training, validation, test, 64, 15)

    PlotModel(history)
    PrintAccuracy(result)
