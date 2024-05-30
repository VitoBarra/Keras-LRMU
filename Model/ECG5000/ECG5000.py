import tensorflow as tf
import keras as ks
import keras_tuner

from Utility.ArffFormatUtill import *
from Utility.DataUtil import *
import LMU as LMULayer
import LRMU as LRMULayer
import os

from Utility.Debug import PrintAvailableGPU
from Utility.ModelUtil import TrainAndTestModel_OBJ
from Utility.PlotUtil import *

PROBLEM_NAME = "ECG5000"

def ModelFFBaseline():
    sequence_length = 140
    classNumber = 5
    inputs = ks.Input(shape=(sequence_length,), name="ECG5000_Input")
    feature = ks.layers.Dense(200)(inputs)
    feature = ks.layers.Dense(200)(feature)
    outputs = ks.layers.Dense(classNumber, activation="softmax")(feature)
    model = ks.Model(inputs=inputs, outputs=outputs, name="ECG5000Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def ModelLSTM():
    sequence_length = 140
    classNumber = 5
    inputs = ks.Input(shape=(sequence_length, 1), name="ECG5000_Input")
    feature = ks.layers.LSTM(52, return_sequences=True)(inputs)
    feature = ks.layers.LSTM(52, return_sequences=False)(feature)
    outputs = ks.layers.Dense(classNumber, activation="softmax")(feature)
    model = ks.Model(inputs=inputs, outputs=outputs, name="ECG5000Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def ModelLMU():
    sequence_length = 140
    classNumber = 5
    inputs = ks.Input(shape=(sequence_length, 1), name="ECG5000_Input_LMU")
    feature = LMULayer.LMU(
        10, theta=140, order=32, memory_to_memory=True, hidden_to_memory=True,
        use_bias=True, trainable_theta=True, seed=159,
        hidden_cell=ks.layers.LSTMCell(50))(inputs)
    outputs = ks.layers.Dense(classNumber, activation="softmax")(feature)
    model = ks.Model(inputs=inputs, outputs=outputs, name="ECG5000Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def ModelLRMU(memoryDim, order, hiddenUnit, spectraRadius, reservoirMode, hiddenCell,
              memoryToMemory, hiddenToMemory, inputToCell, useBias):
    sequence_length = 140
    classNumber = 5
    inputs = ks.Input(shape=(sequence_length, 1), name="ECG5000_Input_LRMU")
    feature = LRMULayer.LRMU(memoryDimension=memoryDim, order=order, theta=sequence_length, hiddenUnit=hiddenUnit,
                             spectraRadius=spectraRadius, reservoirMode=reservoirMode, hiddenCell=hiddenCell,
                             memoryToMemory=memoryToMemory, hiddenToMemory=hiddenToMemory, inputToHiddenCell=inputToCell,
                             useBias=useBias)(inputs)
    outputs = ks.layers.Dense(classNumber, activation="softmax")(feature)
    model = ks.Model(inputs=inputs, outputs=outputs, name="ECG5000Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def ModelLRMUWhitTuning(hp):
    memoryDim = hp.Int("memoryDim", min_value=10, max_value=100, step=5)
    order = hp.Int("order", min_value=32, max_value=512, step=32)
    hiddenUnit = hp.Int("hiddenUnit", min_value=50, max_value=1500, step=50)
    spectraRadius = hp.Float("spectraRadius", min_value=0.8, max_value=1.25, step=0.05)
    reservoirMode = True
    hiddenCell = None
    memoryToMemory = hp.Boolean("memoryToMemory")
    hiddenToMemory = hp.Boolean("hiddenToMemory")
    inputToHiddenCell = hp.Boolean("inputToCell")
    useBias = hp.Boolean("useBias")
    return ModelLRMU(memoryDim, order, hiddenUnit, spectraRadius, reservoirMode, hiddenCell,
                         memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias)

def ModelLRMU_P():
    return ModelLRMU(15, 64, 1050, 0.8, True, None, True, False, True, False)

def FullTraining(training, validation, test):
        history, result = TrainAndTestModel_OBJ(ModelLRMU_P, training, validation, test, 64, 10)

        print('Test loss:', result[0])
        print('Test accuracy:', result[1])
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

    path = "Data/"
    Data, Label = ReadFromCSVToKeras(path + "ECG5000_ALL.csv")
    Label -= 1
    training, validation, test = SplitDataset(Data, Label, 0.15, 0.1)


    if fullTraining:
        FullTraining(training, validation, test)
    else:
        TunerTraining(training, validation, test)

