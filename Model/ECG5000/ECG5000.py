import tensorflow as tf
import keras as ks
import keras_tuner

from Utility.ArffFormatUtill import *
from Utility.DataUtil import *
import LMU as LMULayer
import LRMU as LRMULayer
import os

from Utility.Debug import PrintAvailableGPU


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
        hidden_cell=ks.layers.LSTM(50))(inputs)
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
                             memoryToMemory=memoryToMemory, hiddenToMemory=hiddenToMemory, inputToCell=inputToCell,
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
    inputToCell = hp.Boolean("inputToCell")
    useBias = hp.Boolean("useBias")
    return ModelLRMU(memoryDim, order, hiddenUnit, spectraRadius, reservoirMode, hiddenCell,
                         memoryToMemory, hiddenToMemory, inputToCell, useBias)

    #return ModelLRMU(15, 64, 1050, 0.8, True, None, True, False, True, False)


def Run():

    PrintAvailableGPU()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    path = "Data/"
    Data, Label = ReadFromCSVToKeras(path + "ECG5000_ALL.csv")
    Label -= 1
    training, validation, test = SplitDataset(Data, Label, 0.15, 0.1)

    tuner = keras_tuner.RandomSearch(
        ModelLRMUWhitTuning,
        project_name="ECG5000",
        max_trials=100,
        executions_per_trial=1,
        # Do not resume the previous search in the same directory.
        overwrite=True,
        objective="val_accuracy",
        # Set a directory to store the intermediate results.
        directory="./logs/Ecg5000/tmp",

    )

    tuner.search(
        training.Data,
        training.Label,
        validation_data=(validation.Data, validation.Label),
        epochs=2,
        # Use the TensorBoard callback.
        callbacks=[ks.callbacks.TensorBoard("./logs/Ecg5000")],
    )


# history, result = TrainAndTestModel_OBJ(ModelLRMU, training, validation, test, 128, 15)

    # pu.PlotModel(history)
    # pu.PrintAccuracy(result)
