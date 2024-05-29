import os

import keras as ks

from Utility.ArffFormatUtill import *
import LMU as kslu
import Utility.PlotUtil as pu
from Utility.ModelUtil import *
from Utility.DataUtil import *
from Utility.Debug import *

PROBLEM_NAME = "EthanolLevel"

def ModelLMU():
    sequence_length = 500
    numberOfClass = 4
    inputs = ks.Input(shape=(sequence_length, 1), name=f"{PROBLEM_NAME}_Input")
    feature = kslu.LMU(5, return_sequences=True,
                       order=256, theta=sequence_length,
                       trainable_theta=True,
                       hidden_cell=ks.layers.SimpleRNNCell(100))(inputs)
    feature = kslu.LMU(5, return_sequences=False,
                       order=256, theta=sequence_length,
                       trainable_theta=True,
                       hidden_cell=ks.layers.SimpleRNNCell(100))(feature)
    outputs = ks.layers.Dense(numberOfClass, activation="softmax")(feature)
    outputs = ks.layers.Reshape((numberOfClass,))(outputs)
    model = ks.Model(inputs=inputs, outputs=outputs, name=f"{PROBLEM_NAME}Model")

    model.summary()
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def ModelLSTM():
    sequence_length = 500
    classNumber = 4
    inputs = ks.Input(shape=(sequence_length, 1), name=f"{PROBLEM_NAME}_Input")
    feature = ks.layers.LSTM(250, return_sequences=True)(inputs)
    feature = ks.layers.LSTM(250, return_sequences=True)(feature)
    feature = ks.layers.LSTM(250, return_sequences=True)(feature)
    feature = ks.layers.LSTM(250, return_sequences=True)(feature)
    feature = ks.layers.LSTM(250, return_sequences=True)(feature)
    feature = ks.layers.LSTM(250, return_sequences=False)(feature)
    outputs = ks.layers.Dense(classNumber, activation="softmax")(feature)
    model = ks.Model(inputs=inputs, outputs=outputs, name=f"{PROBLEM_NAME}Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def ModelFFBaseline():
    sequence_length = 500
    classNumber = 4
    inputs = ks.Input(shape=(sequence_length,), name=f"{PROBLEM_NAME}_Input")
    feature = ks.layers.Dense(1000)(inputs)
    feature = ks.layers.Dense(1000)(feature)
    feature = ks.layers.Dense(1000)(feature)
    feature = ks.layers.Dense(1000)(feature)
    feature = ks.layers.Dense(1000)(feature)
    feature = ks.layers.Dense(1000)(feature)
    outputs = ks.layers.Dense(classNumber, activation="softmax")(feature)

    model = ks.Model(inputs=inputs, outputs=outputs, name=f"{PROBLEM_NAME}Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def Run(dataPath):



    samplingRate = 3

    rawData, rawLabel = ReadFromCSVToKeras(dataPath + "EthanolLevel_ALL.csv")

    Data = CropTimeSeries(rawData, 250)
    Data = TimeSeriesSampleRate(Data, samplingRate)
    Data = Data / Data.mean()

    Label = rawLabel - 1
    training, validation, test = SplitDataset(Data, Label, 0.15, 0.1)

    history, result = TrainAndTestModel_OBJ(ModelLMU, training, validation, test, 32, 50)
    pu.PlotModelAccuracy(history, result)
