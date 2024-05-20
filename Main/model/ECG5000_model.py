import keras as ks
from Main.Util.ArffFormatUtill import *
import Main.Layer.keras_lmu as kslu
import Main.Util.PlotUtil as pu
from Main.Util.ModelUtil import *
from Main.Util.DataUtil import *
import Main.Layer.LRMU.layer as lrmu
from Main.Layer.ESN.layer import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
    feature = kslu.LMU(
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


def ModelLRMU():
    sequence_length = 140
    classNumber = 5
    inputs = ks.Input(shape=(sequence_length, 1), name="ECG5000_Input_LRMU")
    feature = lrmu.LRMU(
        memoryDimension=10, theta=140, order=32, hiddenUnit=250, spectraRadius=1.01,
        memoryToMemory=True, hiddenToMemory=True, useBias=True,
        reservoirMode=True, seed=159, returnSequences=False,
        hiddenCell=ks.layers.LSTMCell(50))(inputs)
    outputs = ks.layers.Dense(classNumber, activation="softmax")(feature)
    model = ks.Model(inputs=inputs, outputs=outputs, name="ECG5000Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


if __name__ == '__main__':
    path = "../../DataSets/ECG5000/"
    Data, Label = ReadFromCSVToKeras(path + "ECG5000_ALL.csv")
    Label -= 1
    training, validation, test = SplitDataset(Data, Label, 0.15, 0.1)

    history, result = TrainAndTestModel_OBJ(ModelLRMU, training, validation, test, 128, 15)

    pu.PlotModel(history)
    pu.PrintAccuracy(result)
