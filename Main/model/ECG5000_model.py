import keras as ks
from Main.Util.ArffFormatUtill import *
import Main.Layer.keras_lmu as kslu
import Main.Util.PlotUtil as pu
from Main.Util.ModelUtil import *
from Main.Util.DataUtil import *


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
    inputs = ks.Input(shape=(sequence_length, 1), name="ECG5000_Input")
    feature = kslu.LMU(2, return_sequences=True, trainable_theta=True,
                       theta=140, order=128, hidden_cell=ks.layers.LSTMCell(50))(inputs)
    feature = kslu.LMU(2, return_sequences=False, trainable_theta=True,
                       theta=140, order=128, hidden_cell=ks.layers.LSTMCell(50))(feature)
    outputs = ks.layers.Dense(classNumber, activation="softmax")(feature)
    model = ks.Model(inputs=inputs, outputs=outputs, name="ECG5000Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


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


if __name__ == '__main__':
    path = "../../DataSets/ECG5000/"
    Data, Label = ReadFromCSVToKeras(path + "ECG5000_ALL.csv")
    Label -= 1
    training, validation, test = SplitDataset(Data, Label, 0.15, 0.1)

    history, result = TrainAndTestModel_OBJ(ModelFFBaseline, training, validation, test, 128, 15)
    pu.PlotModel(history, result)
