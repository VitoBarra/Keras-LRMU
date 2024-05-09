import keras as ks
import matplotlib.pyplot as plt
from Main.Util.ArffFormatUtill import *
import Main.Layer.keras_lmu as kslu
import Main.Util.PlotUtil as pu
from Main.Util.Util import TrainAndTestModel


def model():
    sequence_length = 1751
    numberOfClass = 4
    inputs = ks.Input(shape=(sequence_length, 1), name="ET_Input")
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
    model = ks.Model(inputs=inputs, outputs=outputs, name="EthanolLevelModel")

    model.summary()
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == '__main__':
    path = "../../DataSets/EthanolLevel/"
    train_Data, train_Label = ReadFromCSVToKeras(path + "EthanolLevel_TRAIN.csv")
    test_Data, test_Label = ReadFromCSVToKeras(path + "EthanolLevel_TEST.csv")
    train_Label -= 1
    test_Label -= 1

    Validation_Data, Validation_Label = train_Data[450:], train_Label[450:]
    train_Data, train_Label = train_Data[:450], train_Label[:450]

    history, result = TrainAndTestModel(model, train_Data, train_Label,
                                        Validation_Data, Validation_Label,
                                        test_Data, test_Label,
                                        32, 15)
    pu.PlotModel(history, result)
