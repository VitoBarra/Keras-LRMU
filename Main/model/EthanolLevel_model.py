import keras as ks

import matplotlib.pyplot as plt
from Main.Util.ArffFormatUtill import *


def model():
    sequence_length = 1751
    dataLength = 504
    testLength = 500
    inputs = ks.Input(shape=(sequence_length, 1), name="ET_Input")
    feature = ks.layers.SimpleRNN(500, return_sequences=True)(inputs)
    feature = ks.layers.SimpleRNN(400, return_sequences=True)(feature)
    feature = ks.layers.SimpleRNN(300, return_sequences=True)(feature)
    feature = ks.layers.SimpleRNN(200, return_sequences=True)(feature)
    feature = ks.layers.SimpleRNN(100, return_sequences=False)(feature)
    outputs = ks.layers.Dense(5, activation="softmax")(feature)
    outputs = ks.layers.Reshape((5,))(outputs)
    model = ks.Model(inputs=inputs, outputs=outputs, name="EthanolLevelModel")
    model.summary()
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["categorical_accuracy"])
    return model


if __name__ == '__main__':
    path = "../../DataSets/EthanolLevel/"
    train_Data, train_Label = ReadFromCSVToKeras(path + "EthanolLevel_TRAIN.csv")
    test_Data, test_Label = ReadFromCSVToKeras(path + "EthanolLevel_TEST.csv")

    Validation_Data, Validation_Label = train_Data[450:], train_Label[450:]
    train_Data, train_Label = train_Data[:450], train_Label[:450]


    model = model()
    history = model.fit(train_Data, train_Label, epochs=10, validation_data=(Validation_Data, Validation_Label))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
