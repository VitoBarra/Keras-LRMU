import numpy as np
import keras as ks
import scipy as sp
import matplotlib.pyplot as plt
from Main.Util.ArffFormatUtill import ExtractFromArffToKeras

if __name__ == '__main__':
    sequence_length = 140
    Ecg5000_train, Ecg5000_train_label = ExtractFromArffToKeras("../../DataSets/ECG5000/ECG5000_TRAIN.arff",
                                                                sequence_length, 500)
    Ecg5000_test, Ecg5000_test_label = ExtractFromArffToKeras("../../DataSets/ECG5000/ECG5000_TEST.arff",
                                                              sequence_length, 4500)

    Ecg5000_val, Ecg5000_val_label = Ecg5000_train[:50], Ecg5000_train_label[:50]

    inputs = ks.Input(shape=(sequence_length, 1), name="ECG5000_Input")
    feature = ks.layers.SimpleRNN(500, return_sequences=True)(inputs)
    feature = ks.layers.SimpleRNN(400, return_sequences=True)(feature)
    feature = ks.layers.SimpleRNN(500, return_sequences=False)(feature)
    outputs = ks.layers.Dense(5, activation="softmax")(feature)


    model = ks.Model(inputs=inputs, outputs=outputs, name="ECG5000Model")
    model.summary()
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
    history = model.fit(Ecg5000_train, Ecg5000_train_label, epochs=10, validation_data=(Ecg5000_val, Ecg5000_val_label))
