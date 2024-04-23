from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
import numpy as np


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results



if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_label) = imdb.load_data(num_words=10000)

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    model = keras.models.Sequential([
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
