from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results


def plot_loss(history_in):
    history_dict = history_in.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('plot/test_loss.png')
    plt.close()


if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    y_train = np.asarray(train_labels).astype("float32")
    y_test = np.asarray(test_labels).astype("float32")

    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    model = keras.models.Sequential(name="imdb_test")
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    model.build(input_shape=(None, 10000))
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))
    plot_loss(history)
    model.summary()

#keras.utils.plot_model(model, "imdb_model.png", show_shapes=True)
