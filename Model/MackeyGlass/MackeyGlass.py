import collections

import keras_tuner
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import seaborn as sns
import LRMU as lrmu

import matplotlib.pyplot as plt

from Utility.DataUtil import SplitDataset
from Utility.ModelUtil import TrainAndTestModel_OBJ
from Utility.PlotUtil import PrintAccuracy, PlotModelAccuracy, PrintLoss, PlotModelLoss

PROBLEM_NAME = "MackeyGlass"
SEQUENCE_LENGHT = 5000


def mackey_glass(sample_len=1000, tau=17, delta_t=10, seed=None, n_samples=1):
    # Adapted from https://github.com/mila-iqia/summerschool2015/blob/master/rnn_tutorial/synthetic.py
    '''
    mackey_glass(sample_len=1000, tau=17, seed = None, n_samples = 1) -> input
    Generate the Mackey Glass time-series. Parameters are:
        - sample_len: length of the time-series in timesteps. Default is 1000.
        - tau: delay of the MG - system. Commonly used values are tau=17 (mild
          chaos) and tau=30 (moderate chaos). Default is 17.
        - seed: to seed the random generator, can be used to generate the same
          timeseries at each invocation.
        - n_samples : number of samples to generate
    '''
    history_len = tau * delta_t
    # Initial conditions for the history of the system
    timeseries = 1.2

    if seed is not None:
        np.random.seed(seed)

    samples = []

    for _ in range(n_samples):
        history = collections.deque(1.2 * np.ones(history_len) + 0.2 * \
                                    (np.random.rand(history_len) - 0.5))
        # Preallocate the array for the time-series
        inp = np.zeros((sample_len, 1))

        for timestep in range(sample_len):
            for _ in range(delta_t):
                xtau = history.popleft()
                history.append(timeseries)
                timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - \
                                            0.1 * history[-1]) / delta_t
            inp[timestep] = timeseries

        # Squash timeseries through tanh
        inp = np.tanh(inp - 1)
        samples.append(inp)
    return samples


def generate_data(n_batches, length, split=0.5, seed=0,
                  predict_length=15, tau=17, washout=100, delta_t=1,
                  center=True):
    X = np.asarray(mackey_glass(
        sample_len=length + predict_length + washout, tau=tau,
        seed=seed, n_samples=n_batches))
    X = X[:, washout:, :]
    cutoff = int(split * n_batches)
    if center:
        X -= np.mean(X)  # global mean over all batches, approx -0.066
    Y = X[:, predict_length:, :]
    X = X[:, :-predict_length, :]
    assert X.shape == Y.shape
    return ((X[:cutoff], Y[:cutoff]),
            (X[cutoff:], Y[cutoff:]))


def cool_plot(X, Y, title=""):
    plt.figure(figsize=(14, 8))
    plt.title(title)
    plt.scatter(X[:, 0], Y[:, 0] - X[:, 0], s=8, alpha=0.7,
                c=np.arange(X.shape[0]), cmap=sns.cubehelix_palette(as_cmap=True))
    plt.plot(X[:, 0], Y[:, 0] - X[:, 0], c='black', alpha=0.2)
    plt.xlabel("$x(t)$")
    plt.ylabel("$y(t) - x(t)$")
    sns.despine(offset=15)

    plt.show()


def ModelLRMU_P():
    return ModelLRMU(1, 256, 212, 1.18,
                     True, None, False, True, False, False, 0)


def ModelLRMU(memoryDim, order, hiddenUnit, spectraRadius, reservoirMode, hiddenCell,
              memoryToMemory, hiddenToMemory, inputToCell, useBias, seed):
    inputs = ks.Input(shape=(SEQUENCE_LENGHT, 1), name="Mackey-Glass_Input_LRMU")
    feature = lrmu.LRMU(memoryDimension=memoryDim, order=order, theta=SEQUENCE_LENGHT, hiddenUnit=hiddenUnit,
                        spectraRadius=spectraRadius, reservoirMode=reservoirMode, hiddenCell=hiddenCell,
                        memoryToMemory=memoryToMemory, hiddenToMemory=hiddenToMemory, inputToHiddenCell=inputToCell,
                        useBias=useBias, seed=seed)(inputs)
    outputs = ks.layers.Dense(SEQUENCE_LENGHT, activation="linear")(feature)
    model = ks.Model(inputs=inputs, outputs=outputs, name="Mackey-Glass_LRMU_Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="mse",
                  metrics=["mse"])
    return model


def ModelLRMUWhitTuning(hp):
    memoryDim = hp.Int("memoryDim", min_value=1, max_value=16, step=1)
    order = hp.Int("order", min_value=16, max_value=256, step=16)
    hiddenUnit = hp.Int("hiddenUnit", min_value=128, max_value=512, step=16)
    spectraRadius = hp.Float("spectraRadius", min_value=0.8, max_value=1.3, step=0.025)
    memoryToMemory = True
    hiddenToMemory = True
    inputToHiddenCell = True
    useBias = True

    reservoirMode = True
    hiddenCell = None
    seed = 0
    return ModelLRMU(memoryDim, order, hiddenUnit, spectraRadius, reservoirMode, hiddenCell,
                     memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias, seed)


def FullTraining(training, validation, test):
    history, result = TrainAndTestModel_OBJ(ModelLRMU_P, training, validation, test, 64, 10, "val_loss")

    PrintLoss(result)
    PlotModelLoss(history, "Model LRMU", f"./plots/{PROBLEM_NAME}", f"{PROBLEM_NAME}_LRMU_ESN")


def TunerTraining(training, validation, test):
    tuner = keras_tuner.GridSearch(
        ModelLRMUWhitTuning,
        project_name=f"{PROBLEM_NAME}",
        executions_per_trial=1,
        # Do not resume the previous search in the same directory.
        overwrite=True,
        objective="val_loss",
        # Set a directory to store the intermediate results.
        directory=f"./logs/{PROBLEM_NAME}/tmp",
    )

    tuner.search(
        training.Data,
        training.Label,
        validation_data=(validation.Data, validation.Label),
        epochs=2,
        # Use the TensorBoard callback.
        callbacks=[ks.callbacks.TensorBoard(f"./logs/{PROBLEM_NAME}_2")],
    )


def Run(fullTraining=True):
    (train_X, train_Y), (test_X, test_Y) = generate_data(128, SEQUENCE_LENGHT)
    cool_plot(train_X[0], train_Y[0])

    data = np.concatenate((train_X, test_X), axis=0)
    label = np.concatenate((train_Y, test_Y), axis=0)
    training, validation, test = SplitDataset(data, label, 0.1, 0.1)

    if fullTraining:
        FullTraining(training, validation, test)
    else:
        TunerTraining(training, validation, test)
