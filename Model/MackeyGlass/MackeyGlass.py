import collections

import keras_tuner
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import seaborn as sns
import LRMU as lrmu
from keras.initializers import *

import matplotlib.pyplot as plt

from Utility.DataUtil import SplitDataset
from Utility.ModelUtil import TrainAndTestModel_OBJ
from Utility.PlotUtil import *

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
                timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - 0.1 * history[-1]) / delta_t
            inp[timestep] = timeseries

        # Squash timeseries through tanh
        inp = np.tanh(inp - 1)
        samples.append(inp)
    return samples


def generate_data(n_series, length, seed=0,
                  predict_length=15, tau=17, washout=100, delta_t=1,
                  center=True):
    X = np.asarray(mackey_glass(
        sample_len=length + predict_length + washout, tau=tau,
        seed=seed, n_samples=n_series))
    X = X[:, washout:, :]
    if center:
        X -= np.mean(X)  # global mean over all batches, approx -0.066
    Y = X[:, predict_length:, :]
    X = X[:, :-predict_length, :]
    assert X.shape == Y.shape
    return X, Y


def cool_plot(X, Y, title=""):
    plt.figure(figsize=(14, 8))
    plt.title(title)
    plt.scatter(X[:, 0], Y[:, 0] - X[:, 0], s=8, alpha=0.7,
                c=np.arange(X.shape[0]), cmap=sns.cubehelix_palette(as_cmap=True))
    plt.plot(X[:, 0], Y[:, 0] - X[:, 0], c='black', alpha=0.2)
    plt.xlabel("$x(t)$")
    plt.ylabel("$y(t) - x(t)$")
    sns.despine(offset=15)

    ShowOrSavePlot("./plots", "MackeyGlass")


def ModelLRMU_SelectedHP():
    return ModelLRMU(2, 16, 416, 1.05,
                     True, None,
                     False, True, False, False, 0)


def ModelLRMU(memoryDim, order, hiddenUnit, spectraRadius, reservoirMode, hiddenCell,
              memoryToMemory, hiddenToMemory, inputToCell, useBias, seed):
    inputs = ks.Input(shape=(SEQUENCE_LENGHT, 1), name="Mackey-Glass_Input_LRMU")
    feature = lrmu.LRMU(memoryDimension=memoryDim, order=order, theta=32, hiddenUnit=hiddenUnit,
                        spectraRadius=spectraRadius, reservoirMode=reservoirMode, hiddenCell=hiddenCell,
                        memoryToMemory=memoryToMemory, hiddenToMemory=hiddenToMemory, inputToHiddenCell=inputToCell,
                        useBias=useBias, seed=seed)(inputs)
    outputs = ks.layers.Dense(1, activation="linear", kernel_initializer=GlorotUniform(seed))(feature)
    model = ks.Model(inputs=inputs, outputs=outputs, name="Mackey-Glass_LRMU_Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="mse",
                  metrics=["mse"])
    return model


def ModelLRMUWhitTuning(hp):
    memoryDim =2
    order = 16
    hiddenUnit = 416
    spectraRadius = 1.05
    memoryToMemory = hp.Boolean("memoryToMemory")
    hiddenToMemory = hp.Boolean("hiddenToMemory")
    inputToHiddenCell = hp.Boolean("inputToCell")
    useBias = hp.Boolean("useBias")

    reservoirMode = True
    hiddenCell = None
    seed = 0
    return ModelLRMU(memoryDim, order, hiddenUnit, spectraRadius, reservoirMode, hiddenCell,
                     memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias, seed)


def FullTraining(training, validation, test):
    history, result = TrainAndTestModel_OBJ(ModelLRMU_SelectedHP, training, validation, test, 64, 15, "val_loss")

    print("test loss:", result[0])
    print("test mse:", result[1])
    PlotModelLoss(history, "Model LRMU", f"./plots/{PROBLEM_NAME}", f"{PROBLEM_NAME}_LRMU_ESN")


def TunerTraining(training, validation):
    dirName = f"./logs/{PROBLEM_NAME}_bool_linear"
    tuner = keras_tuner.RandomSearch(
        ModelLRMUWhitTuning,
        max_trials=200,
        project_name=f"{PROBLEM_NAME}",
        executions_per_trial=1,
        # Do not resume the previous search in the same directory.
        overwrite=True,
        objective="val_loss",
        # Set a directory to store the intermediate results.
        directory=f"{dirName}/tmp",
    )

    tuner.search(
        training.Data,
        training.Label,
        validation_data=(validation.Data, validation.Label),
        epochs=15,
        # Use the TensorBoard callback.
        callbacks=[ks.callbacks.TensorBoard(f"{dirName}")],
    )


def Run(fullTraining=True):
    data, label = generate_data(128, SEQUENCE_LENGHT)
    training, validation, test = SplitDataset(data, label, 0.1, 0.1)
    print(data.shape, label.shape)

    if fullTraining:
        FullTraining(training, validation, test)
    else:
        TunerTraining(training, validation)
