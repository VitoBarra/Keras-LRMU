from Problems.psMNIST import DataGeneration as psMNIST
from Problems.MackeyGlass import DataGeneration as MackeyGlass
import matplotlib as plt
from Utility.PlotUtil import *
import seaborn as sns
import numpy as np
import numpy.random as rng
import tensorflow

PLOT_PATH = "./plots/DatasetVisualization"


def plot_mackey_glass(X, Y, title=""):
    plt.figure(figsize=(14, 8))
    plt.title(title)
    plt.scatter(X[:, 0], Y[:, 0] - X[:, 0], s=8, alpha=0.7,
                c=np.arange(X.shape[0]), cmap=sns.cubehelix_palette(as_cmap=True))
    plt.plot(X[:, 0], Y[:, 0] - X[:, 0], c='black', alpha=0.2)
    plt.xlabel("$x(t)$")
    plt.ylabel("$y(t) - x(t)$")
    sns.despine(offset=15)

    ShowOrSavePlot(PLOT_PATH, title)


def WritePlotMNIST(ax, image, label, title=""):
    ax.title.set_text(f"Digit = {label}")
    ax.title.set_size(40)

    ax.imshow(image.reshape(28, 28))


def PlotMarkeyGlass(tau):
    chosenExemple = 0
    X,Y= MackeyGlass.generate_data(1, 5000, seed=0, predict_length=15, tau=tau, washout=100,
                                     delta_t=1, center=True )
    plot_mackey_glass(X[0],Y[0],title=f"Mackey Glass T{tau}")


def PlotMNISTVariant():
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    ((X_train, Y_train), test) = psMNIST.LoadMNISTData()
    chosenExemple = 144

    image = X_train[chosenExemple]
    label = Y_train[chosenExemple]

    WritePlotMNIST(axs[ 0], image, label, "MNIST")
    # Generate a permutation of pixel indices based on the seed
    image = np.reshape(image, (28 * 28, 1))

    rng.seed(0)
    perm = rng.permutation(image.shape[0])
    image = image[perm, :]
    WritePlotMNIST(axs[1], image, label, "psMNIST")

    ShowOrSavePlot(PLOT_PATH, "MNIST Variant")


if __name__ == "__main__":
    PlotMNISTVariant()
    for tau in [17, 30]:
        PlotMarkeyGlass(tau)
