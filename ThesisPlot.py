import Problems.psMNIST as psMNIST
import Problems.MackeyGlass as MackeyGlass
import matplotlib as plt
from Utility.PlotUtil import *
import seaborn as sns
import numpy as np
import numpy.random as rng
from GlobalConfig import *


def plot_mackey_glass(X, Y, title=""):
    plt.figure(figsize=(14, 8))
    plt.title(title)
    plt.scatter(X[:, 0], Y[:, 0] - X[:, 0], s=8, alpha=0.7,
                c=np.arange(X.shape[0]), cmap=sns.cubehelix_palette(as_cmap=True))
    plt.plot(X[:, 0], Y[:, 0] - X[:, 0], c='black', alpha=0.2)
    plt.xlabel("$x(t)$")
    plt.ylabel("$y(t) - x(t)$")
    sns.despine(offset=15)

    ShowOrSavePlot(PLOT_DATAVIS_PATH, title)


def PlotMarkeyGlass(tau):
    chosenExemple = 0
    X, Y = MackeyGlass.data.generate_data(1, 5000, tau=tau)
    plot_mackey_glass(X[chosenExemple], Y[chosenExemple], title=f"Mackey Glass T{tau}")


def WritePlotMNIST(ax, image, label, variantName):
    ax.title.set_text(f"{variantName}-Digit = {label}")
    ax.title.set_size(40)

    ax.imshow(image)


def PlotMNISTVariant():
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=200)
    ((X_train, Y_train), test) = psMNIST.data.LoadMNISTData()
    chosenExemple = 144

    image = X_train[chosenExemple]
    label = Y_train[chosenExemple]
    image = image / 255

    WritePlotMNIST(axs[0], image, label, "MNIST")

    # Generate a permutation of pixel indices based on the seed
    image = np.reshape(image, (28 * 28, 1))
    rng.seed(0)
    perm = rng.permutation(image.shape[0])
    image = image[perm,: ]
    WritePlotMNIST(axs[1], image.reshape(28, 28), label, "pMNIST")

    ShowOrSavePlot(PLOT_DATAVIS_PATH, "MNIST_Variant")

    plt.figure(figsize=(5, 1.5), dpi=200)
    plt.imshow(image.reshape(8, 98))
    plt.title(f"psMNIST-Digit = {label}", fontsize=20)
    ShowOrSavePlot(PLOT_DATAVIS_PATH, "psMNIST_Series")


if __name__ == "__main__":
     # PlotMNISTVariant()
    #  for tau in [17, 30]:
    #     PlotMarkeyGlass(tau)
    #  ReadAndPlotAll(DATA_DIR, PLOTS_DIR, f"{psMNIST.conf.PROBLEM_NAME}", True)
    #  ReadAndPlotAll(DATA_DIR, PLOTS_DIR, f"{MackeyGlass.Config.PROBLEM_NAME}", False)
     PrintAllDataAllSubProblem(DATA_DIR, psMNIST.conf.PROBLEM_NAME, True)
     PrintAllDataAllSubProblem(DATA_DIR, MackeyGlass.Config.PROBLEM_NAME, False)
