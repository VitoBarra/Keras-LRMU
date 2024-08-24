import os
import pickle

from matplotlib import pyplot as plt
import json


def PlotModelLoss(history, plotTitle='Problems', path=None, filename=None):
    plt.title(plotTitle)

    plt.plot(history.history['loss'])
    if hasattr(history.history, "val_loss"):
        plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    ShowOrSavePlot(path, filename)


def PlotModelAccuracy(history, plotTitle='Problems', path=None, filename=None):
    plt.title(plotTitle)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.plot(history.history['loss'])
    if hasattr(history.history, "val_loss"):
        ax1.plot(history.history['val_loss'])
        ax1.legend(['train', 'validation'], loc='upper left')
    else:
        ax1.legend(['train'], loc='upper left')

    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.legend(['train', 'validation'], loc='upper left')
    ShowOrSavePlot(path, filename)


def ShowOrSavePlot(path=None, filename=None):
    if path is None or path == '':
        plt.show()
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        if filename is None or filename == '':
            filename = 'model'
        plt.savefig(f"{path}/{filename}.png")


def SaveDataForPlotJson(path, problem_name, test_name, history, result):
    dir = f"{path}/{problem_name}/{test_name}"
    os.makedirs(dir, exist_ok=True)
    # Writing to sample.json
    with open(f"{dir}/history.bin", "wb") as outfile:
        pickle.dump(history.history,outfile)
    with open(f"{dir}/result.bin", "wb") as outfile:
        pickle.dump(result,outfile)


def ReadDataForPlotJson(path, problem_name, test_name):
    dir =f"{path}/{problem_name}/{test_name}"
    # Writing to sample.json

    with open(f"{dir}/history.bin", "rb") as inputfile:
        history = pickle.load(inputfile)
    with open(f"{dir}/{test_name}/result.bin", "rb") as inputfile:
        result = pickle.load(inputfile)
    return history, result


def ReadAndPlot(path, problem_name, test_name, classification):
    dir=f"{path}/{problem_name}"
    try:
        history, result = ReadDataForPlotJson(path, problem_name, test_name)
    except FileNotFoundError:
            print(f"file {dir}/{test_name} not found")
            return

    if classification:
        PlotModelAccuracy(history,test_name,dir,test_name)
    else:
        PlotModelLoss(history,test_name,dir,test_name)


#
# #Original code from: https://www.practicaldatascience.org/notebooks/class_5/week_5/46_making_plots_pretty.html
def PrityPlot(loss,mse=None,accuracy = None,baseline=None):

    fig, ax = plt.subplots(figsize=(6, 5))

    # Define font sizes
    SIZE_DEFAULT = 14
    SIZE_LARGE = 16
    plt.rc("font", family="Roboto")  # controls default font
    plt.rc("font", weight="normal")  # controls default font
    plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

    # Plot the baseline
    if baseline is not None:
        ax.plot(
            [loss[0], max(loss)],
            [baseline, baseline],
            label="Baseline",
            color="lightgray",
            linestyle="--",
            linewidth=1,
        )
        # Plot the baseline text
        ax.text(
            loss[-1] * 1.01,
            baseline,
            "Baseline",
            color="lightgray",
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="center",
        )

    # Define a nice color palette:
    colors = ["#2B2F42", "#8D99AE", "#EF233C"]
    labels = ["mse"]

    # Plot each of the main lines
    for i, label in enumerate(labels):
        # Line
        ax.plot(loss, label=label, color=colors[i], linewidth=2)

        # Text
        ax.text(
            loss["epoch"][-1] * 1.01,
            loss["loss"][i][-1],
            label,
            color=colors[i],
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="center",
        )

    # Hide the all but the bottom spines (axis lines)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["bottom"].set_bounds(min(loss["epoch"]), max(loss["epoch"]))

    ax.set_xlabel(r"epoch")
    ax.set_ylabel("loss")
    plt.savefig("great.png", dpi=300)


