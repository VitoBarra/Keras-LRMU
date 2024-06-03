import os

from matplotlib import pyplot as plt



def PlotModelLoss(history, modelName='Problems', path=None, filename=None):
    plt.title(modelName)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    ShowOrSavePlot(path, filename)


def PlotModelAccuracy(history, modelName='Problems', path=None, filename=None):
    plt.title(modelName)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.legend(['train', 'validation'], loc='upper left')

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



