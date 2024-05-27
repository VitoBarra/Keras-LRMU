import os

from matplotlib import pyplot as plt


def PlotModel(history,path= None,filename= None):
    plt.title('model loss')

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

    if path is None or path is '':
        plt.show()
    else :
        if not os.path.exists(path):
            os.makedirs(path)
        if filename is None or filename is '':
            filename = 'model'
        plt.savefig(f"{path}/{filename}.png")


def PrintAccuracy(result):
    print('Test loss:', result[0])
    print('Test accuracy:', result[1])
    return result
