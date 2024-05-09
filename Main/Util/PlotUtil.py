import matplotlib.pyplot as plt


def PlotModel(history, testResult):
    plt.title('model loss')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.plot(history.history['val_loss'])
    ax1.plot(history.history['loss'])
    ax1.legend(['train', 'validation', 'test'], loc='upper left')

    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    ax2.legend(['train'], loc='upper left')
    ax2.plot(testResult)
    fig.show()
