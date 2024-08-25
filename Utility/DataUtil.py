import numpy as np
import numpy.random as rng
import tensorflow as tf
from typing import Type

def TimeSeriesSampleRate(data, rate=2):
    '''
    :param data: array of shape (Data, TimeSeriesPoints)
    :param rate: int, rate of sampling
    :return: "Data" of data with  Timeseries with "TimeSeriesPoints/rate" points
    '''

    dataRes = []
    for i in range(data.shape[0]):
        dataRes.append(data[i][::rate])  # get sample with rate "rate"
    return np.array(dataRes)


def CropTimeSeries(data, start=0, end=-1):
    '''
    :param data: array of shape (Data, TimeSeriesPoints)
    :param start:
    :param end:
    :return: cropped time series
    '''
    return np.array([t[start:end] for t in data])


class DataLabel(object):
    def __init__(self, data, label):
        dataLength = data.shape[0]
        if dataLength != label.shape[0]:
            raise ValueError('Data and label must have the same length')
        self.Data = data
        self.Label = label
        self.isCategorical=False

    def SplitDataset(self, validationParcent=0.15, testParcent=0.1):
        if validationParcent < 0 or testParcent < 0:
            raise ValueError('Validation and test rate must be in range [0,1]')
        if validationParcent + testParcent > 1:
            raise ValueError('Validation + test rate must be less than 1')
        if validationParcent <= 0:
            training, test = self.SplitIn2(testParcent)
            return training, None, test

        dataLength = self.Data.shape[0]
        trainingBound = int(dataLength * (1 - validationParcent - testParcent))
        valBound = int(dataLength * validationParcent)
        training = DataLabel(self.Data[:trainingBound], self.Label[:trainingBound])
        validation = DataLabel(self.Data[trainingBound:trainingBound + valBound],
                               self.Label[trainingBound:trainingBound + valBound])
        test = DataLabel(self.Data[trainingBound + valBound:], self.Label[trainingBound + valBound:])
        return training, validation, test

    def SplitIn2(self, rate=0.15):
        if rate <= 0:
            return self, None

        dataLength = self.Data.shape[0]
        splitIndex = int(dataLength * rate)
        dataSplit = DataLabel(self.Data[:splitIndex], self.Label[:splitIndex])
        self.Data = self.Data[splitIndex:]
        self.Label = self.Label[splitIndex:]
        return self, dataSplit

    def Concatenate(self, dataLabel):
        if self.isCategorical != dataLabel.isCategorical:
            raise ValueError("each of the dataLabe class must same type of label")
        self.Data = np.concatenate((self.Data, dataLabel.Data), axis=0)
        self.Label = np.concatenate((self.Label, dataLabel.Label), axis=0)

    def Shuffle(self, seed=0):
        rng.seed(seed)
        perm = rng.permutation(self.Data.shape[0])
        self.Data = self.Data[perm,]

    def ToCategoricalLabel(self):
        self.Label = tf.keras.utils.to_categorical(self.Label)
        self.isCategorical=True
