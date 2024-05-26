import numpy as np


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


def SplitDataset(Data, Label, validationParcent=0.15, testParcent=0.1):
    if (validationParcent + testParcent > 1):
        raise ValueError('Validation and test rate must be less than 1')
    dataLength = Data.shape[0]
    if dataLength != Label.shape[0]:
        raise ValueError('Data and label must have the same length')
    trainingBound = int(dataLength * (1-validationParcent- testParcent))
    valBound = int(dataLength * validationParcent)
    training = DataLabel(Data[:trainingBound], Label[:trainingBound])
    validation = DataLabel(Data[trainingBound:trainingBound + valBound],
                           Label[trainingBound:trainingBound + valBound])
    test = DataLabel(Data[trainingBound + valBound:], Label[trainingBound + valBound:])
    return training, validation, test


def SplitArray(data, label, rate=0.15):
    dataLength = data.shape[0]
    if dataLength != label.shape[0]:
        raise ValueError('Data and label must have the same length')

    splitIndex = int(dataLength * rate)
    return data[splitIndex:], label[splitIndex:], data[:splitIndex], label[:splitIndex]


class DataLabel(object):
    def __init__(self, data, label):
        self.Data = data
        self.Label = label
