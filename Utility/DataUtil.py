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


class DataLabel(object):
    def __init__(self, data, label):
        self.Data = data
        self.Label = label

    def SplitDataset(self, validationParcent=0.15, testParcent=0.1):
        if (validationParcent + testParcent > 1):
            raise ValueError('Validation and test rate must be less than 1')
        dataLength = self.Data.shape[0]
        if dataLength != self.Label.shape[0]:
            raise ValueError('Data and label must have the same length')
        trainingBound = int(dataLength * (1 - validationParcent - testParcent))
        valBound = int(dataLength * validationParcent)
        training = DataLabel(self.Data[:trainingBound], self.Label[:trainingBound])
        validation = DataLabel(self.Data[trainingBound:trainingBound + valBound],
                               self.Label[trainingBound:trainingBound + valBound])
        test = DataLabel(self.Data[trainingBound + valBound:], self.Label[trainingBound + valBound:])
        return training, validation, test

    def SplitArray(self, rate=0.15):
        dataLength = self.Data.shape[0]
        if dataLength != self.Label.shape[0]:
            raise ValueError('Data and label must have the same length')

        splitIndex = int(dataLength * rate)
        dataSplit = DataLabel(self.Data[:splitIndex], self.Label[:splitIndex])
        self.Data = self.Data[splitIndex:]
        self.Label = self.Label[splitIndex:]
        return dataSplit

    def Concatenate(self, dataLable):
        self.Data = np.concatenate((self.Data, dataLable.Data) , axis=0)
        self.Label = np.concatenate((self.Label, dataLable.Label), axis=0)
