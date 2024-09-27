import numpy as np
import numpy.random as rng
import tensorflow as tf


def TimeSeriesSampleRate(data, rate=2):
    """
    :param data: array of shape (Data, TimeSeriesPoints)
    :param rate: int, rate of sampling
    :return: "Data" of data with  Timeseries with "TimeSeriesPoints/rate" points
    """

    dataRes = []
    for i in range(data.shape[0]):
        dataRes.append(data[i][::rate])  # get sample with rate "rate"
    return np.array(dataRes)


def CropTimeSeries(data, start=0, end=-1):
    """
    :param data: array of shape (Data, TimeSeriesPoints)
    :param start:
    :param end:
    :return: cropped time series
    """
    return np.array([t[start:end] for t in data])


class DataLabel(object):
    def __init__(self, data, label):
        dataLength = data.shape[0]
        if dataLength != label.shape[0]:
            raise ValueError('Data and label must have the same length')
        self.Data = data
        self.Label = label
        self.isCategorical = False

    def Concatenate(self, dataLabel):
        if self.isCategorical != dataLabel.isCategorical:
            raise ValueError("each of the dataLabel class must same type of label")
        self.Data = np.concatenate((self.Data, dataLabel.Data), axis=0)
        self.Label = np.concatenate((self.Label, dataLabel.Label), axis=0)

    def Shuffle(self, seed=0):
        rng.seed(seed)
        perm = rng.permutation(self.Data.shape[0])
        self.Data = self.Data[perm,]
        self.Label = self.Label[perm,]

    def SplitDataset(self, validationPercent=0.15, testPercent=0.1):
        if validationPercent < 0 or testPercent < 0:
            raise ValueError('Validation and test rate must be in range [0,1]')
        if validationPercent + testPercent > 1:
            raise ValueError('Validation + test rate must be less than 1')
        if validationPercent <= 0:
            training, test = self.SplitIn2(testPercent)
            return training, None, test

        dataLength = self.Data.shape[0]
        trainingBound = int(dataLength * (1 - validationPercent - testPercent))
        valBound = int(dataLength * validationPercent)
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
        Data = DataLabel(self.Data[splitIndex:], self.Label[splitIndex:])
        return Data, dataSplit

    def Normalize(self, mean=None, std=None):
        if mean is None:
            mean = np.mean(self.Data, axis=0)
        if std is None:
            std = np.std(self.Data, axis=0)
        self.Data = (self.Data - mean) / std

    def ToCategoricalLabel(self):
        self.Label = tf.keras.utils.to_categorical(self.Label)
        self.isCategorical = True

    def Slice(self, start, end):
        self.Data = self.Data[start:end]
        self.Label = self.Label[start:end]
        assert self.Data.shape[0] == self.Label.shape[0]



class DataSet(object):

    def __init__(self, data, label):
        if data is not None and label is not None:
            self.Data = DataLabel(data, label)
        else:
            self.Data = None

        self.Training = None
        self.Validation = None
        self.Test = None

    def SplitDataset(self, validationPercent=0.15, testPercent=0.1):
        self.Training, self.Validation, self.Test = self.Data.SplitDataset(validationPercent, testPercent)
        self.Data = None
        return self

    def Normalize(self, mean=None, std=None):
        if self.Data is None:
            raise ValueError('normalize function must be called before splitDataset')
        self.Data.Normalize(mean, std)
        return self

    def ToCategoricalLabel(self):
        if self.Data is not None:
            self.Data.ToCategoricalLabel()
        if self.Training is not None:
            self.Training.ToCategoricalLabel()
        if self.Validation is not None:
            self.Validation.ToCategoricalLabel()
        if self.Test is not None:
            self.Test.ToCategoricalLabel()
        return self

    def Shuffle(self, seed=0):
        if self.Data is None:
            raise ValueError('shuffle function must be called before splitDataset')
        self.Data.Shuffle(seed)
        return self

    def Unpack(self):
        return self.Training, self.Validation, self.Test

    def PrintSplit(self):
        string = ""
        if self.Training is not None:
            string += f"Training:{self.Training.Data.shape}, "
        if self.Validation is not None:
            string += f"Validation:{self.Validation.Data.shape}, "
        if self.Test is not None:
            string += f"Test:{self.Test.Data.shape}"

        print(string)

    def FlattenSeriesData(self):
        if self.Data is not None:
            self.Data.Data = self.Data.Data.reshape(self.Data.Data.shape[0], -1)
            self.Data.Label = self.Data.Label.reshape(self.Data.Label.shape[0], -1)
        if self.Training is not None:
            self.Training.Data = self.Training.Data.reshape(self.Training.Data.shape[0], -1)
            self.Training.Label = self.Training.Label.reshape(self.Training.Label.shape[0], -1)
        if self.Validation is not None:
            self.Validation.Data = self.Validation.Data.reshape(self.Validation.Data.shape[0], -1)
            self.Validation.Label = self.Validation.Label.reshape(self.Validation.Label.shape[0], -1)
        if self.Test is not None:
            self.Test.Data = self.Test.Data.reshape(self.Test.Data.shape[0], -1)
            self.Test.Label = self.Test.Label.reshape(self.Test.Label.shape[0], -1)

    @classmethod
    def init(cls, training, Validation, Test):
        instance = DataSet(None, None)

        instance.Training = training
        instance.Validation = Validation
        instance.Test = Test
        return instance
