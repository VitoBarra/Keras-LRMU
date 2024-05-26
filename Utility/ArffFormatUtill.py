import scipy as sp
import numpy as np


def ReadFromArffToKeras(file):
    arffData = sp.io.arff.loadarff(file)[0]
    arffData = [list(t) for t in arffData]
    data = [t[:-1] for t in arffData]
    label = [t[-1] for t in arffData]
    data = [np.array(t) for t in data]
    data = np.array(data)
    label = np.array(label).astype(np.byte)

    return data, label


def SaveKerasToCSV(path, data, label):
    if len(data) != len(label):
        raise ValueError('Data and label must have the same length')

    with open(path, 'w') as f:
        for i in range(len(data)):
            f.write(','.join([str(t) for t in data[i]]))
            f.write(',' + str(label[i]) + '\n')


def ReadFromCSVToKeras(file):
    with open(file, 'r') as f:
        data = []
        label = []
        for line in f:
            line = line.strip()
            line = line.split(',')
            data.append([float(t) for t in line[:-1]])
            label.append(int(line[-1]))
        data = np.array(data)
        label = np.array(label).astype(np.byte)
        return data, label


def ArffToCSV(file):
    data, label = ReadFromArffToKeras(file + ".arff")
    SaveKerasToCSV(file + ".csv", data, label)
    return data, label
