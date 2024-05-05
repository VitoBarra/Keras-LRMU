import scipy as sp
import numpy as np


def ExtractFromArffToKeras(file, sequenceLength, expectedDataLength):
    arffData = sp.io.arff.loadarff(file)[0]
    arffData = [list(t) for t in arffData]
    data = [t[:-1] for t in arffData]
    label = [t[-1] for t in arffData]
    data = [np.array(t) for t in data]
    data = np.array(data)
    label = np.array(label).astype(np.byte)
    return data, label
