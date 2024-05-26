import numpy as np
import random
from numpy import linalg as la




def CalculateSpectralRadius(matrix):
    """Calculate the spectral radius of a matrix"""
    return max(abs(la.eigvals(matrix)))


def GenerateWeightMatrix(n, Value, probability):
    """
    Generate
    :param n: matrix size
    :param Value: value and probability of the weight
    :return : weight matrix
    """

    # draw value v with probability p, get v and p from valueProb

    if len(Value) != len(probability):
        raise ValueError('Value and probability must have the same length')
    if sum(probability) != 1:
        raise ValueError('Sum of probability must be 1')

    weight = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            weight[i][j] = DrawRandomValueWithProbability(Value, probability)

    return weight


def DrawRandomValueWithProbability(value, probability):
    """
    Draw value v with probability p, get v and p from valueProb
    :param value: value and probability of the weight
    :return: value
    """
    partialProSum = np.cumsum(probability)
    return value[np.argmax(random.random() < partialProSum)]




def ScalingFactorToGetSpectralRadius(matrix, target):
    eighen = CalculateSpectralRadius(matrix)
    return target/eighen

def GetMatrixWithSpectralRadius(size, tagetSR):
    weight = GenerateWeightMatrix(size, [-1,0 ,1],[ 0.25, 0.5, 0.25])
    return weight * ScalingFactorToGetSpectralRadius(weight, tagetSR)
