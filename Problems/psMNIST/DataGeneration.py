import numpy.random as rng
from Utility.DataUtil import *
import tensorflow.keras as ks


def LoadMNISTData():
    return ks.datasets.mnist.load_data()


def psMNISTDataset(shuffle=False, validationSplit=0.1, dataPartition=-1, seed=1509):
    """
    Creates a permuted sequential MNIST (psMNIST) dataset with optional shuffling and data partitioning.

    Parameters:
    - shuffle (bool): Whether to shuffle the dataset after creating it.
    - validationSplit (float): Fraction of the training data to use as validation data.
    - dataPartition (int): Number of samples to keep from the dataset (negative for all samples).
    - seed (int): Seed for random number generation to ensure reproducibility.

    Returns:
    - Tuple: A tuple containing the training and validation datasets, and the test dataset.
    """

    # Load the MNIST dataset
    ((training_sample, training_label), (test_sample, test_labels)) = LoadMNISTData()
    training_sample =training_sample/255
    test_sample=test_sample/255

    # Reshape the data to the psMNIST format
    training_sample = training_sample.reshape(training_sample.shape[0], -1, 1)
    test_sample = test_sample.reshape(test_sample.shape[0], -1, 1)

    # Generate a permutation of pixel indices based on the seed
    rng.seed(seed)
    perm = rng.permutation(training_sample.shape[1])

    #Optionally partition the data
    if dataPartition > 0:
        training_sample = training_sample[:dataPartition]
        training_label = training_label[:dataPartition]

    #Apply the permutation to the Data
    training_sample = training_sample[:, perm]
    test_sample = test_sample[:, perm]

    # Create dataset objects for the training and test data
    training_set = DataLabel(training_sample, training_label)
    test_set = DataLabel(test_sample, test_labels)

    # Shuffle the datasets if required
    if shuffle:
        training_set.Shuffle()
        test_set.Shuffle()

    # Split the training data into training and validation sets and return
    training_set, validation_set = training_set.SplitIn2(validationSplit)
    return DataSet.init(training_set, validation_set, test_set)
