from LRMU import LRMU
import tensorflow.keras as ks
from keras.initializers import *

def GenerateLRMUFeatureLayer(inputs,
                             memoryDim, order, theta,
                             hiddenUnit,spectraRadius, leaky,
                             reservoirMode, hiddenCell,
                             memoryToMemory, hiddenToMemory, inputToCell, useBias,
                             memoryToMemoryScaler, hiddenToMemoryScaler, inputToHiddenCellScaler, biasScaler,
                             seed, layerN=1):
    feature = LRMU(memoryDim, order,theta,
                   hiddenUnit,spectraRadius,leaky,
                   reservoirMode,hiddenCell,
                   memoryToMemory,hiddenToMemory, inputToCell,useBias,
                   memoryToMemoryScaler, hiddenToMemoryScaler,inputToHiddenCellScaler, biasScaler,
                   seed, returnSequences=layerN > 1)(inputs)
    for i in range(layerN - 1):
        feature = LRMU(memoryDim, order,theta,
                       hiddenUnit,spectraRadius,leaky,
                       reservoirMode, hiddenCell,
                       memoryToMemory,hiddenToMemory, inputToCell,useBias,
                       memoryToMemoryScaler, hiddenToMemoryScaler, inputToHiddenCellScaler, biasScaler,
                       seed, returnSequences=i != layerN - 2)(feature)
    return feature


def Model_LRMU_Classification(PROBLEM_NAME, model_type, SEQUENCE_LENGHT, CLASS_NUMBER,
                              memoryDim, order, theta, hiddenUnit,
                              spectraRadius, leaky,
                              reservoirMode, hiddenCell,
                              memoryToMemory, hiddenToMemory, inputToCell, useBias,
                              memoryToMemoryScaler, hiddenToMemoryScaler, inputToHiddenCellScaler, biasScaler,
                              seed, layerN):
    inputs = ks.Input(shape=(SEQUENCE_LENGHT, 1), name=f"{model_type}_Input")
    feature = GenerateLRMUFeatureLayer(inputs,
                                                 memoryDim, order, theta,
                                                 hiddenUnit, spectraRadius, leaky,
                                                 reservoirMode, hiddenCell,
                                                 memoryToMemory, hiddenToMemory, inputToCell, useBias,
                                                 memoryToMemoryScaler, hiddenToMemoryScaler, inputToHiddenCellScaler, biasScaler,
                                                 seed, layerN)
    outputs = ks.layers.Dense(CLASS_NUMBER, activation="softmax", kernel_initializer=GlorotUniform(seed))(feature)

    model = ks.Model(inputs=inputs, outputs=outputs, name=f"{PROBLEM_NAME}_{model_type}_Model")
    model.summary()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def Model_LRMU_Prediction(PROBLEM_NAME, model_type, SEQUENCE_LENGHT,
                          memoryDim, order, theta, hiddenUnit,
                          spectraRadius, leaky,
                          reservoirMode, hiddenCell,
                          memoryToMemory, hiddenToMemory, inputToCell, useBias,
                          memoryToMemoryScaler, hiddenToMemoryScaler, inputToHiddenCellScaler, biasScaler,
                          seed, layerN):
    inputs = ks.Input(shape=(SEQUENCE_LENGHT, 1), name=f"{model_type}_Input")
    feature = GenerateLRMUFeatureLayer(inputs,
                                                 memoryDim, order, theta,
                                                 hiddenUnit, spectraRadius, leaky,
                                                 reservoirMode, hiddenCell,
                                                 memoryToMemory, hiddenToMemory, inputToCell, useBias,
                                                 memoryToMemoryScaler, hiddenToMemoryScaler, inputToHiddenCellScaler, biasScaler,
                                                 seed, layerN)

    outputs = ks.layers.Dense(1, activation="linear", kernel_initializer=GlorotUniform(seed))(feature)

    model = ks.Model(inputs=inputs, outputs=outputs, name=f"{PROBLEM_NAME}_{model_type}_Model")
    model.summary()
    model.compile(optimizer="adam",
                      loss="mse",
                      metrics=["mse"])
    return model

