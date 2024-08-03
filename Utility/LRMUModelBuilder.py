import tensorflow.keras as ks
from keras.initializers import *
from LRMU import LRMU




class LRMUModelBuilder:

    def __init__(self, problemName, modelName, seed=0):
        self.Input = None
        self.Model = None
        self.ProblemName = problemName
        self.ModelType = modelName
        self.Seed = seed

    def inputLayer(self, sequenceLenght):
        self.Input = ks.Input(shape=(sequenceLenght, 1), name=f"{self.ModelType}_Input")
        return self

    def featureLayer(self, memoryDim, order, theta,
                     reservoirMode, hiddenCell,
                     memoryToMemory, hiddenToMemory, inputToCell, useBias,
                     memoryToMemoryScaler, hiddenToMemoryScaler, inputToHiddenCellScaler, biasScaler,
                     layerN):
        feature = LRMU(memoryDim, order, theta,
                       reservoirMode, hiddenCell,
                       memoryToMemory, hiddenToMemory, inputToCell, useBias,
                       memoryToMemoryScaler, hiddenToMemoryScaler, inputToHiddenCellScaler, biasScaler,
                       self.Seed, returnSequences=layerN > 1)(self.Input)
        for i in range(layerN - 1):
            feature = LRMU(memoryDim, order, theta,
                           reservoirMode, hiddenCell,
                           memoryToMemory, hiddenToMemory, inputToCell, useBias,
                           memoryToMemoryScaler, hiddenToMemoryScaler, inputToHiddenCellScaler, biasScaler,
                           self.Seed, returnSequences=i != layerN - 2)(feature)
        self.Feature = feature
        return self

    def outputClassification(self, classNuber):
        self.Outputs = ks.layers.Dense(classNuber, activation="softmax", kernel_initializer=GlorotUniform(self.Seed))(
            self.Feature)

        return self

    def outputPrediction(self):
        self.Outputs = ks.layers.Dense(1, activation="linear", kernel_initializer=GlorotUniform(self.Seed))(
            self.Feature)
        return self

    def composeModel(self):
        self.Model = ks.Model(inputs=self.Input, outputs=self.Outputs,
                              name=f"{self.ProblemName}_{self.ModelType}_Model")
        self.Model.summary()
        return self

    def buildPrediction(self):
        self.Model.compile(optimizer="adam",
                           loss="mse",
                           metrics=["mse"])
        return self.Model

    def buildClassification(self):
        self.Model.compile(optimizer="adam",
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])
        return self.Model
