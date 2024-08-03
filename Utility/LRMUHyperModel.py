import keras_tuner as kt
import tensorflow.keras as ks
from ESN.layer import *
from keras.initializers import *
from Utility.LRMUModelBuilder import LRMUModelBuilder


class LRMUHyperModel(kt.HyperModel):

    def __init__(self, hyperModelName,problemName, sequenceLenght, classNuber=0, seed=0):
        super().__init__(hyperModelName)
        self.Seed = seed
        self.ProblemName = problemName
        self.SequenceLength = sequenceLenght
        self.ClassNumber = classNuber

        if classNuber < 1:
            self.ModelType = 1
        else:
            self.ModelType = 2

        self.LRMUBuilder = None
        self.ModelName = None
        self.UseESN = None
        self.ReservoirMode = None

    def LMU_ESN(self):
        self.ModelName = "LMU-ESN"
        self.UseESN = True
        self.ReservoirMode = False
        self.CreateLRMUBuilder()
        return self

    def LMU_RE(self):
        self.ModelName = "LMU-RE"
        self.UseESN = False
        self.ReservoirMode = True
        self.CreateLRMUBuilder()
        return self

    def LRMU(self):
        self.ModelName = "LRMU"
        self.UseESN = True
        self.ReservoirMode = True
        self.CreateLRMUBuilder()
        return self

    def CreateLRMUBuilder(self):
        self.LRMUBuilder = LRMUModelBuilder(self.ProblemName, self.ModelName, self.Seed)

    def selectCell(self, hp, hiddenUnit):
        if self.UseESN:
            spectraRadius = hp.Float("spectraRadius", min_value=0.8, max_value=1.3, step=0.05)
            leaky = hp.Float("leaky", min_value=0.5, max_value=1, step=0.1)
            inputScaler = 1
            hiddenCell = ReservoirCell(hiddenUnit, spectral_radius=spectraRadius, leaky=leaky,
                                       input_scaling=inputScaler)
        else:
            hiddenCell = ks.layers.SimpleRNNCell(hiddenUnit, kernel_initializer=GlorotUniform(self.Seed))
        return hiddenCell

    def LMUParam(self, hp):
        layerN = 1
        memoryDim = hp.Choice("memoryDim", values=[1, 2, 4, 8, 16, 32])
        order = hp.Choice("order", values=[1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64])
        theta = hp.Int("theta", min_value=16, max_value=258, step=16)
        hiddenUnit = hp.Int("hiddenUnit", min_value=16, max_value=16 * 20, step=16)
        return layerN, memoryDim, order, theta, self.selectCell(hp, hiddenUnit)

    def selectScaler(self, hp, reservoirMode, memoryToMemory, hiddenToMemory, useBias):
        InputToMemoryScaler = hp.Float("InputToMemoryScaler", min_value=0.5, max_value=2, step=0.25)
        memoryToMemoryScaler = None
        hiddenToMemoryScaler = None
        biasScaler = None
        if reservoirMode:
            if memoryToMemory:
                memoryToMemoryScaler = hp.Float("memoryToMemoryScaler", min_value=0.5, max_value=2, step=0.25)
            if hiddenToMemory:
                hiddenToMemoryScaler = hp.Float("hiddenToMemoryScaler", min_value=0.5, max_value=2, step=0.25)
            if useBias:
                biasScaler = hp.Float("biasScaler", min_value=0.5, max_value=2, step=0.25)

        return memoryToMemoryScaler, hiddenToMemoryScaler, InputToMemoryScaler, biasScaler

    def selectConnection(self, hp):
        memoryToMemory = hp.Boolean("memoryToMemory")
        hiddenToMemory = hp.Boolean("hiddenToMemory")
        inputToHiddenCell = hp.Boolean("inputToHiddenCell")
        useBias = hp.Boolean("useBias")
        return memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias, self.selectScaler(hp,self.ReservoirMode,
                                                                                             memoryToMemory,
                                                                                             hiddenToMemory, useBias)

    def constructHyperModel(self, hp):

        layerN, memoryDim, order, theta, hiddenCell = self.LMUParam(hp)
        memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias, scaler = self.selectConnection(hp)
        (memoryToMemoryScaler, hiddenToMemoryScaler, InputToMemoryScaler, biasScaler) = scaler

        self.LRMUBuilder.inputLayer(self.SequenceLength)
        self.LRMUBuilder.featureLayer(memoryDim, order, theta,
                                      self.ReservoirMode, hiddenCell,
                                      memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                                      memoryToMemoryScaler, hiddenToMemoryScaler, InputToMemoryScaler, biasScaler,
                                      layerN)
        if self.ModelType == 1:
            return self.LRMUBuilder.outputPrediction().composeModel().buildPrediction()
        elif self.ModelType == 2:
            return self.LRMUBuilder.outputClassification(self.ClassNumber).composeModel().buildClassification()
        else:
            return None

    def build(self, hp):
        return self.constructHyperModel(hp)
