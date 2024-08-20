import keras_tuner as kt
import tensorflow.keras as ks
from ESN.layer import *
from keras.initializers import *
from Utility.LRMUModelBuilder import LRMUModelBuilder


class LRMUHyperModel(kt.HyperModel):

    def __init__(self, hyperModelName, problemName, sequenceLenght, classNuber=0, seed=0):
        super().__init__(hyperModelName)
        self.UseLeaky = True
        self.UseInputScaler = False
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
        self.ReservoirEncoders = None

    def LMU_ESN(self, useLeaky=True, useInputScaler=False):
        self.ModelName = "LMU-ESN"
        self.UseESN = True
        self.UseLeaky = useLeaky
        self.UseInputScaler = useInputScaler
        self.ReservoirEncoders = False
        self.CreateLRMUBuilder()
        return self

    def LMU_RE(self):
        self.ModelName = "LMU-RE"
        self.UseESN = False
        self.ReservoirEncoders = True
        self.CreateLRMUBuilder()
        return self

    def LRMU(self, useLeaky=True, useInputScaler=False):
        self.ModelName = "LRMU"
        self.UseESN = True
        self.UseLeaky = useLeaky
        self.UseInputScaler = useInputScaler
        self.ReservoirEncoders = True
        self.CreateLRMUBuilder()
        return self

    def CreateLRMUBuilder(self):
        self.LRMUBuilder = LRMUModelBuilder(self.ProblemName, self.ModelName, self.Seed)

    def selectCell(self, hp, hiddenUnit):
        if not self.UseESN:
            hiddenCell = ks.layers.SimpleRNNCell(hiddenUnit, kernel_initializer=GlorotUniform(self.Seed))
        else:
            spectraRadius = hp.Float("ESN_spectraRadius", min_value=0.8, max_value=1.3, step=0.05)
            leaky = hp.Float("ESN_leaky", min_value=0.5, max_value=1, step=0.1) if self.UseLeaky else 1
            inputScaler = hp.Float("ESN_inputScaler", min_value=0.5, max_value=2,
                                   step=0.25) if self.UseInputScaler else 1
            hiddenCell = ReservoirCell(hiddenUnit, spectral_radius=spectraRadius, leaky=leaky,
                                       input_scaling=inputScaler)
        return hiddenCell

    def LMUParam(self, hp):
        layerN = 1
        memoryDim = hp.Choice("memoryDim", values=[1, 2, 4, 8, 16, 32])
        order = hp.Choice("order", values=[1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64])
        theta = hp.Int("theta", min_value=16, max_value=258, step=16)
        hiddenUnit = hp.Int("hiddenUnit", min_value=16, max_value=16 * 20, step=16)
        return layerN, memoryDim, order, theta, self.selectCell(hp, hiddenUnit)

    def selectScaler(self, hp, reservoirEncoders, memoryToMemory, hiddenToMemory, useBias):
        InputEncoderScaler = None
        memoryEncoderScaler = None
        hiddenEncoderScaler = None
        biasScaler = None

        if reservoirEncoders:
            InputEncoderScaler = hp.Float("InputEncoderScaler", min_value=0.5, max_value=2, step=0.25)
            if memoryToMemory:
                memoryEncoderScaler = hp.Float("memoryEncoderScaler", min_value=0.5, max_value=2, step=0.25)
            if hiddenToMemory:
                hiddenEncoderScaler = hp.Float("hiddenEncoderScaler", min_value=0.5, max_value=2, step=0.25)
            if useBias:
                biasScaler = hp.Float("biasScaler", min_value=0.5, max_value=2, step=0.25)

        return memoryEncoderScaler, hiddenEncoderScaler, InputEncoderScaler, biasScaler

    def selectConnection(self, hp):
        memoryToMemory = hp.Boolean("memoryToMemory")
        hiddenToMemory = hp.Boolean("hiddenToMemory")
        inputToHiddenCell = hp.Boolean("inputToHiddenCell")
        useBias = hp.Boolean("useBias")
        return memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias, self.selectScaler(hp, self.ReservoirEncoders,
                                                                                             memoryToMemory,
                                                                                             hiddenToMemory, useBias)

    def constructHyperModel(self, hp):

        layerN, memoryDim, order, theta, hiddenCell = self.LMUParam(hp)
        memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias, scaler = self.selectConnection(hp)
        (memoryEncoderScaler, hiddenEncoderScaler, InputEncoderScaler, biasScaler) = scaler

        self.LRMUBuilder.inputLayer(self.SequenceLength)
        self.LRMUBuilder.featureLayer(memoryDim, order, theta,
                                      self.ReservoirEncoders, hiddenCell,
                                      memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                                      memoryEncoderScaler, hiddenEncoderScaler, InputEncoderScaler, biasScaler,
                                      layerN)
        if self.ModelType == 1:
            return self.LRMUBuilder.outputPrediction().composeModel().buildPrediction()
        elif self.ModelType == 2:
            return self.LRMUBuilder.outputClassification(self.ClassNumber).composeModel().buildClassification()
        else:
            return None

    def build(self, hp):
        return self.constructHyperModel(hp)
