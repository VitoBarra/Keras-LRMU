import keras_tuner as kt
import tensorflow.keras as ks
from ESN.layer import *
from tensorflow.keras.initializers import *
from Utility.ModelBuilder import ModelBuilder


class HyperModel(kt.HyperModel):

    def __init__(self, hyperModelName, problemName, sequenceLength, classNuber=None, searchTheta=True, useLeaky=True,
                 seed=0):
        super().__init__(hyperModelName)
        self.UseLeaky = useLeaky
        self.UseInputScaler = False
        self.Seed = seed
        self.ProblemName = problemName
        self.SequenceLength = sequenceLength
        self.SearchTheta = searchTheta

        self.ClassNumber = classNuber

        if classNuber is None:
            self.ModelType = 1
        else:
            self.ModelType = 2

        self.Builder = None
        self.ModelName = None
        self.UseESN = None
        self.UseLRMU = None

    def LMU(self):
        self.ModelName = "LMU"
        self.UseLRMU = False
        self.UseESN = False
        self.CreateModelBuilder()
        return self

    def LMU_ESN(self, useLeaky=True, useInputScaler=False):
        self.ModelName = "LMU-ESN"
        self.UseLRMU = False
        self.UseESN = True
        self.UseLeaky = useLeaky
        self.UseInputScaler = useInputScaler
        self.CreateModelBuilder()
        return self

    def LRMU(self):
        self.ModelName = "LRMU"
        self.UseLRMU = True
        self.UseESN = False
        self.CreateModelBuilder()
        return self

    def LRMU_ESN(self, useLeaky=True, useInputScaler=False):
        self.ModelName = "LRMU-ESN"
        self.UseLRMU = True
        self.UseESN = True
        self.UseLeaky = useLeaky
        self.UseInputScaler = useInputScaler
        self.CreateModelBuilder()
        return self

    def CreateModelBuilder(self):
        self.Builder = ModelBuilder(self.ProblemName, self.ModelName, self.Seed)

    def selectCell(self, hp, hiddenUnit):
        if not self.UseESN:
            hiddenCell = ks.layers.SimpleRNNCell(hiddenUnit, kernel_initializer=GlorotUniform(self.Seed))
        else:
            spectraRadius = hp.Float("ESN_spectraRadius", min_value=0.8, max_value=1.1, step=0.025)
            leaky = hp.Float("ESN_leaky", min_value=0.5, max_value=1, step=0.1) if self.UseLeaky else 1
            inputScaler = hp.Float("ESN_inputScaler", min_value=0.5, max_value=2,
                                   step=0.25) if self.UseInputScaler else 1
            hiddenCell = ReservoirCell(hiddenUnit, spectral_radius=spectraRadius, leaky=leaky,
                                       input_scaling=inputScaler)
        return hiddenCell

    def LMUParam(self, hp):
        layerN = 1
        if self.ModelType == 1:
            memoryDim = hp.Choice("memoryDim", values=[1, 2, 4, 8, 16, 32])
            order = hp.Choice("order", values=[1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64])
            theta = hp.Int("theta", min_value=4, max_value=64, step=4) if self.SearchTheta else self.SequenceLength
        else:
            memoryDim = hp.Int("memoryDim", min_value=128, max_value=256, step=32)
            order = hp.Int("order", min_value=128, max_value=512, step=64)
            theta = hp.Int("theta", min_value=16, max_value=258, step=16) if self.SearchTheta else self.SequenceLength
        hiddenUnit = hp.Int("hiddenUnit", min_value=16, max_value=16 * 20, step=16)
        return layerN, memoryDim, order, theta, self.selectCell(hp, hiddenUnit)

    def selectScaler(self, hp, memoryToMemory, hiddenToMemory, useBias):
        InputEncoderScaler = None
        memoryEncoderScaler = None
        hiddenEncoderScaler = None
        biasScaler = None

        if self.UseLRMU:
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
        return memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias, self.selectScaler(hp, memoryToMemory,
                                                                                             hiddenToMemory, useBias)

    def constructHyperModel(self, hp):

        layerN, memoryDim, order, theta, hiddenCell = self.LMUParam(hp)
        memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias, scaler = self.selectConnection(hp)
        (memoryEncoderScaler, hiddenEncoderScaler, InputEncoderScaler, biasScaler) = scaler

        self.Builder.inputLayer(self.SequenceLength)
        if self.UseLRMU:
            self.Builder.LRMU(memoryDim, order, theta, hiddenCell,
                              memoryToMemory, hiddenToMemory, inputToHiddenCell, useBias,
                              memoryEncoderScaler, hiddenEncoderScaler, InputEncoderScaler, biasScaler,
                              layerN)
        else:
            self.Builder.LMU(memoryDim, order, theta, hiddenCell, False,
                             memoryToMemory, inputToHiddenCell, hiddenToMemory, useBias,
                             layerN)

        if self.ModelType == 1:
            return self.Builder.BuildPrediction(1)
        elif self.ModelType == 2:
            return self.Builder.BuildClassification(self.ClassNumber)
        else:
            print(f"\nModelType:{self.ModelType}\n")
            return None

    def build(self, hp):
        return self.constructHyperModel(hp)
