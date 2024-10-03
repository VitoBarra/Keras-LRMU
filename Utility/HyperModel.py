import keras_tuner as kt
import tensorflow.keras as ks
from Reservoir.layer import *
from tensorflow.keras.initializers import *
from Utility.ModelBuilder import ModelBuilder
from LRMU.utility import ModelType


class HyperModel(kt.HyperModel):

    def __init__(self, hyperModelName: str, problemName: str, sequenceLength: int, extraName: str = None,
                 seed: int = 0):
        super().__init__(hyperModelName)
        self.IsCategorical = None
        self.Seed = seed
        self.ProblemName = problemName
        self.ExtraName = extraName
        self.SequenceLength = sequenceLength
        self.UseLeaky = True
        self.UseInputScaler = True

        self.FeatureDimension = None
        self.Activation = None
        self.ClassNumber = None
        self.SearchTheta = None
        self.ModelType = None

        self.Builder = None
        self.ModelName = None
        self.UseESN = None
        self.UseLRMU = None
        self.UseRidge = None

        self.LMUForceParam = False
        self.LMUParam = None
        self.LMUForceConnection = False
        self.LMUConnection = None

    def SetUpPrediction(self, featureDimension, activation: str):
        self.ModelType = ModelType.Prediction
        self.FeatureDimension = featureDimension
        self.Activation = activation
        self.SearchTheta = True
        return self

    def SetUpClassification(self, classNumber: int, isCategorical: bool):
        self.ModelType = ModelType.Classification
        self.ClassNumber = classNumber
        self.IsCategorical = isCategorical
        self.SearchTheta = False
        return self

    def LMU(self):
        self.ModelName = "LMU"
        self.UseLRMU = False
        self.UseESN = False
        self.UseRidge = False
        self.CreateModelBuilder()
        return self

    def LMU_ESN(self, useLeaky=True):
        self.ModelName = "LMU-ESN"
        self.UseLRMU = False
        self.UseESN = True
        self.UseRidge = False
        self.UseLeaky = useLeaky
        self.CreateModelBuilder()
        return self

    def LRMU(self):
        self.ModelName = "LRMU"
        self.UseLRMU = True
        self.UseESN = False
        self.UseRidge = False
        self.CreateModelBuilder()
        return self

    def LRMU_ESN(self, useLeaky=True):
        self.ModelName = "LRMU-ESN"
        self.UseLRMU = True
        self.UseESN = True
        self.UseRidge = False
        self.UseLeaky = useLeaky
        self.CreateModelBuilder()
        return self

    def LRMU_ESN_Ridge(self, useLeaky=True):
        self.ModelName = "LRMU-ESN-RC"
        self.UseLRMU = True
        self.UseESN = True
        self.UseRidge = True
        self.UseLeaky = useLeaky
        self.CreateModelBuilder()
        return self

    def CreateModelBuilder(self):
        self.Builder = ModelBuilder(self.ModelName, self.ProblemName, self.ExtraName, self.Seed)

    def selectCell(self, hp, hiddenUnit):
        spectraRadius, leaky, inputScaler, biasScaler = None, None, None, None

        if not self.UseESN:
            hiddenCell = ks.layers.SimpleRNNCell(hiddenUnit, kernel_initializer=GlorotUniform(self.Seed))
        else:
            spectraRadius = hp.Float("ESN_spectraRadius", min_value=0.8, max_value=1.1, step=0.01)
            leaky = hp.Float("ESN_leaky", min_value=0.5, max_value=1, step=0.1) if self.UseLeaky else 1
            inputScaler = hp.Float("ESN_inputScaler", min_value=0.5, max_value=2, step=0.25)
            biasScaler = hp.Float("ESN_BiasScaler", min_value=0.5, max_value=2, step=0.25)
            hiddenCell = ReservoirCell(hiddenUnit, spectral_radius=spectraRadius, leaky=leaky,
                                       input_scaling=inputScaler, useBias=True, bias_scaling=biasScaler)

        return hiddenUnit, spectraRadius, leaky, inputScaler, biasScaler, hiddenCell

    def LMUSelectParam(self, hp):
        layerN = 1
        if self.ModelType == ModelType.Prediction:
            memoryDim = hp.Choice("memoryDim", values=[1, 2, 4, 8, 16, 32])
            order = hp.Choice("order", values=[1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64])
            theta = hp.Int("theta", min_value=4, max_value=64, step=4) if self.SearchTheta else self.SequenceLength
        else:
            memoryDim = hp.Int("memoryDim", min_value=128, max_value=256, step=32)
            order = hp.Int("order", min_value=128, max_value=512, step=64)
            theta = hp.Int("theta", min_value=16, max_value=258, step=16) if self.SearchTheta else self.SequenceLength
        hiddenUnit = hp.Int("hiddenUnit", min_value=16, max_value=16 * 20, step=16)
        return layerN, memoryDim, order, theta, self.selectCell(hp, hiddenUnit)

    def selectScaler(self, hp, hiddenToMemory, memoryToMemory, useBias):
        InputEncoderScaler = None
        hiddenEncoderScaler = None
        memoryEncoderScaler = None
        biasScaler = None

        if self.UseLRMU:
            InputEncoderScaler = hp.Float("InputEncoderScaler", min_value=0.5, max_value=2, step=0.25)
            if hiddenToMemory:
                hiddenEncoderScaler = hp.Float("hiddenEncoderScaler", min_value=0.5, max_value=2, step=0.25)
            if memoryToMemory:
                memoryEncoderScaler = hp.Float("memoryEncoderScaler", min_value=0.5, max_value=2, step=0.25)
            if useBias:
                biasScaler = hp.Float("biasScaler", min_value=0.5, max_value=2, step=0.25)

        return hiddenEncoderScaler, memoryEncoderScaler, InputEncoderScaler, biasScaler

    def selectConnection(self, hp):
        memoryToMemory = hp.Boolean("memoryToMemory")
        hiddenToMemory = hp.Boolean("hiddenToMemory")
        inputToHiddenCell = hp.Boolean("inputToHiddenCell")
        useBias = hp.Boolean("useBias")
        return hiddenToMemory, memoryToMemory, inputToHiddenCell, useBias, self.selectScaler(hp, hiddenToMemory,
                                                                                             memoryToMemory, useBias)

    def ForceLMUParam(self, Nlayer: int, memoryDim: int, order: int, theta: int, hiddenUnit: int):
        self.LMUForceParam = True
        self.LMUParam = (Nlayer, memoryDim, order, theta, hiddenUnit)
        return self

    def ForceConnection(self, hiddenToMemory: bool, memoryToMemory: bool, inputToHiddenCell: bool, useBias: bool):
        self.LMUForceConnection = True
        self.LMUConnection = (hiddenToMemory, memoryToMemory, inputToHiddenCell, useBias)
        return self

    def constructHyperModel(self, hp):

        if self.LMUForceParam:
            layerN, memoryDim, order, theta, hiddenUnit = self.LMUParam
            (hiddenUnit, spectraRadius, leaky, inputScaler, biasScaler, hiddenCell) = self.selectCell(hp, hiddenUnit)
        else:
            layerN, memoryDim, order, theta, (
                hiddenUnit, spectraRadius, leaky, inputScaler, biasScaler, hiddenCell) = self.LMUSelectParam(hp)

        if self.LMUForceConnection:
            hiddenToMemory, memoryToMemory, inputToHiddenCell, useBias = self.LMUConnection
            encoderScaler = self.selectScaler(hp, hiddenToMemory, memoryToMemory, useBias)
        else:
            hiddenToMemory, memoryToMemory, inputToHiddenCell, useBias, encoderScaler = self.selectConnection(hp)

        (hiddenEncoderScaler, memoryEncoderScaler, InputEncoderScaler, biasEncoderScaler) = encoderScaler

        if self.UseRidge:
            regularization = hp.Float("regularization", min_value=0.5, max_value=1.5, step=0.25)
            model = self.Builder.LRMU_ESN_Ridge(self.ModelType, self.SequenceLength, memoryDim, order, theta,
                                                hiddenToMemory, memoryToMemory, inputToHiddenCell, useBias,
                                                hiddenEncoderScaler, memoryEncoderScaler, InputEncoderScaler,
                                                biasEncoderScaler, hiddenUnit, spectraRadius, leaky, inputScaler,
                                                biasScaler, regularization)
            if self.ModelType == ModelType.Prediction:
                model.compile(optimizer='adam', loss="mse", metrics=["mae"])
                return model.custom_compile([keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()], )
            elif self.ModelType == ModelType.Classification:
                if self.IsCategorical:
                    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["categorical_accuracy"])
                    model.custom_compile([keras.losses.CategoricalCrossentropy(), keras.metrics.CategoricalAccuracy()])
                else:
                    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                    model.custom_compile([keras.losses.SparseCategoricalCrossentropy(), keras.metrics.Accuracy()])
                return model


        else:
            self.Builder.inputLayer(self.SequenceLength)
            if self.UseLRMU:
                self.Builder.LRMU(memoryDim, order, theta, hiddenCell,
                                  hiddenToMemory, memoryToMemory, inputToHiddenCell, useBias,
                                  hiddenEncoderScaler, memoryEncoderScaler, InputEncoderScaler, biasEncoderScaler,
                                  layerN)
            else:
                self.Builder.LMU(memoryDim, order, theta, hiddenCell, False,
                                 hiddenToMemory, memoryToMemory, inputToHiddenCell, useBias,
                                 layerN)

            if self.ModelType == ModelType.Prediction:
                return self.Builder.BuildPrediction(self.FeatureDimension, self.Activation)
            elif self.ModelType == ModelType.Classification:
                return self.Builder.BuildClassification(self.ClassNumber, self.IsCategorical)

    def build(self, hp):
        return self.constructHyperModel(hp)
