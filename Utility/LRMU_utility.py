from LRMU import LRMU

def GenerateLRMUFeatureLayer(inputs, memoryDim, order, theta, hiddenUnit, spectraRadius, leaky, reservoirMode, hiddenCell,
                             memoryToMemory, hiddenToMemory, inputToCell, useBias, seed, layerN=1):
    feature = LRMU(memoryDimension=memoryDim, order=order, theta=theta,
                        hiddenUnit=hiddenUnit, spectraRadius=spectraRadius, leaky=leaky,
                        reservoirMode=reservoirMode, hiddenCell=hiddenCell,
                        memoryToMemory=memoryToMemory, hiddenToMemory=hiddenToMemory, inputToHiddenCell=inputToCell,
                        useBias=useBias,
                        seed=seed, returnSequences=layerN > 1)(inputs)
    for i in range(layerN - 1):
        feature = LRMU(memoryDimension=memoryDim, order=order, theta=theta,
                            hiddenUnit=hiddenUnit, spectraRadius=spectraRadius, leaky=leaky,
                            reservoirMode=reservoirMode, hiddenCell=hiddenCell,
                            memoryToMemory=memoryToMemory, hiddenToMemory=hiddenToMemory, inputToHiddenCell=inputToCell,
                            useBias=useBias,
                            seed=seed, returnSequences=i != layerN - 2)(feature)
    return feature
