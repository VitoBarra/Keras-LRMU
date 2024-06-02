import os
import sys

import Model.pmMNIST.pmMNIST as pmMNIST
import Model.ECG5000.ECG5000 as ECG5000
import Model.EthanolLevel.EthanolLevel as EthanolLevel
import Model.MackeyGlass.MackeyGlass as MackeyGlass
import Utility.GPUSelection as GPUSelection
from Utility.Debug import PrintAvailableGPU

if __name__ == "__main__":
    PrintAvailableGPU()
    selectedGPU = "0"

    if len(sys.argv)>2 and bool(sys.argv[1]):
        selectedGPU = GPUSelection.pick_gpu_lowest_memory()

    print(selectedGPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{selectedGPU}"

    pmMNIST.Run(False)
