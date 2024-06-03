import os
import sys

import Problems.pmMNIST.pmMNIST as pmMNIST
import Problems.ECG5000.ECG5000 as ECG5000
import Problems.MackeyGlass.MackeyGlass as MackeyGlass
import Utility.GPUSelection as GPUSelection
from Utility.Debug import PrintAvailableGPU

if __name__ == "__main__":
    PrintAvailableGPU()
    selectedGPU = "0"

    if len(sys.argv)>2 and bool(sys.argv[1]):
        selectedGPU = GPUSelection.pick_gpu_lowest_memory()

    print(selectedGPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{selectedGPU}"

    MackeyGlass.Run(False)
    pmMNIST.Run(False)
    ECG5000.Run(False)
