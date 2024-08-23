import os
import sys

import Problems.psMNIST.psMNIST as psMNIST
import Problems.MackeyGlass.MackeyGlass as MackeyGlass
import Utility.GPUSelection as GPUSelection
from Utility.Debug import PrintAvailableGPU

if __name__ == "__main__":
    PrintAvailableGPU()

    # not working
    # selectedGPU = "0"
    # if len(sys.argv)>2 and bool(sys.argv[1]):
    #     selectedGPU = GPUSelection.pick_gpu_lowest_memory()


    selectedGPU= "3"

    print(selectedGPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{selectedGPU}"

    MackeyGlass.RunEvaluation(128,5000,17)
    MackeyGlass.RunEvaluation(128,5000,17)
    #MackeyGlass.RunTuning(128,5000,30)
    #MackeyGlass.RunTuning(128,5000,30)
    psMNIST.RunEvaluation()
    #psMNIST.RunTuning(10000)
