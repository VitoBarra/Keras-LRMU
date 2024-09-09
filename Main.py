import os

import Problems.psMNIST as psMNIST
import Problems.MackeyGlass as MackeyGlass
import Utility.GPUSelection as GPUSelection
from Utility.Debug import PrintAvailableGPU

if __name__ == "__main__":
    PrintAvailableGPU()

    # not working
    # selectedGPU = "0"
    # if len(sys.argv)>2 and bool(sys.argv[1]):
    #     selectedGPU = GPUSelection.pick_gpu_lowest_memory()
    #     print("selected GPU:{selectedGPU}")

    selectedGPU = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{selectedGPU}"

    # for activation in ["linear", "relu"]:
    #     for tau in [17, 30]:
    #         MackeyGlass.models.RunEvaluation(128, tau, activation, epochs=500)
    #         MackeyGlass.models.RunTuning(128, 5000, tau, 5,50)

    #psMNIST.models.RunTuning(10000)
    psMNIST.models.RunEvaluation(epochs=25)
