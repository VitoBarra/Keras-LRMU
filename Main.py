import os

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
    #     print("selected GPU:{selectedGPU}")

    selectedGPU = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{selectedGPU}"

    # for tau in [17, 30]:
    #     #MackeyGlass.RunTuning(128,5000,tau,50)
    #     MackeyGlass.RunEvaluation(128, 5000, tau , epochs=25)
    #     MackeyGlass.PlotAll(128, 5000, tau)

    #psMNIST.RunTuning(10000)
    psMNIST.RunEvaluation()
    # psMNIST.PlotAll()
