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

    selectedGPU = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{selectedGPU}"



    for tau in [30]:
        MackeyGlass.models.RunEvaluation(128, tau, "linear", 64, 500)
        #  MackeyGlass.models.RunTuning(128, tau, "linear", 25, 50)


    # MackeyGlass.models.PrintSummary(17, "linear")

    psMNIST.models.RunEvaluation(epochs=25,isCategorical=False)
    # psMNIST.models.RunTuning(5000, 5,50)
    # psMNIST.models.PrintSummary()
