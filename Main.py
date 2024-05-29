import os
import Model.pmMNIST.pmMNIST as pmMNIST
import Model.ECG5000.ECG5000 as ECG5000
import Model.EthanolLevel.EthanolLevel as EthanolLevel
import Model.MackeyGlass.MackeyGlass as MackeyGlass
import Utility.GPUSelection as GPUSelection
from Utility.Debug import PrintAvailableGPU

if __name__ == "__main__":
    PrintAvailableGPU()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    MackeyGlass.Run(False)