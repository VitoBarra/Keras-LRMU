from Utility.DataUtil import SplitDataset
from Utility.ModelUtil import *
from Utility.PlotUtil import *
import Problems.MackeyGlass.DataGeneration as dg
from ESN.layer import *
from Utility.LRMUHyperModel import LRMUHyperModel
from Utility.LRMUModelBuilder import LRMUModelBuilder

PROBLEM_NAME = "Mackey-Glass"
SEQUENCE_LENGTH = 5000

def ModelLRMU_SelectedHP():
    LRMUBuilder = LRMUModelBuilder(PROBLEM_NAME,"manual test")
    LRMUBuilder.inputLayer(SEQUENCE_LENGTH)
    LRMUBuilder.featureLayer(10,256, 212,
                             ReservoirCell(10),
                             True,True, True, False, False,
                             False,
                             1,1,1,1)
    return LRMUBuilder.outputPrediction().composeModel().buildPrediction()


def SingleTraining(training, validation, test):
    history, result = TrainAndTestModel_OBJ(ModelLRMU_SelectedHP, training, validation, test, 64, 15, "val_loss")

    print("test loss:", result[0])
    print("test mse:", result[1])
    PlotModelLoss(history, "Problems LRMU", f"./plots/{PROBLEM_NAME}", f"{PROBLEM_NAME}_LRMU_ESN")



def Run(singleTraining=True):
    data, label = dg.generate_data(128, SEQUENCE_LENGTH,0,15, 30)
    training, validation, test = SplitDataset(data, label, 0.1, 0.1)

    hyperModels =LRMUHyperModel("MackeyGlass-hyperModel",PROBLEM_NAME,SEQUENCE_LENGTH)

    if singleTraining:
        SingleTraining(training, validation, test)
    else:
        #TunerTraining(hyperModels.LMU_ESN(), "LMU_ESN_Tuning_5k_T30", PROBLEM_NAME, training, validation, 5, 150, False)
        TunerTraining(hyperModels.LMU_RE(), "LMU_RE_Tuning_5k_T30", PROBLEM_NAME, training, validation, 5, 150, True)
        TunerTraining(hyperModels.LRMU(), "LRMU_Tuning_5k_T30", PROBLEM_NAME, training, validation, 5, 150, True)
