import os
import shutil
import keras_tuner
import tensorflow.keras as ks
from timeit import default_timer as timer
from GlobalConfig import *
from Utility.DataUtil import DataSet
from Utility.PlotUtil import SaveTrainingData


class TimingCallback(ks.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)


def EvaluateModel(model, testName, dataSet: DataSet, batch_size=128, epochs=15, monitorStat='val_loss'):
    train, validation, test = dataSet.Unpack()
    # Define callback.
    checkpoint_filepath = f"{TEMP_DIR}/{testName}/model.keras"
    model_checkpoint_callback = ks.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor=monitorStat,
        mode='auto',
        verbose=0,
        save_best_only=True,
        initial_value_threshold=None
    )
    early_stop = ks.callbacks.EarlyStopping(
        monitor=monitorStat,
        min_delta=0.00001,
        patience=10,
        verbose=0,
        mode="auto",
        baseline=2.0,
        restore_best_weights=False,
        start_from_epoch=0,
    )

    time_tracker = TimingCallback()

    history = model.fit(train.Data, train.Label,
                        validation_data=(validation.Data, validation.Label),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[model_checkpoint_callback, early_stop, time_tracker]
                        )
    try:
        history.history["time"] = time_tracker.logs
        model = ks.models.load_model(checkpoint_filepath)
        directory = f"{BEST_MODEL_DIR}/{testName}"
        os.makedirs(directory, exist_ok=True)
        model.save(f"{directory}/best_model.keras")
        result = model.evaluate(test.Data, test.Label, batch_size=batch_size)
        return history, result
    except FileNotFoundError:
        print("no best model found")
        raise
    except Exception as e:
        print(e)
        raise


def ModelEvaluation(model, testName, saveDir, dataset, batchSize, epochs, monitorStat):
    try:
        history, result = EvaluateModel(model, testName, dataset, batchSize, epochs, monitorStat)
    except Exception as e:
        print(f"\nError during model evaluation:\n {e}\n")
        raise

    print(f"total training time: {sum(history.history['time'])}s", )
    print(f"Test loss: {result[0]}")
    print(f"Test {monitorStat[-3:]}: {result[1]}")

    try:
        SaveTrainingData(f"{saveDir}/{testName}", history, result)
    except Exception as e:
        print(f"\nexception during Data saving:\n {e}\n")
        raise


def TunerTraining(hyperModel, tuningName, problemName, dataSet, epochs=10, maxTrial=100,
                  override_test=False):
    training, validation, _ = dataSet.Unpack()
    testDir = f"{TUNING_DIR}/{problemName}/{tuningName}"
    folder_already_exists = os.path.exists(testDir)

    if folder_already_exists:
        if override_test:
            shutil.rmtree(testDir)
            print(f"old trial for {tuningName} deleted \n\n")
        else:
            raise Exception("folder already exists and the function is been called without override option")

    tuner = keras_tuner.BayesianOptimization(
        hypermodel=hyperModel,
        max_trials=maxTrial,
        project_name=f"{problemName}",
        executions_per_trial=1,
        max_retries_per_trial=3,
        max_consecutive_failed_trials=8,

        objective="val_loss",
        # Set a directory to store the intermediate results using Docker this isn't saved and that's intended
        directory=f"{problemName}/tmp",
        overwrite=True
    )

    early_stop = ks.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.01,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=1.0,
        restore_best_weights=False,
        start_from_epoch=0,
    )
    tensorboardCB = ks.callbacks.TensorBoard(f"{testDir}")
    try:
        tuner.search(
            training.Data,
            training.Label,
            validation_data=(validation.Data, validation.Label),
            epochs=epochs,
            # Use the TensorBoard callback.
            callbacks=[tensorboardCB, early_stop],
        )
    except keras_tuner.errors.FatalError as e:
        print("search failed")
        return
    except Exception as e:
        print("search failed")
        print(e)
        return

    try:
        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.summary()
        directory = f"{BEST_MODEL_DIR}/{tuningName}"
        os.makedirs(directory, exist_ok=True)
        best_model.save(f"{directory}/best_model_tuning.keras")
    except Exception as e:
        print(e)

    tuner.results_summary()
    return
