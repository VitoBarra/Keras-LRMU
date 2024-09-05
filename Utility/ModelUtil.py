import os
import shutil
import keras_tuner
import tensorflow.keras as ks
from timeit import default_timer as timer
from GlobalConfig import *

class TimingCallback(ks.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

def EvaluateModel(buildModel, testName, train, test, batch_size=128, epochs=15, monitorStat='loss'):


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
        monitor="loss",
        min_delta=0,
        patience=2,
        verbose=0,
        mode="auto",
        baseline=1.0,
        restore_best_weights=False,
        start_from_epoch=0,
    )

    time_tracker= TimingCallback()

    # Build model.
    model = buildModel()
    history = model.fit(train.Data, train.Label,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[model_checkpoint_callback, early_stop,time_tracker]
                        )
    try:
        history.history["time"]=time_tracker.logs
        model = ks.models.load_model(checkpoint_filepath)
        model.save(f"{BEST_MODEL_DIR}/{testName}/best_model.h5")
        result = model.evaluate(test.Data, test.Label, batch_size=batch_size)
        return history, result
    except FileNotFoundError:
        print("no best model found")
        raise
    except Exception as e:
        print(e)
        raise


def TunerTraining(hyperModel, tuningName, problemName, training, validation, epochs=10, maxTrial=100,
                  override_test=False):
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
    tensorboardCB=ks.callbacks.TensorBoard(f"{testDir}")
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
        print("serach failed")
        return
    except Exception as e:
        print("serach failed")
        print(e)
        return

    try:
        best_model = tuner.get_best_models()
        best_model.save(f"{BEST_MODEL_DIR}/{tuningName}/best_model_tuning.h5")
        best_model.summary()
    except Exception as e:
        print(e)

    tuner.results_summary()
    return
