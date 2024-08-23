import os

import keras_tuner
import tensorflow.keras as ks


def EvaluateModel(buildModel,testName, train, test, batch_size=128, epochs=15, monitorStat='val_loss'):
    model = buildModel()

    checkpoint_filepath = f"./tmp/ckpt/{testName}/model.keras"
    model_checkpoint_callback = ks.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor=monitorStat,
        mode='auto',
        verbose=0
    )

    history = model.fit(train.Data, train.Label,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[model_checkpoint_callback])
    model = ks.models.load_model(checkpoint_filepath)

    result = model.evaluate(test.Data, test.Label, batch_size=batch_size)
    return history, result


def TunerTraining(hyperModel, testName, problemName, training, validation, epochs=10, maxTrial=100, force=False):
    testDir = f"./logs/{problemName}/{testName}"
    if not force:
        assert not os.path.exists(testDir)

    tuner = keras_tuner.RandomSearch(
        hypermodel=hyperModel,
        max_trials=maxTrial,
        project_name=f"{problemName}",
        executions_per_trial=1,
        max_retries_per_trial=3,
        max_consecutive_failed_trials=8,

        # Do resume the previous search in the same directory.
        overwrite=False,
        objective="val_loss",
        # Set a directory to store the intermediate results.
        directory=f"{problemName}/tmp",
    )

    try:
        tuner.search(
            training.Data,
            training.Label,
            validation_data=(validation.Data, validation.Label),
            epochs=epochs,
            # Use the TensorBoard callback.
            callbacks=[ks.callbacks.TensorBoard(f"{testDir}")],
        )
    except keras_tuner.errors.FatalError as e:
        print("serach failed")
    except Exception as e:
        print("serach failed")
        print(e)


    best_model = tuner.get_best_models(num_models=1)[0]

    best_model.save(f"./logs/{problemName}/{testName}/$best_model.h5")

    tuner.results_summary()






