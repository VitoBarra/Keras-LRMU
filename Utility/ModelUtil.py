import tensorflow.keras as ks


def TrainAndTestModel_OBJ(buildModel, train, validation, test, batch_size=128, epochs=15, monitorStat='val_accuracy'):
    return TrainAndTestModel(buildModel, train.Data, train.Label, validation.Data, validation.Label, test.Data,
                             test.Label,
                             batch_size, epochs, monitorStat)


def TrainAndTestModel(buildModel, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=128, epochs=15, monitorStat='val_accuracy'):

    model = buildModel()

    checkpoint_filepath = '/tmp/ckpt/checkpoint.model.keras'
    model_checkpoint_callback = ks.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor=monitorStat,
    mode='auto',
    save_best_only=True)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        callbacks=[model_checkpoint_callback])
    model = ks.models.load_model(checkpoint_filepath)

    result = model.evaluate(x_test, y_test, batch_size=batch_size)
    return history, result
