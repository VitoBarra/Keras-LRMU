def TrainAndTestModel_OBJ(buildModel, train, validation, test, batch_size=128, epochs=15):
    return TrainAndTestModel(buildModel, train.Data, train.Label, validation.Data, validation.Label, test.Data,
                             test.Label,
                             batch_size, epochs)


def TrainAndTestModel(buildModel, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=128, epochs=15):

    model = buildModel()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_val, y_val))
    result = model.evaluate(x_test, y_test, batch_size=batch_size)
    return history, result
