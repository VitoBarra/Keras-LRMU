def TrainAndTestModel(BuildModel, x_train, y_train, x_validation, y_validation, x_test, y_test, batch_size=128, epochs=15):
    model = BuildModel()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_validation, y_validation))
    result = model.evaluate(x_test, y_test, batch_size=128)
    return history, result
