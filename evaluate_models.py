from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    if predictions.ndim == 3:
        predictions = predictions[:, -1, 0]
    else:
        predictions = predictions.reshape(-1)

    y_test = y_test.reshape(-1)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    return mse, mae, predictions
