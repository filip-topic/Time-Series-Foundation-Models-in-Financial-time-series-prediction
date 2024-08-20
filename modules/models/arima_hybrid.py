import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import os

from modules.models import arima

from modules.models import timegpt


def get_arima_timegpt_predictions(data, prediction_horizon, frequency, error_train_size):
    
    tscv = TimeSeriesSplit(n_splits=error_train_size, test_size=prediction_horizon)

    series = data["y"]

    predictions = []
    actual = []
    timestamps = []

    i = 0
    for train_index, test_index in tscv.split(series):

        train = data.iloc[train_index]
        valid = list(data.iloc[test_index]["y"])
        timestamp = list(data.iloc[test_index]["ds"])

        arima_model = arima.get_autoarima(train)
        arima_prediction = arima.autoarima_predictions(arima_model, prediction_horizon)

        predictions.append(arima_prediction[0])
        actual.append(valid[0])
        timestamps.append(timestamp[0])

        i +=1
        print(f"error {i} calculated")

    errors = np.array(actual) - np.array(predictions)

    d = pd.DataFrame()
    x = pd.DataFrame()
    d["ds"] = timestamps
    d["y"] = errors

    x["ds"] = timestamps
    x["x1"] = predictions
    #x["x2"] = actual

    error_forecast = timegpt.get_timegpt_forecast(data=d, prediction_length=prediction_horizon, frequency=frequency, ft_steps=50, x=x)
    
    model = arima.get_autoarima(data)
    forecast = arima.autoarima_predictions(model, prediction_horizon)

    return forecast+error_forecast

