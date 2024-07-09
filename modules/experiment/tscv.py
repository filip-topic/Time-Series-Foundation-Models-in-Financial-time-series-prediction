from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os

# Get the path of the MSc_dissertation directory
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add the modules directory to the sys.path
sys.path.append(base_dir)




from modules.models import arima, lag_llama, autoregressor


def mean_directional_accuracy(actual, predicted, last_train_point):
    
    a = actual.copy()
    p = predicted.copy()

    a.append(last_train_point)
    p.append(last_train_point)

    a = pd.Series(a)
    p = pd.Series(p)

    actual_diff = a.diff().dropna()
    predicted_diff = p.diff().dropna()
    
    correct_directions = (actual_diff * predicted_diff > 0).sum()
    total_directions = len(actual_diff)
    
    mda_value = correct_directions / total_directions
    
    return mda_value


def get_summary(results):
    summary = pd.DataFrame({
    'r2': [results['r2'].mean(), results['r2'].median(), results['r2'].std()],
    'mse': [results['mse'].mean(), results['mse'].median(), results['mse'].std()],
    'mae': [results['mae'].mean(), results['mae'].median(), results['mae'].std()],
    'rmse': [results['rmse'].mean(), results['rmse'].median(), results['rmse'].std()],
    'mda': [results['mda'].mean(), results['mda'].median(), results['mda'].std()]
    }, index=['mean', 'median', 'std'])
    return summary

def fill_metrics(valid, predictions, last_train):
    metrics = [r2_score(valid, predictions), 
               mean_squared_error(valid, predictions), 
               mean_absolute_error(valid, predictions),
               np.sqrt(mean_squared_error(valid, predictions)),
               mean_directional_accuracy(valid, predictions, last_train)]
    return metrics


def get_tscv_results(data, prediction_horizon, context_length, folds):

    tscv = TimeSeriesSplit(n_splits=folds, test_size=prediction_horizon)
    
    prediction_cols = [f"t_{i}" for i in range(1, prediction_horizon + 1)]

    # initializing empty lists of outputs
    results = []
    predictions = []
    actual = pd.DataFrame(columns=prediction_cols)

    metrics=["r2", "mse", "mae", "rmse", "mda"] 

    arima_results = pd.DataFrame(columns=metrics)
    llama_results = pd.DataFrame(columns= metrics)
    autoregressor_results = pd.DataFrame(columns=metrics)

    
    arima_preds = pd.DataFrame(columns=prediction_cols)
    llama_preds = pd.DataFrame(columns=prediction_cols)
    autoregressor_preds = pd.DataFrame(columns=prediction_cols)

    

    series = data["y"]

    i = 0

    for train_index, test_index in tscv.split(series):
    # subsetting the original data according to train/test split
        train = data.iloc[train_index]
        valid = list(data.iloc[test_index]["y"])


        # inputting data into the models
        arima_model = arima.get_autoarima(train)
        autoarima_predictions = arima.autoarima_predictions(arima_model, prediction_horizon)
        lag_llama_predictions, tss = lag_llama.get_lam_llama_forecast(train, prediction_horizon, context_length=context_length)
        lag_llama_predictions = list(lag_llama_predictions[0].samples.mean(axis = 0))
        autoregressor_predictions = autoregressor.get_autoregressor_prediction(train, prediction_horizon)

        last_train = train["y"].iloc[-1]

        # calculating metrics for this fold
        arima_metrics = fill_metrics(valid, autoarima_predictions, last_train)
        llama_metrics = fill_metrics(valid, lag_llama_predictions, last_train)
        autoregressor_metrics = fill_metrics(valid, autoregressor_predictions, last_train)

        # concating the metrics for current fold to the results
        arima_results = pd.concat([arima_results, pd.DataFrame([arima_metrics], columns=metrics)], ignore_index=True)
        llama_results = pd.concat([llama_results, pd.DataFrame([llama_metrics], columns=metrics)], ignore_index=True)
        autoregressor_results = pd.concat([autoregressor_results, pd.DataFrame([autoregressor_metrics], columns=metrics)], ignore_index=True)

        # concatinating the predictions
        arima_preds = pd.concat([arima_preds, pd.DataFrame([autoarima_predictions], columns = prediction_cols)], ignore_index=True)
        llama_preds = pd.concat([llama_preds, pd.DataFrame([lag_llama_predictions], columns = prediction_cols)], ignore_index=True)
        autoregressor_preds = pd.concat([autoregressor_preds, pd.DataFrame([autoregressor_predictions], columns = prediction_cols)], ignore_index=True)

        # concating the actual
        actual = pd.concat([actual, pd.DataFrame([valid], columns = prediction_cols)], ignore_index=True)

        i += 1

    results = [arima_results, llama_results, autoregressor_results]
    predictions = [arima_preds, llama_preds, autoregressor_preds]

    
    return results, predictions, actual

