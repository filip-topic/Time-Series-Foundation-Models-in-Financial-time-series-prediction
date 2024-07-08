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




from modules.models import arima, lag_llama


def mean_directional_accuracy(actual, predicted, last_train_point):
    
    a = actual.copy()
    p = predicted.copy()

    a.append(last_train_point)
    p.append(last_train_point)

    a = pd.Series(a)
    p = pd.Series(p)

    actual_diff = a.diff().dropna()
    predicted_diff = p.diff().dropna()
    
    # Align the series after dropping NaN values due to differencing
    #actual_diff = actual_diff 
    #predicted_diff = predicted_diff 
    
    correct_directions = (actual_diff * predicted_diff > 0).sum()
    total_directions = len(actual_diff)
    
    mda_value = correct_directions / total_directions
    
    return mda_value


def get_tscv_results(data, prediction_horizon, context_length, folds):

    tscv = TimeSeriesSplit(n_splits=folds, test_size=prediction_horizon)

    models=["arima", "llama"]
    metrics=["r2", "mse", "mae", "rmse", "mda"]
    results = {metric: {model: {f"fold_{i}": [] for i in range(folds)} for model in models} for metric in metrics}

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

        # for my own testing purposes
        """
        print(autoarima_predictions)
        print(lag_llama_predictions)
        print(valid)
        """

        # recording the metrics
        results["r2"]["arima"][f"fold_{i}"].append(r2_score(valid, autoarima_predictions))
        results["mse"]["arima"][f"fold_{i}"].append(mean_squared_error(valid, autoarima_predictions))
        results["mae"]["arima"][f"fold_{i}"].append(mean_absolute_error(valid, autoarima_predictions))
        results["rmse"]["arima"][f"fold_{i}"].append(np.sqrt(mean_squared_error(valid, autoarima_predictions)))
        results["mda"]["arima"][f"fold_{i}"].append(mean_directional_accuracy(valid, autoarima_predictions))

        results["r2"]["llama"][f"fold_{i}"].append(r2_score(valid, lag_llama_predictions))
        results["mse"]["llama"][f"fold_{i}"].append(mean_squared_error(valid, lag_llama_predictions))
        results["mae"]["llama"][f"fold_{i}"].append(mean_absolute_error(valid, lag_llama_predictions))
        results["rmse"]["llama"][f"fold_{i}"].append(np.sqrt(mean_squared_error(valid, lag_llama_predictions)))
        results["mda"]["llama"][f"fold_{i}"].append(mean_directional_accuracy(valid, lag_llama_predictions))

        i += 1
    
    return results



if __name__ == "__main__":
    
    actual = pd.Series([100, 102, 101, 103, 105])
    predicted = pd.Series([100, 101, 102, 104, 106])

    print(f"Mean Directional Accuracy: {mean_directional_accuracy(actual, predicted)}")
