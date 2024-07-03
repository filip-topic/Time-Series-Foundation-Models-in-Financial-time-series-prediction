import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
import os

# Get the path of the MSc_dissertation directory
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add the modules directory to the sys.path
sys.path.append(os.path.join(base_dir, 'modules'))

# Now you can import modules from the models folder
from models import arima, lag_llama
from data import data_loader, data_splitter, data_reader

def fill_results(name, results, y_true, y_pred):
    results["r2"][name].append(r2_score(y_true, y_pred))
    results["mse"][name].append(mean_squared_error(y_true, y_pred))
    results["mae"][name].append(mean_absolute_error(y_true, y_pred))
    results["rmse"][name].append(np.sqrt(mean_squared_error(y_true, y_pred)))

# takes multi-column data
def run_experiment(data, models=["arima", "llama"], metrics=["r2", "mse", "mae", "rmse"], prediction_length=10):
    
    results = {metric: {model: [] for model in models} for metric in metrics}
    
    first_col = True
    
    for column in list(data.columns):
        if first_col:
            first_col = False
            continue
        
        column_data = data[["ds", column]]
        column_data.columns = ["ds", "y"]
        
        train, test = data_splitter.split_data(column_data, prediction_length)
        
        if "arima" in models:
            arima_model = arima.get_autoarima(train)
            arima_predictions = arima.autoarima_predictions(arima_model, prediction_length)
        
        if "llama" in models:
            lag_llama_predictions, tss = lag_llama.get_lam_llama_forecast(train, prediction_length)
            lag_llama_predictions = list(lag_llama_predictions[0].samples.mean(axis=0))
        
        y_true = list(test["y"])
        
        if "arima" in models:
            y_pred_arima = arima_predictions
            results = fill_results("arima", results, y_true, y_pred_arima)
        
        if "llama" in models:
            y_pred_llama = lag_llama_predictions
            results = fill_results("arima", results, y_true, y_pred_llama)
            
        
        print(column + " done")
        print("------------")
    
    """
    # Converting results to 2D list
    output = []
    for metric in metrics:
        row = []
        for column in list(data.columns):
            if column == "ds":
                continue
            if "arima" in models:
                row.append(results[metric]["arima"][data.columns.get_loc(column)-1])
            if "llama" in models:
                row.append(results[metric]["llama"][data.columns.get_loc(column)-1])
        output.append(row)
    """
    
    return results
