import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
import os

# Get the path of the MSc_dissertation directory
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add the modules directory to the sys.path
sys.path.append(base_dir)

# Now you can import modules from the models folder
from modules.models import arima, lag_llama
from modules.data import data_loader, data_splitter, data_reader

def fill_results(name, results, y_true, y_pred):
    if len(y_true) == 1:
        results[name].append(y_pred)
        if len(results["true"]) == 0 or results["true"][-1] != y_true: # this is to make sure we dont double append the "true" list and also not to subindex an empty list
            results["true"].append(y_true)
    else:
        results["r2"][name].append(r2_score(y_true, y_pred))
        results["mse"][name].append(mean_squared_error(y_true, y_pred))
        results["mae"][name].append(mean_absolute_error(y_true, y_pred))
        results["rmse"][name].append(np.sqrt(mean_squared_error(y_true, y_pred)))
    return results

# takes multi-column data
def run_experiment(data, models=["arima", "llama"], metrics=["r2", "mse", "mae", "rmse"], prediction_length=1, tscv = False, context_length = 64):

    if tscv:
        prediction_length = 1

    if prediction_length == 1:
        results = {model: [] for model in models}
        results["true"] = []
    else:
        results = {metric: {model: [] for model in models} for metric in metrics}
    
    first_col = True
    
    i = 0

    for column in list(data.columns):
        if first_col:
            first_col = False
            continue
        
        # this is here just to speed up testing
        #if i == 5:
            #break

        i = i+1
        column_data = data[["ds", column]]
        column_data.columns = ["ds", "y"]
        

        train, test = data_splitter.split_data(column_data, prediction_length)
        
        y_true = list(test["y"])

        if "arima" in models:
            arima_model = arima.get_autoarima(train)
            arima_predictions = arima.autoarima_predictions(arima_model, prediction_length)
            y_pred_arima = arima_predictions
            if tscv:
                y_pred_arima = y_pred_arima[-1]
            results = fill_results("arima", results, y_true, y_pred_arima)

        
        if "llama" in models:
            lag_llama_predictions, tss = lag_llama.get_lam_llama_forecast(train, prediction_length, context_length=context_length)
            lag_llama_predictions = list(lag_llama_predictions[0].samples.mean(axis=0))
            y_pred_llama = lag_llama_predictions
            if tscv:
                y_pred_llama = y_pred_llama[-1]
            results = fill_results("llama", results, y_true, y_pred_llama)
            
        
        print(column + f" done  {i}")
        print("------------")
    
    

    if tscv:
        prediction = results
        results = {metric: {model: [] for model in models} for metric in metrics}
        for model in prediction.keys():
            if (model != "true"):
                results["r2"][model] = r2_score(prediction["true"], prediction[model])
                results["mse"][model] = mean_squared_error(prediction["true"], prediction[model])
                results["mae"][model] = mean_absolute_error(prediction["true"], prediction[model])
                results["rmse"][model] = np.sqrt(mean_squared_error(prediction["true"], prediction[model]))
        return results, prediction


    return results
