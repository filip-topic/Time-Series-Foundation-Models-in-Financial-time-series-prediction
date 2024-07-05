import pandas as pd
import numpy as np

def create_tscv_dataset(data, context_length, n_folds, prediction_horizon, max_folds = True):

    if max_folds:
        n_folds = data.shape[0] - context_length - prediction_horizon + 1
        print(f"Number of folds is {n_folds}")

    if prediction_horizon + (n_folds - 1) + context_length > data.shape[0]:
        print("Dataset is too small for the given parameters")
    
    tscv_dataframe = pd.DataFrame()
    tscv_dataframe["ds"] = data["ds"].iloc[:context_length+1]



    for i in range(n_folds):
        col = list(data["y"].iloc[i:i+context_length])
        actual = data["y"].iloc[i + context_length + prediction_horizon - 1]
        col.append(actual)
        tscv_dataframe[f"y{i}"] = col

    return tscv_dataframe