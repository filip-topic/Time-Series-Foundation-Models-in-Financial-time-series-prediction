
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import pmdarima as pm

# automated dickey fuller test
'''def adf(x):
    res = adfuller(x)
    print("Test-Statistic:", res[0])
    print("P-Value:", res[1])
    if res[1] < 0.05:
      s = "stationary"
    else:
      s = "non_stationary"
    print(s)
    return (s)

def find_d(data):
   df_diff = data
   for i in range(5):
    s = adf(df_diff['y'])
    if s == "stationary":
        return i, df_diff
    df_diff = df_diff.diff().dropna()


# IMPORTANT I'm not sure whether this should be performed on df or df_diff
def find_optimal_order(df_diff, confidence_level=0.2, nlags = 50):
    acf_vals = acf(df_diff["y"], nlags=nlags)
    pacf_vals = pacf(df_diff["y"], nlags=nlags)


    # Find the optimal q value from ACF
    for i in range(1, len(acf_vals)):
        confidence_threshold = acf_vals[i]*(1-confidence_level)
        if abs(acf_vals[i]) < confidence_threshold:
            q = i - 1
            break
        else:
            q = len(acf_vals) - 1

    # Find the optimal p value from PACF
    for i in range(1, len(pacf_vals)):
        confidence_threshold = pacf_vals[i]*(1-confidence_level)
        if abs(pacf_vals[i]) < confidence_threshold:
            p = i - 1
            break
        else:
            p = len(pacf_vals) - 1

    return p, q

def get_trained_arima_model(data):
   d, df_diff = find_d(data)
   # IMPORTANT: Im not sure whether we should use df or df_diff for FInd_optimal_order()
   p, q = find_optimal_order(df_diff)
   model = ARIMA(data, order=(p, d, q))
   return model.fit()

def arima_forecast(model, prediction_length):
   predictions = model.forecast(steps = prediction_length)
   return pd.Series(predictions, index = test.index)'''

# Uses AIC to chose p, q and KPSS unit root test for d
def get_autoarima(data):
   return pm.auto_arima(data["y"], suppress_warnings=True, stepwise = False, seasonal = False)

def autoarima_predictions(model, prediction_length):
   return list(model.predict(n_periods = prediction_length))