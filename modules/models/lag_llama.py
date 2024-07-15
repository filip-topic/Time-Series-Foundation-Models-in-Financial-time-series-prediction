import os
import sys

old_wd = os.getcwd()

sys.path.append(old_wd + "\\modules\\models\\llama")

from itertools import islice

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset

from gluonts.dataset.pandas import PandasDataset
import pandas as pd



from lag_llama.gluon.estimator import LagLlamaEstimator  



def get_lag_llama_predictions(dataset, prediction_length, device, context_length=32, use_rope_scaling=False, num_samples=100):
    ckpt = torch.load("modules/models/llama/lag-llama.ckpt", map_location=device) # Uses GPU since in this Colab we use a GPU.    # also modified the path FILIP
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
    }

    estimator = LagLlamaEstimator(
        ckpt_path="modules/models/llama/lag-llama.ckpt",   # modified by FILIPS
        prediction_length=prediction_length,
        context_length=context_length, # Lag-Llama was trained with a context length of 32, but can work with any context length

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_arguments if use_rope_scaling else None,

        batch_size=1,
        num_parallel_samples=100,
        device=device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss

def prepare_data(data, prediction_length, frequency):
    x = data.copy()
    x.set_index('ds', inplace=True)
    x.index.name = None
    x.index = pd.to_datetime(x.index, errors="coerce")
    
    freq_map = {
        'minutely': 'T',  # minute frequency
        'hourly': 'H',    # hourly frequency
        'daily': 'D',     # daily frequency
        'weekly': 'W',    # weekly frequency
        'monthly': 'M',   # monthly frequency
        'quarterly': 'Q'  # quarterly frequency
    }

    #return x

    # filling in the gaps in the index
    if frequency == "daily":
        full_date_range = pd.date_range(start=x.index.min(), end=x.index.max(), freq='D')
        x = x.reindex(full_date_range)
    
    #return x

    # adding PREDICTION_LENGTH dummy rows
    last_date = x.index[-1]
    
    new_dates = pd.date_range(start = last_date+pd.Timedelta(days=1), periods = prediction_length, freq = freq_map[frequency]) ## im not sure this is 100% correctS
    new_rows = pd.DataFrame({'y': [1]*prediction_length}, index=new_dates)
    x = pd.concat([x, new_rows])
    x = PandasDataset(dict(x))
    return x

def get_lam_llama_forecast(data, prediction_length, device = torch.device("cpu"), context_length = 32, use_rope_scaling=False, num_samples=100, frequency = "daily"):
    data = prepare_data(data, prediction_length, frequency=frequency)
    forecasts, tss = get_lag_llama_predictions(data, prediction_length, device, context_length, use_rope_scaling, num_samples)
    return forecasts, tss

if __name__ == "__main__":
    import yfinance as yf
    from datetime import datetime, timedelta
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    apple_stock_price = yf.download("AAPL", start=start_date, end=end_date, interval="1d")["Adj Close"]
    apple_stock_price = apple_stock_price.reset_index()
    apple_stock_price.columns = ['ds', 'y']
