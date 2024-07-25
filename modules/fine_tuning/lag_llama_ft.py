from itertools import islice

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from tqdm.autonotebook import tqdm

import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset

from gluonts.dataset.pandas import PandasDataset
import pandas as pd

from lag_llama.gluon.estimator import LagLlamaEstimator

def get_predictor(prediction_length, context_length, batch_size = 10, max_epochs = 30):
    ckpt_path = "modules/models/llama/ft-lag-llama.ckpt"

    import torch

    from lag_llama.gluon.estimator import LagLlamaEstimator

    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=prediction_length,
            context_length=context_length,

            # distr_output="neg_bin",
            # scaling="mean",
            nonnegative_pred_samples=True,
            aug_prob=0,
            lr=5e-4,

            # estimator args
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            time_feat=estimator_args["time_feat"],

            # rope_scaling={
            #     "type": "linear",
            #     "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
            # },

            batch_size=batch_size,
            num_parallel_samples=100,
            trainer_kwargs = {"max_epochs": max_epochs,}, # <- lightning trainer arguments # modified by FILIP to speed up testing
        )
    
    return estimator


def train_estimator(estimator, data, cache_data = True, shuffle_buffer_length = 1000):
    predictor = estimator.train(data, cache_data = cache_data, shuffle_buffer_length = shuffle_buffer_length)
    return predictor

def make_predictions(predictor, data, num_samples = 100):
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=data,
        predictor=predictor,
        num_samples=num_samples
    )

    forecast = list(tqdm(forecast_it, total=len(data), desc="Forecasting batches"))

    return list(forecast[0].samples.mean(axis = 0))

    






