import os
import pandas as pd
from nixtlats import TimeGPT
from nixtla import NixtlaClient

nixtla_client = NixtlaClient(
    #api_key=os.getenv("NIXTLA_API")
    api_key = "nixtla-tok-cbPeVBdkCzxF15OGVqvHcHMKxFbYNLPH500WpmFFrimf0zBvhH3QMmTA9w6rh5pXLf2oTej9JQ4tIARJ"
)


def get_timegpt_forecast(data, prediction_length, frequency, ft_steps = 0, x = None):

    freq_map = {
        'minutely': 'T',  # minute frequency
        'hourly': 'H',    # hourly frequency
        'daily': 'B',     # daily frequency
        'weekly': 'W',    # weekly frequency
        'monthly': 'M',   # monthly frequency
        'quarterly': 'Q'  # quarterly frequency
    } 
    
    if x is None:
        if ft_steps > 0:
            forecast_df = nixtla_client.forecast(df = data, finetune_steps=ft_steps, h = prediction_length, freq = freq_map[frequency], time_col="ds", target_col="y")
        else:
            forecast_df = nixtla_client.forecast(df = data, h = prediction_length, freq = freq_map[frequency], time_col="ds", target_col="y")
    else:
        x_df_future = pd.DataFrame()

        # forecasting future exogenous variables
        for col in x.columns:
            if col == "ds":
                continue
            d = x[["ds", col]]
            d.columns = ["ds", "y"]
            fcst = nixtla_client.forecast(df=d, h=prediction_length, freq=freq_map[frequency], time_col="ds", target_col = "y")
            x_df_future["ds"] = fcst["ds"]
            x_df_future[col] = fcst["TimeGPT"]

        #merging target variable with past exogenous variables
        data = pd.merge(data, x, on="ds", how="inner")

        if ft_steps > 0:
            forecast_df = nixtla_client.forecast(df = data, finetune_steps=ft_steps, X_df=x_df_future, h=prediction_length, time_col="ds", target_col="y")
        else:
            forecast_df = nixtla_client.forecast(df = data, X_df=x_df_future, h=prediction_length, time_col="ds", target_col="y")


    forecast = forecast_df.tail(prediction_length)["TimeGPT"]

    return list(forecast)



