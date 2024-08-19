import os
import pandas as pd
from nixtlats import TimeGPT
from nixtla import NixtlaClient

nixtla_client = NixtlaClient(
    #api_key=os.getenv("NIXTLA_API")
    api_key = "nixtla-tok-co7tFFNHZUZfr0Q1FS7VlWmnzoZpWD3x0eVwRx6U3gCUWOoYkpMSf3BJC9fySpa96kb3QdDUPdIpPE9V"
)


def get_timegpt_forecast(data, prediction_length, frequency, ft_steps = 0):

    freq_map = {
        'minutely': 'T',  # minute frequency
        'hourly': 'H',    # hourly frequency
        'daily': 'B',     # daily frequency
        'weekly': 'W',    # weekly frequency
        'monthly': 'M',   # monthly frequency
        'quarterly': 'Q'  # quarterly frequency
    } 
    
    forecast_df = nixtla_client.forecast(df = data, finetune_steps=ft_steps, h = prediction_length, freq = freq_map[frequency], time_col="ds", target_col="y")

    

    forecast = forecast_df.tail(prediction_length)["TimeGPT"]

    return list(forecast)



