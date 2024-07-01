
from prophet import Prophet

def get_prophet(train):
    m = Prophet()
    m.fit(train)
    return m

def prophet_forecast(m, prediction_length):
    future = m.make_future_dataframe(periods=prediction_length)
    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prediction_length)