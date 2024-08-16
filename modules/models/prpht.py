
from prophet import Prophet

def get_prophet(train):
    m = Prophet()
    m.fit(train)
    return m

def prophet_forecast(m, prediction_length):
    future = m.make_future_dataframe(periods=prediction_length)
    forecast = m.predict(future)
    return forecast['yhat'].tail(prediction_length)

def get_prophet_predictions(train, prediction_length):
    m = get_prophet(train=train)
    return list(prophet_forecast(m, prediction_length))
