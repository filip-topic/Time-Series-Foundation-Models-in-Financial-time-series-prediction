def get_autoregressor_prediction(train, prediction_length):
    return [train["y"].iloc[-1]] * prediction_length 