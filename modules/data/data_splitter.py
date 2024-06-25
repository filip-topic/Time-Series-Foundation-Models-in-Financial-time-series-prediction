def split_data(data, prediction_length):
    train = data[:len(data)-prediction_length]
    test = data[len(data)-prediction_length:]
    return train, test