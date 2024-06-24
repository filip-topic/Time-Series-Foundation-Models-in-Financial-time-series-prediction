import numpy as np

def mean_absolute_error(predictions, observations):
    predictions = np.array(predictions)
    observations = np.array(observations)
    mae = np.mean(np.abs(predictions - observations))
    return mae

def relative_mean_absolute_error(predictions, observations, benchmark_predictions):
    mae = mean_absolute_error(predictions, observations)
    benchmark_mae = mean_absolute_error(benchmark_predictions, observations)
    rmae = mae / benchmark_mae
    return rmae


if __name__ == "__main__":
    # Example predictions and observations
    model_predictions = [2.5, 0.0, 2.1, 7.8]
    actual_observations = [3.0, -0.5, 2.0, 8.0]
    benchmark_predictions = [2.0, 1.0, 2.0, 7.0]

    rmae_score = relative_mean_absolute_error(model_predictions, actual_observations, benchmark_predictions)
    print("RMAE for the predictions:", rmae_score)
