import numpy as np

def root_mean_square_error(predictions, observations):
    predictions = np.array(predictions)
    observations = np.array(observations)
    mse = np.mean((predictions - observations) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def relative_root_mean_square_error(predictions, observations, benchmark_predictions):
    rmse = root_mean_square_error(predictions, observations)
    benchmark_rmse = root_mean_square_error(benchmark_predictions, observations)
    rrmse = rmse / benchmark_rmse
    return rrmse


if __name__ == "__main__":
    # Example predictions and observations
    model_predictions = [2.5, 0.0, 2.1, 7.8]
    actual_observations = [3.0, -0.5, 2.0, 8.0]
    benchmark_predictions = [2.0, 1.0, 2.0, 7.0]  

    rrmse_score = relative_root_mean_square_error(model_predictions, actual_observations, benchmark_predictions)
    print("RRMSE for the predictions:", rrmse_score)
