import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

def print_evaluate(actual, **predictions):
    for name, pred in predictions.items():
        mse = mean_squared_error(actual, pred)
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        rmse = np.sqrt(mse)
        
        print(f"Evaluation for {name}:")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared (R^2): {r2}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print("-" * 30)

def get_single_evaluation(actual, prediction):
    return list(mean_squared_error(actual, prediction), mean_absolute_error(actual, prediction), r2_score(actual, prediction), np.sqrt(mean_squared_error(actual, prediction)))

def get_multiple_evaluations(actual, **predictions):
    output = pd.DataFrame()
    for name, pred in predictions.items():
        output[name] = get_single_evaluation(actual, pred)
    return output

# Example usage:
if __name__ == "__main__":
    actual = [10, 20, 30, 40, 50]
    prediction1 = [12, 22, 32, 42, 52]
    prediction2 = [8, 18, 28, 38, 48]
    
    evaluate(actual, prediction1=prediction1, prediction2=prediction2)