import numpy as np
from scipy.integrate import quad

def empirical_cdf(x, y):
    """
    Empirical CDF for a given observed value y.
    
    Parameters:
    x (float): The value at which to evaluate the CDF.
    y (float): The observed value.
    
    Returns:
    float: Empirical CDF value at x.
    """
    return 1 if x >= y else 0

def predictive_cdf(x, lower_bound, upper_bound):
    """
    Predictive CDF for interval predictions.
    
    Parameters:
    x (float): The value at which to evaluate the CDF.
    lower_bound (float): Lower bound of the prediction interval.
    upper_bound (float): Upper bound of the prediction interval.
    
    Returns:
    float: Predictive CDF value at x.
    """
    if x < lower_bound:
        return 0
    elif x > upper_bound:
        return 1
    else:
        return (x - lower_bound) / (upper_bound - lower_bound)

def crps_single_interval(lower_bound, upper_bound, y):
    """
    Calculate CRPS for a single interval prediction.
    
    Parameters:
    lower_bound (float): Lower bound of the prediction interval.
    upper_bound (float): Upper bound of the prediction interval.
    y (float): Observed value.
    
    Returns:
    float: CRPS value.
    """
    integrand = lambda x: (predictive_cdf(x, lower_bound, upper_bound) - empirical_cdf(x, y))**2
    crps, _ = quad(integrand, -np.inf, np.inf)
    return crps

def crps_time_series_interval(intervals, y_series):
    """
    Calculate CRPS for a time-series of interval predictions.
    
    Parameters:
    intervals (list of tuples): List of (lower_bound, upper_bound) for each time step.
    y_series (list of floats): List of observed values.
    
    Returns:
    float: Mean CRPS value for the time-series.
    """
    crps_values = [crps_single_interval(lb, ub, y) for (lb, ub), y in zip(intervals, y_series)]
    return np.mean(crps_values)

# Example usage
if __name__ == "__main__":
    # Example interval predictions for each time step
    intervals = [(0, 2), (1, 3), (5, 7)]
    y_series = [1, 1, 9]


    crps_score = crps_time_series_interval(intervals, y_series)
    print("CRPS for the time-series:", crps_score)
