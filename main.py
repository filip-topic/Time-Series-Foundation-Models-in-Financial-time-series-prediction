# modules
from modules.data import  data_reader, data_loader
from modules.sr import result_saver
from modules.models import lag_llama
from modules.experiment.tscv import get_tscv_results, get_summary, extract_metrics
from modules.experiment import run_experiment
from modules.visualization import graphs
from modules.fine_tuning import lag_llama_ft

# import libraries
import itertools

# parameters
PREDICTION_LENGTH = [1] #fixed
TICKER = ["AAPL"] # fixed
FREQUENCY = ["minutely"] 
TYPE_OF_DATA = ["return"] 
MODELS = ["arima", "llama", "autoregressor", "fine-tuned Llama"] # fixed non-argument
FOLDS = [20] # fixed
CONTEXT_LENGTH = [64]
METRICS = ['r2', 'mse', 'mae', 'rmse', 'mda', "mape"] # fixed non-argument

# fine-tuning parameters
BATCH_SIZE = [5] # fixed
MAX_EPOCHS = [4] # fixed

# data parameters
FT_LENGTH = [100]
START_DATE = ["2023-07-07"] # fixed
END_DATE = ["2024-07-07"] # fixed

# experiment parameters
parameters = [
    PREDICTION_LENGTH,
    TICKER,
    FREQUENCY,
    TYPE_OF_DATA,
    FOLDS,
    CONTEXT_LENGTH,
    BATCH_SIZE,
    MAX_EPOCHS,
    FT_LENGTH,
    START_DATE,
    END_DATE
]

# all combinations of parameters
all_combinations = itertools.product(*parameters)

# loop for running experiment
for combination in all_combinations:
    run_experiment.save_results(
        prediction_length=combination[0],
        ticker=combination[1],
        frequency=combination[2],
        type_of_data=combination[3],
        folds=combination[4],
        context_length=combination[5],
        batch_size=combination[6],
        max_epochs=combination[7],
        ft_length=combination[8],
        start_date=combination[9],
        end_date=combination[10]
    )