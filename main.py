

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
from collections import namedtuple



# data-specific parameters
TYPE_OF_DATA = ["return", "stock", "index", "crypto", "exchange_rate", "commodity"] 
TICKER = ["IBM", "S&P500", "BTC", "USD/GBP", "AUX"]
FREQUENCY = ["minutely", "daily", "weekly", "monthly"]
START_DATE = ["2023-03-01", "2024-08-05"] 
END_DATE = ["2024-03-01", "2024-08-09"] 

# experiment-specific parameters
PREDICTION_LENGTH = [1] #fixed
FOLDS = [20] # fixed
CONTEXT_LENGTH = [32, 64, 128]

# fine-tuning parameters
BATCH_SIZE = [5] # fixed
MAX_EPOCHS = [4] # fixed
FT_LENGTH = [150, 300, 500]
FT_FREQUENCY = [5] # fixed
FT_GAP = [0, 32, 64, 128, 256]

# filter-specific 


# Define a namedtuple with all the parameter names
ExperimentParams = namedtuple('ExperimentParams', [
    'prediction_length', 
    'ticker', 
    'frequency', 
    'type_of_data', 
    'folds', 
    'context_length', 
    'batch_size', 
    'max_epochs', 
    'ft_length', 
    'start_date', 
    'end_date', 
    'ft_frequency', 
    'ft_gap'
])

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
    END_DATE,
    FT_FREQUENCY,
    FT_GAP
]

# all combinations of parameters
all_combinations = itertools.product(*parameters)

# Convert combinations to namedtuples
all_combinations_named = [ExperimentParams(*combination) for combination in all_combinations]

# filtering out impossible combinations of parameters
def filter_combinations(params):

    # ticker constraints
    if params.type_of_data == "stock" and params.ticker not in ["IBM"]:
        return False
    if params.type_of_data == "index" and params.ticker not in ["S&P500"]:
        return False
    if params.type_of_data == "crypto" and params.ticker not in ["BTC"]:
        return False
    if params.type_of_data == "exchange_rate" and params.ticker not in ["USD/GBP"]:
        return False
    if params.type_of_data == "commodity" and params.ticker not in ["AUX"]:
        return False

    # frequency constraints
    if params.type_of_data == "commodity" and params.ticker in ["minutely", "hourly"]:
        return False
    
    # start_date and end_date constraints
    if params.start_date > params.end_date:
        return False
    



    return True

# Apply filter
valid_combinations_named = filter(filter_combinations, all_combinations_named)



# loop for running experiment
for combination in valid_combinations_named:
    run_experiment.save_results(
        prediction_length=combination.prediction_length,
        ticker=combination.ticker,
        frequency=combination.frequency,
        type_of_data=combination.type_of_data,
        folds=combination.folds,
        context_length=combination.context_length,
        batch_size=combination.batch_size,
        max_epochs=combination.max_epochs,
        ft_length=combination.ft_length,
        start_date=combination.start_date,
        end_date=combination.end_date,
        ft_frequency=combination.ft_frequency,
        ft_gap=combination.ft_gap
    )