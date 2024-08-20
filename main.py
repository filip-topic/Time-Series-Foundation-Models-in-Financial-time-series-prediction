

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
from datetime import datetime, timedelta


today_date = datetime.today().strftime('%Y-%m-%d')
tomorrow_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
yesterday_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')


# data-specific parameters
TYPE_OF_DATA = ["index", "fx", "commodity", "crypto"] 
RTRN = [True]
EXOGENOUS_DATA = [True]
TICKER = ["NATURAL_GAS"] # "NASDAQ Composite", "Dow Jones Industrial Average"
FREQUENCY = ["weekly"]
START_DATE = ["2019-01-01"] 
END_DATE = ["2024-01-01"] 

# experiment-specific parameters
PREDICTION_LENGTH = [1] #fixed
FOLDS = [5] # fixed
CONTEXT_LENGTH = [32, 64, 128]
TSCV_REPEATS = [6] 

# fine-tuning parameters
BATCH_SIZE = [5] # fixed
MAX_EPOCHS = [4] # fixed
FT_LENGTH = [200]
FT_FREQUENCY = [5] 
FT_GAP = [0]

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
    'ft_gap',
    "tscv_repeats",
    "rtrn",
    "exogenous_data"
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
    FT_GAP,
    TSCV_REPEATS,
    RTRN,
    EXOGENOUS_DATA
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
    if params.type_of_data == "index" and params.ticker not in ["S&P 500", "FTSE 100", "NASDAQ", "DOWJ"]:
        return False
    if params.type_of_data == "crypto" and params.ticker not in ["BTC", "ETH"]:
        return False
    if params.type_of_data == "exchange_rate" and params.ticker not in ["USD/GBP"]:
        return False
    if params.type_of_data == "commodity" and params.ticker not in ["WTI", "NATURAL_GAS"]:
        return False

    # frequency constraints
    if params.type_of_data in ["commodity"] and params.frequency in ["minutely", "hourly"]: # we can only get daily, weekly and monthly data
        return False
    
    # start_date and end_date constraints
    if params.start_date > params.end_date:
        return False
    
        # this is to make sure that for minutely data we only use data from one day
    start = datetime.strptime(params.start_date, "%Y-%m-%d")
    end = datetime.strptime(params.end_date, "%Y-%m-%d")
    difference = end - start
    gap = difference.days
    if params.frequency == "minutely" and gap > 2:
        return False
    if params.frequency == "daily" and gap < 200:
        return False
    if params.frequency == "daily" and gap > 740:
        return False
    
    # start and end time cosntraints
        # this is to make sure we only request the data we can get
    if params.type_of_data in ["crypto", "exchange_rate"] and params.frequency == "minutely" and params.end_date != tomorrow_date:
        return False
    
        # this is to make sure we only get one day worth od index data (yesterday)
    if params.type_of_data == "index" and params.frequency == "minutely" and params.end_date != today_date:
        return False
    
    # exogenous variables constraint
    if params.exogenous_data == True and params.frequency != "monthly":
        return False
    if params.exogenous_data == True and params.type_of_data not in ["index", "fx"]:
        return False

    # fine-tune length constraint
    """if params.ft_length != 4 * params.context_length:
        return False"""

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
        ft_gap=combination.ft_gap,
        tscv_repeats=combination.tscv_repeats,
        rtrn = combination.rtrn,
        exogenous_data = combination.exogenous_data
    )
    #break # for testing purposes