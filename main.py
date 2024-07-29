# modules
from modules.data import  data_reader, data_loader
from modules.sr import result_saver
from modules.models import lag_llama
from modules.experiment.tscv import get_tscv_results, get_summary, extract_metrics
from modules.visualization import graphs
from modules.fine_tuning import lag_llama_ft

# import libraries
import itertools

# parameters
PREDICTION_LENGTH = 1 
TICKER = "AAPL"
FREQUENCY = ["minutely", "hourly", "daily"] 
TYPE_OF_DATA = ["stock", "return"] 
MODELS = ["arima", "llama", "autoregressor", "fine-tuned Llama"] 
FOLDS = 10 
CONTEXT_LENGTH = [32, 64, 128]
METRICS = ['r2', 'mse', 'mae', 'rmse', 'mda', "mape"]

# fine-tuning parameters
BATCH_SIZE = 10
MAX_EPOCHS = 20

# data parameters
FT_START_DATE = "2022-07-07"
START_DATE = "2023-07-07"
END_DATE = "2024-07-07"

# want to add
#TRAIN_PERIOD = # context lenghts. Should take a look into this
TRAIN_SIZE = CONTEXT_LENGTH

