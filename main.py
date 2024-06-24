from modules.evaluation.crps_interval import crps_time_series_interval
from modules.evaluation.rmae import relative_mean_absolute_error
from modules.evaluation.rrmse import relative_root_mean_square_error
from modules.utils.arg_parser import parse_args_cmd

args = parse_args_cmd()
parsed_args = args.parse_args()

