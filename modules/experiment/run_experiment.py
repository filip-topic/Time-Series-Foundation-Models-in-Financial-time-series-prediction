from modules.data import  data_reader, data_loader
from modules.sr import result_saver
from modules.models import lag_llama
from modules.experiment.tscv import get_tscv_results, get_summary, extract_metrics
from modules.visualization import graphs
from modules.fine_tuning import lag_llama_ft

def save_results(prediction_length,
                ticker, 
                frequency, 
                type_of_data,
                folds,
                context_length,
                batch_size,
                max_epochs,
                ft_length,
                start_date,
                end_date):
    
    if frequency == "minutely":
        start_date = "2024-07-22"
        end_date = "2024-07-27"

    if frequency == "hourly":

        start_date = "2023-06-01"
        end_date = "2024-01-01"

    # data config
    data_config = {"ticker" : ticker,
               "frequency" : frequency,
               "start" : start_date,
               "end" : end_date}


    """ft_data_config = {"ticker" : ticker,
                     "frequency" : frequency,
                     "start" : ft_start_date,
                     "end" : start_date}"""
    
    # loading the data
    data = data_loader.get_data(data_type=type_of_data, kwargs=data_config)
    #ft_data = data_loader.get_data(data_type=type_of_data, kwargs=ft_data_config)

    data_length = len(data)
    #ft_length = len(ft_data)

    if folds == "max":
        folds = int((data_length - ft_length) / prediction_length)

    """
    # fine tuning Lag Llama - preparing the training data
    ft_train_data = lag_llama.prepare_data(data=ft_data, 
                                       prediction_length=0, 
                                       frequency=frequency)
    
    # fine tuning Lag Llama - creating the predictor object
    predictor = lag_llama_ft.get_predictor(prediction_length=prediction_length, 
                                       context_length=context_length, 
                                       batch_size=batch_size, 
                                       max_epochs=max_epochs)
    
    # fine tuning Lag Llama - fine-tuning the predictor object
    predictor = predictor.train(ft_train_data, 
                            cache_data = True, 
                            shuffle_buffer_length = 1000)
    """
    
    # getting the TSCV results
    r, p = get_tscv_results(data = data,
                           prediction_horizon=prediction_length,
                           context_length=context_length, 
                           folds=folds, 
                           frequency=frequency,
                           fine_tune_length=ft_length,
                           batch_size=batch_size,
                           max_epochs=max_epochs,
                           fine_tune_frequency=5)
    
    # experiment name
    experiment_name = f"P_L={prediction_length}__T={ticker}__FR={frequency}__T_O_D={type_of_data}__FO={folds}__C_L_T_S={context_length}__S_D={start_date}__E_D={end_date}__FT_L={ft_length}__D_L={data_length}.csv"

    # saving the results
    result_saver.save_results(r, experiment_name, type="evaluation")
    result_saver.save_results(p, experiment_name, type="prediction")    

