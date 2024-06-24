import argparse

def parse_args_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_type',
        required=True,
        choices=['stock_price', 'revenue', 'profit' 'inflation', 'gdp', 'unemployment', 'interest_rate'],        
        type=str,
        help='Provide what kind of data you want forecasted'
    )
    parser.add_argument(
        '--frequency',
        required=False,
        help='Provide Provide the frequency of data'
    )
    parser.add_argument(
        '--prediction_length',
        required=True,
        help='How many instances in the future you want predicted'
    )
    parser.add_argument(
        '--benchmark_model',
        required=True,
        help='Which benchmark model do you want to compare with'
    )
    return parser