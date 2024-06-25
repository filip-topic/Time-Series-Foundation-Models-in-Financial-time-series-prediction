import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta

def get_stock_price_data(ticker: str, start: str, end: str):
    stock_data = yf.download(ticker, start=start, end=end, interval="1d")["Adj Close"]
    return stock_data

def get_inflation_data(country_code: str, start_year: int, end_year: int):
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/FP.CPI.TOTL?date={start_year}:{end_year}&format=json"
    response = requests.get(url)
    data = response.json()
    inflation_data = pd.json_normalize(data[1])
    return inflation_data[['date', 'value']].rename(columns={'date': 'Year', 'value': 'Inflation'})

def get_interest_rate_data(series_id: str, start: str, end: str, api_key: str):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={start}&observation_end={end}"
    response = requests.get(url)
    data = response.json()
    interest_rate_data = pd.json_normalize(data['observations'])
    return interest_rate_data[['date', 'value']].rename(columns={'date': 'Date', 'value': 'Interest Rate'})

def get_data(data_type: str, **kwargs):
    """
    Download data about stock prices, inflation, or interest rates.

    Parameters:
    data_type (str): Type of data to download ('stock', 'inflation', 'interest_rate').
    kwargs: Additional keyword arguments specific to the data type.
        For 'stock':
            - ticker (str): The ticker symbol of the company (e.g., 'AAPL' for Apple).
            - start (str): The start date in the format 'YYYY-MM-DD'.
            - end (str): The end date in the format 'YYYY-MM-DD'.
        For 'inflation':
            - country_code (str): The ISO 3166-1 alpha-3 country code (e.g., 'USA' for the United States).
            - start_year (int): The start year (e.g., 2000).
            - end_year (int): The end year (e.g., 2020).
        For 'interest_rate':
            - series_id (str): The series ID for the interest rate data (e.g., 'FEDFUNDS' for the Federal Funds Rate).
            - start (str): The start date in the format 'YYYY-MM-DD'.
            - end (str): The end date in the format 'YYYY-MM-DD'.
            - api_key (str): Your FRED API key.

    Returns:
    DataFrame: A pandas DataFrame containing the requested data.
    """

    if data_type == 'stock':
        return get_stock_price_data(kwargs['ticker'], kwargs['start'], kwargs['end'])
    elif data_type == 'inflation':
        return get_inflation_data(kwargs['country_code'], kwargs['start_year'], kwargs['end_year'])
    elif data_type == 'interest_rate':
        return get_interest_rate_data(kwargs['series_id'], kwargs['start'], kwargs['end'], kwargs['api_key'])
    else:
        raise ValueError("Invalid data_type. Expected 'stock', 'inflation', or 'interest_rate'.")
    
# for simple model testing purposes
def get_simle_data():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    apple_stock_price = get_stock_price_data("AAPL", start_date, end_date)
    apple_stock_price = apple_stock_price.reset_index()
    apple_stock_price.columns = ['ds', 'y']
    return apple_stock_price


if __name__ == "__main__":
    inflation_data = get_inflation_data('USA', 2000, 2020)
    for inf in inflation_data["Inflation"]:
        print(inf)

    