import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
import os

alphavantage_api_key = 1234

def get_stock_price_data(ticker: str, frequency = "daily", start = "2023-07-09", end = "2024-07-09"):
    freq_map = {
        "minutely": "1m",
        "hourly": "60m",
        "daily": "1d",
        "weekly": "1wk",
        "monthly": "1mo",
        "quarterly": "3mo"
    }

    """
    if start == "":
        end = datetime.now()
        start = end - timedelta(days=365 * years)
    else:
        start = datetime.strptime(start, "%Y-%m-%d")
        end = start + timedelta(days=365 * years)
        end = end.strftime("%Y-%m-%d")
    """
    ################## for index data ###################################
    indices = {
    'S&P 500': '^GSPC',
    'FTSE 100': '^FTSE',
    'NASDAQ Composite': '^IXIC',
    'Dow Jones Industrial Average': '^DJI',
    'Russell 2000': '^RUT',
    'CAC 40': '^FCHI',
    'DAX': '^GDAXI',
    'Nikkei 225': '^N225',
    'Hang Seng': '^HSI',
    'Shanghai Composite': '000001.SS',
    'S&P/ASX 200': '^AXJO',
    'BSE Sensex': '^BSESN',
    'Nifty 50': '^NSEI',
    'KOSPI': '^KS11',
    'Taiwan Weighted': '^TWII',
    'S&P/TSX Composite': '^GSPTSE',
    'Swiss Market Index': '^SSMI',
    'Euro Stoxx 50': '^STOXX50E',
    'FTSE MIB': 'FTSEMIB.MI',
    'IBEX 35': '^IBEX',
    'AEX': '^AEX',
    'OMX Stockholm 30': '^OMX',
    'OSEAX': '^OSEAX',
    'Bovespa': '^BVSP',
    'IPC Mexico': '^MXX',
    'MERVAL': '^MERV',
    'Santiago IPSA': '^IPSA',
    'Colcap': '^COLCAP',
    'SP 500 BRL': 'SPBRBRL',
    'BSE Smallcap': '^BSESN',
    'BSE Midcap': '^BSEMID',
    'IBOVESPA': '^BVSP',
    'JSE Top 40': 'J200.JO',
    'MSCI World Index': '^MSCI'
    }

    if ticker in indices.keys():
        ticker = indices[ticker]
    ####################################################################

    data = yf.download(ticker, start=start, end=end, interval=freq_map[frequency])["Close"]

    # Reset the index to move the date from the index to a column
    data = data.reset_index()

    # Rename the columns
    data.columns = ['ds', "y"]

    # Clean the data
    """data = data.loc[:, ~data.iloc[0].isna()]
    data = data.dropna(axis=1, how='all')
    data = data.fillna(method='ffill')"""

    if frequency in ["hourly", "minutely"]:
        data["ds"] = data['ds'].dt.tz_localize(None)

    return data

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

#def get_exchange_rate_data(ticker, start, end):


def get_data(data_type: str, **kwargs):

    if data_type == 'stock':
        return get_stock_price_data(ticker=kwargs["kwargs"]["ticker"], frequency=kwargs["kwargs"]["frequency"], start=kwargs["kwargs"]["start"], end=kwargs["kwargs"]["end"])
    elif data_type == "return":
        return get_stock_returns(ticker=kwargs["kwargs"]["ticker"], frequency=kwargs["kwargs"]["frequency"], start=kwargs["kwargs"]["start"], end=kwargs["kwargs"]["end"])
    elif data_type == "exchange_rate":
        return None
    elif data_type == 'inflation':
        return get_inflation_data(kwargs['country_code'], kwargs['start_year'], kwargs['end_year'])
    elif data_type == 'interest_rate':
        return get_interest_rate_data(kwargs['series_id'], kwargs['start'], kwargs['end'], kwargs['api_key'])
    else:
        raise ValueError("Invalid data_type. Expected 'stock', 'inflation', or 'interest_rate'.")
    
def get_all_stock_data(years=1, frequency="daily"):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    
    freq_map = {
        "minutely": "1m",
        "hourly": "60m",
        "daily": "1d",
        "weekly": "1wk",
        "monthly": "1mo",
        "quarterly": "3mo"
    }
    
    if frequency not in freq_map:
        raise ValueError("Frequency must be 'daily', 'weekly', or 'monthly'.")
    
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    
    data = yf.download(sp500_tickers, start=start_date, end=end_date, interval=freq_map[frequency])['Close']
    
    columns = data.columns
    data = data.reset_index()
    data.columns = ["ds"] + list(columns)

    data = data.loc[:, ~data.iloc[0].isna()]

    data = data.dropna(axis=1, how='all')

    data = data.fillna(method='ffill')

    return data

    
def get_all_stock_returns(years=1, frequency="daily"):
    df = get_all_stock_data(years = years, frequency=frequency)
    returns_df = df.copy()
    
    # Calculate returns for each column except the first one ('ds')
    for col in returns_df.columns[1:]:
        returns_df[col] = returns_df[col].pct_change()
    
    # Drop the first row because pct_change will result in NaN for the first entry
    returns_df.dropna(inplace=True)
    
    return returns_df

def get_stock_returns(ticker, frequency = "daily", start = "2023-07-09", end = "2024-07-09"):
    df = get_stock_price_data(ticker=ticker, frequency=frequency, start = start, end = end)

    returns_df = df.copy()
    
    # Calculate returns for each column except the first one ('ds')
    for col in returns_df.columns[1:]:
        returns_df[col] = returns_df[col].pct_change()
    
    # Drop the first row because pct_change will result in NaN for the first entry
    returns_df.dropna(inplace=True)
    
    return returns_df
    

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

    