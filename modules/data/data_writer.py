import data_loader
import os
import pandas as pd
from datetime import datetime, timedelta

def write_csv(data, filename):
    # Define the path to the data folder
    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

    # Ensure the data folder exists
    #os.makedirs(data_folder, exist_ok=True)
    
    # Define the full path to the CSV file
    file_path = os.path.join(data_folder, filename)
    
    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"The file '{filename}' already exists. No new file will be created.")
        return
    
    # Write the DataFrame to a CSV file
    data.to_csv(file_path, index=False)
    print(f"Data has been written to '{file_path}'.")


years = 1
frequency = "daily"

#stock data
stock_data = data_loader.get_all_stock_data(years=years, frequency=frequency)
write_csv(stock_data, f"all_sp500_stocks_{years}years_{frequency}")

#return data
returns_data = data_loader.get_all_stock_returns(years=years, frequency=frequency)
write_csv(returns_data, f"all_sp500_returns_{years}years_{frequency}")

#inflation data
country_code = "USA"
start_year = 2000
end_year = 2023
inflation_data = data_loader.get_inflation_data(country_code, start_year, end_year)
write_csv(inflation_data, f"{country_code}_inflation_{start_year}_{end_year}")

#interest rate data
fred_api_key = os.getenv("fred_api_key")
start_date = "2000-01-01"
end_date = "2020-12-31"
series_id = "FEDFUNDS"
interest_rate_data = data_loader.get_interest_rate_data(series_id, start_date, end_date, fred_api_key)
write_csv(interest_rate_data, f"{series_id}_{start_date}_{end_date}")
