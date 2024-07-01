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
stock_data = data_loader.get_all_stock_data(years=years, frequency=frequency)
write_csv(stock_data, f"all_sp500_stocks_{years}years_{frequency}")

