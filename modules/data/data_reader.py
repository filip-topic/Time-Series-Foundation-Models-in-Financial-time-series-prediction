import os
import pandas as pd

def read_data(type="stock", frequency = "daily", match = []):
    # Define the path to the data folder
    data_folder_path = os.path.join(os.path.dirname(__file__), '../../data')
    
    # List all files in the data folder
    files = os.listdir(data_folder_path)
    
    # Filter files that contain the type, frequency, and all strings in match in their name and have .csv extension
    def all_match_conditions(file, match):
        return all(m in file for m in match)
    
    csv_files = [file for file in files if type in file and frequency in file and all_match_conditions(file, match)]
    
    # Read all filtered CSV files into a list of DataFrames
    data_frames = [pd.read_csv(os.path.join(data_folder_path, file)) for file in csv_files]
    
    return data_frames
