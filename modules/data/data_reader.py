import os
import pandas as pd

def read_data(type="stock", frequency = "daily", match = [], location = "data"):
    # Define the path to the data folder

    project_folder = os.getcwd()
    if location == "data":
        data_folder_path = os.path.abspath(os.path.join(project_folder, "data"))
    else:
        data_folder_path = os.path.abspath(os.path.join(project_folder, "results", location))
    
    # List all files in the data folder
    files = os.listdir(data_folder_path)
    
    # Filter files that contain the type, frequency, and all strings in match in their name and have .csv extension
    def all_match_conditions(file, match):
        return all(m in file for m in match)
    
    #csv_files = [file for file in files if type in file and frequency in file and all_match_conditions(file, match)]
    csv_files = [file for file in files if all_match_conditions(file, match)]

    
    # Read all filtered CSV files into a list of DataFrames
    data_frames = [pd.read_csv(os.path.join(data_folder_path, file), index_col=0) for file in csv_files]

    
    return data_frames
