import os
import pandas as pd
import chardet

def read_data(match = [], location = "data", file_type = "csv"):
    # Define the path to the data folder

    project_folder = os.getcwd()

    data_folder_path = os.path.abspath(os.path.join(project_folder, location))

    
    # List all files in the data folder
    files = os.listdir(data_folder_path)
    
    # Filter files that contain the type, frequency, and all strings in match in their name and have .csv extension
    def all_match_conditions(file, match):
        return all(m in file for m in match)
    
    #csv_files = [file for file in files if type in file and frequency in file and all_match_conditions(file, match)]
    csv_files = [file for file in files if all_match_conditions(file, match)]
    #return csv_files

    # Read all filtered CSV files into a list of DataFrame

    if file_type == "csv":
        data_frames = [pd.read_csv(os.path.join(data_folder_path, file), index_col=0) for file in csv_files]
    else:
        # this is to fix encoding problem
        with open (os.path.join(data_folder_path, csv_files[0]), "r") as f:
            result = chardet.detect(f.read())
        #return result['encoding']
        data_frames = [pd.read_excel(os.path.join(data_folder_path, file), index_col=0, encoding=result['encoding']) for file in csv_files]

    
    return data_frames
