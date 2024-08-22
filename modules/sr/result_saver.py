import os

def save_results(df, file_name, type = "evaluation", partial = False):
    project_folder = os.getcwd()
    if partial:
        data_folder = os.path.abspath(os.path.join(project_folder, "results", "partial", type))
    else:
        data_folder = os.path.abspath(os.path.join(project_folder, "results", type))
    file_path = os.path.join(data_folder, file_name)
    df.to_csv(file_path, index=True)
    print(f"Data has been written to '{file_path}'.")
    

