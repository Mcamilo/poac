from .config.path import path_meta_dataset, path_simulations, path_meta_features
from os.path import isfile, join, dirname, abspath, exists
from os import listdir
import pandas as pd
class Surrogate:
    def __init__(self):
        """ Induces the Surrogate Model for the problem space"""
        print("[Surrogate Modeling]>> starting...")
        self._generate_meta_dataset()
    
    def _generate_meta_dataset(self):
        print("[Surrogate Modeling]>> generating surrogate dataset...")
        simulations = self._merge_csv_files_in_directory(path_simulations)
        meta_features = self._merge_csv_files_in_directory(path_meta_features)        
        merged_df = pd.merge(simulations, meta_features, on=0)
        merged_df.to_csv(join(path_meta_dataset, "MetaDataset.csv"), index=False)
        

    def _merge_csv_files_in_directory(self, directory):
    # Initialize an empty list to store DataFrames
        dataframes = []

        # Loop through the files in the directory
        for filename in listdir(directory):
            if filename.endswith('.csv'):
                filepath = join(directory, filename)
                # Read each CSV file into a DataFrame and append it to the list
                df = pd.read_csv(filepath,header=None)
                dataframes.append(df)

        # Concatenate all DataFrames in the list vertically (stacked on top of each other)
        if dataframes:
            merged_df = pd.concat(dataframes, axis=0, ignore_index=True)
            return merged_df
        else:
            print("No CSV files found in the directory.")
            return None
    