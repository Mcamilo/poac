from os import listdir, makedirs
from os.path import isfile, join, dirname, abspath, exists
import pandas as pd
from pymfe.mfe import MFE
from sklearn.preprocessing import MinMaxScaler
from .config.meta_features import features, meta_features_names
#from fcmeans import FCM
from .config.path import path_clustering_problems, path_meta_features, path_meta_dataset, path_simulations
import warnings

class FeatureSpace:
    def __init__(self):
        """ Extraction of MF and Populates the Feature space"""
        print("[Feature Space]>> Starting...")
        if not exists(path_meta_features):
            makedirs(path_meta_features)
        self.meta_data_path = join(path_meta_dataset, "Surrogate_MetaDataset_sv6.csv")
        
        print("[Feature Space]>> Done...")


    def _get_clustering_problems_data(self):
        print("[Feature Space]>> Listing clustering problems...")
        return [f for f in listdir(path_clustering_problems) if isfile(join(path_clustering_problems, f))]

    def mf_clustering_algos(self):
        print("[Feature Space]>> Extracting Feature Space...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clustering_problems = self._get_clustering_problems_data()
            for idx, file_name in enumerate(clustering_problems):
                    print(str(idx), "/", str(len(clustering_problems)))
                    print(file_name)
                    
                    file_path = join(path_clustering_problems, file_name)
                    X = pd.read_csv(file_path)
            
                    scaler = MinMaxScaler()
                    X = scaler.fit_transform(X)
                    """ Extract MF"""
                    mfe = MFE(groups="all", features=features)
                    
                    mfe.fit(X)
                    ft = mfe.extract()
                    ft[0].insert(0,"file_name")
                    ft[1].insert(0,file_name)
                    df = pd.DataFrame(columns=ft[0],data=[ft[1]])
                    
                    df.to_csv(join(path_meta_features, file_name), index=False, header=None)

    def _merge_csv_files_in_directory(self, directory):
    # Initialize an empty list to store DataFrames
        dataframes = []

        # Loop through the files in the directory
        for filename in listdir(directory):
            if filename.endswith('.csv'):
                filepath = join(directory, filename)
                # Read each CSV file into a DataFrame and append it to the list
                df = pd.read_csv(filepath, header=None)
                dataframes.append(df)

        # Concatenate all DataFrames in the list vertically (stacked on top of each other)
        if dataframes:
            merged_df = pd.concat(dataframes, axis=0, ignore_index=True)
            return merged_df
        else:
            print("No CSV files found in the directory.")
            return None
        
    def generate_meta_dataset(self):
        print("[Feature Space]>> generating surrogate dataset...")
        
        simulations = self._merge_csv_files_in_directory(path_simulations)
        meta_features = self._merge_csv_files_in_directory(path_meta_features)        
        
        simulations.columns = ['file_name','sil','dbs','ari','cluster_diff','cluster_pred','clusters']
        meta_features.columns = meta_features_names
        
        merged_df = pd.merge(simulations, meta_features, on="file_name")
        merged_df.to_csv(self.meta_data_path, index=False)