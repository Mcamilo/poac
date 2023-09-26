from os import listdir, makedirs
from os.path import isfile, join, dirname, abspath, exists
import pandas as pd
from pymfe.mfe import MFE
from sklearn.preprocessing import MinMaxScaler
from .config.meta_features import features
import warnings

#from fcmeans import FCM
import numpy as np
from .config.path import path_clustering_problems, path_meta_features


class FeatureSpace:
    def __init__(self):
        """ Extraction of MF and Populates the Feature space"""
        print("[Feature Space]>> Starting...")
        
        self.clustering_problems = path_clustering_problems
        self.stage_path = path_meta_features

        self._get_clustering_problems_data()
        self._mf_clustering_algos()
        
        print("[Feature Space]>> Done...")


    def _get_clustering_problems_data(self):
        print("[Feature Space]>> Listing clustering problems...")
        self.onlyfiles = [f for f in listdir(self.clustering_problems) if isfile(join(self.clustering_problems, f))]
        print("Total:"+str(len(self.onlyfiles)))

    def _mf_clustering_algos(self):
        print("[Feature Space] Extracting Feature Space...")
        for idx, file_name in enumerate(self.onlyfiles):
            print(str(idx), "/", str(len(self.onlyfiles)))
            print(file_name)
            
            file_path = join(path_clustering_problems, file_name)
            X = pd.read_csv(file_path)
    
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            """ Extract MF"""
            mfe = MFE(groups="all", features=features)
            output_path = self.stage_path
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)  # Replace "Warning" with the specific warning you want to ignore
                mfe.fit(X)
                ft = mfe.extract()
                ft[0].insert(0,"file_name")
                ft[1].insert(0,file_name)
                df = pd.DataFrame(columns=ft[0],data=[ft[1]])
                
                if not exists(output_path):
                    makedirs(output_path)
                
                df.to_csv(join(output_path, file_name), index=False)
            

        