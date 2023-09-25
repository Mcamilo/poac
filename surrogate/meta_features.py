from os import listdir, makedirs
from os.path import isfile, join, dirname, abspath, exists
import pandas as pd
from pymfe.mfe import MFE
from sklearn.preprocessing import MinMaxScaler
from .config.meta_features import features
import warnings

path_default = join(
    dirname(dirname(abspath(__file__))), "data"
)

class MetaFeatures:
    def __init__(self, data_path=path_default):
        """ Extraction of MF and Populates the problem space"""
        print("[Meta Features]>> Starting...")
        
        self.raw_path = join(path_default, "raw")
        self.stage_path = join(data_path, "stage")
        self._get_clustering_problems_data()
        self._extract_meta_features()
        print("[Meta Features]>> Done...")


    def _get_clustering_problems_data(self):
        print("[Meta Features]>> Listing clustering problems...")
        self.onlyfiles = [f for f in listdir(self.raw_path) if isfile(join(self.raw_path, f))]
        print("Total:"+str(len(self.onlyfiles)))

    def _extract_meta_features(self):
        print("[Meta Features] Extracting Meta Features...")
        mfe = MFE(groups="all", features=features)
        output_path = self.stage_path
        for idx, file_name in enumerate(self.onlyfiles):
            print(str(idx), "/", str(len(self.onlyfiles)))
            print(file_name)
            
            X = pd.read_csv(self.raw_path+"/"+file_name)

            ####### Escalar os Valores?
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)  # Replace "Warning" with the specific warning you want to ignore
                mfe.fit(X)
                ft = mfe.extract()
                df = pd.DataFrame(columns=ft[0],data=[ft[1]])
                
                if not exists(output_path):
                    makedirs(output_path)
                
                df.to_csv(join(output_path, file_name), index=False)