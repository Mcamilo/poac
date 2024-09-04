from tpot import TPOTClustering
from sklearn.preprocessing import MinMaxScaler
from pymfe.mfe import MFE
from .surrogate import SurrogateScorer
from .config.path import path_model
from .config.meta_features import features, meta_features
import joblib
import os
import requests


class Optimizer:
    def __init__(self, data, surrogate_score=None):
        """Automatically synthesizes clustering pipelines for a given dataset"""
        self.data = data
        try:
            if surrogate_score is not None:
                self.surrogate_score = surrogate_score
            else:
                print("Loading default Surrogate model...")
                self.surrogate_score = joblib.load(path_model)
        except Exception as e:
            print(f"ERRO{e}")
            raise Exception(f"Error unable to locate a working surrogate model: {e}")
            
    # TODO -- try catch 
    def _extract_metafeatures(self, X, meta_features):
        scaler = MinMaxScaler()
        scaled_X = scaler.fit_transform(X)
        mfe = MFE(groups="all", features=features)
        mfe.fit(scaled_X)
        ft = mfe.extract()
        
        return [value for key, value in zip(ft[0], ft[1]) if key in meta_features]
    
    def synthesize(self, generations=10, population_size=50, verbosity=2, random_state=42, meta_features=meta_features):
        extracted_mf = self._extract_metafeatures(self.data, meta_features)
        tpot_clustering = TPOTClustering(generations=generations, population_size=population_size, verbosity=verbosity, random_state=random_state, scoring=SurrogateScorer(self.surrogate_score, extracted_mf))
        tpot_clustering.fit(self.data)
        
        return (tpot_clustering.export(''), tpot_clustering.fitted_pipeline_.named_steps, tpot_clustering.fitted_pipeline_[-1].labels_)

   