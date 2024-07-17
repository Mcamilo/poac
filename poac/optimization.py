from tpot import TPOTClustering
from sklearn.preprocessing import MinMaxScaler
from pymfe.mfe import MFE
from .surrogate import SurrogateScorer
from .config.path import path_model, url_model
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
                if not os.path.isfile(path_model):
                    print("Downloading default model...")
                    self._download_surrogate_model(url_model, path_model)
                self.surrogate_score = joblib.load(path_model)
                
        except Exception as e:
            raise Exception(f"Error unable to locate a working surrogate model: {e}")
    
    def _download_surrogate_model(url, local_path):
        """
        Download a file from a URL to a local path.

        Args:
        url (str): URL of the file to download.
        local_path (str): Local path to save the downloaded file.
        """
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=128):
                    f.write(chunk)
        else:
            response.raise_for_status()
        
    # TODO -- try catch 
    def _extract_metafeatures(self, X):
        scaler = MinMaxScaler()
        scaled_X = scaler.fit_transform(X)

        features = ['attr_conc', 'attr_ent', 'attr_to_inst', 'cohesiveness', 'cor', 'cov',
                    'eigenvalues', 'inst_to_attr', 'iq_range', 'kurtosis', 'mad', 'max', 'mean', 'median', 'min',
                    'nr_attr', 'nr_cor_attr', 'nr_inst',
                    'one_itemset', 'range', 'sd', 'skewness',
                    'sparsity', 't2', 't3', 't4', 't_mean', 'two_itemset', 'var', 'wg_dist',
                    ]
        mfe = MFE(groups="all", features=features)
        mfe.fit(scaled_X)
        ft = mfe.extract()
        mfeatures = ['attr_conc.mean', 'attr_conc.sd', 'attr_ent.mean',
                        'attr_ent.sd', 'attr_to_inst', 'cohesiveness.mean', 'cohesiveness.sd',
                        'cov.mean', 'eigenvalues.mean', 'eigenvalues.sd',
                        'inst_to_attr', 'iq_range.mean', 'iq_range.sd',
                        'mad.mean', 'mad.sd',
                        'median.mean',
                        'median.sd',
                        'nr_attr', 'nr_cor_attr', 'nr_inst', 'one_itemset.mean',
                        'one_itemset.sd',
                        'sd.mean', 'sd.sd'
                            , 'sparsity.mean', 'sparsity.sd', 't2', 't3', 't4', 't_mean.mean',
                        't_mean.sd', 'two_itemset.mean', 'two_itemset.sd', 'var.mean', 'var.sd',
                        'wg_dist.mean', 'wg_dist.sd'
                        ]
        
        _meta_features = [value for key, value in zip(ft[0], ft[1]) if key in mfeatures]
        return _meta_features
    
    def synthesize(self, generations=10, population_size=50, verbosity=2, random_state=42):
        meta_features = self._extract_metafeatures(self.data)
        tpot_clustering = TPOTClustering(generations=generations, population_size=population_size, verbosity=verbosity, random_state=random_state, scoring=SurrogateScorer(self.surrogate_score, meta_features))
        tpot_clustering.fit(self.data)
        
        return (tpot_clustering.export(''), tpot_clustering.fitted_pipeline_.named_steps, tpot_clustering.fitted_pipeline_[-1].labels_)
    import requests

   