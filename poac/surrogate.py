import pandas as pd
from .problem_space import ProblemSpace
from .clustering_simulation import ClusteringSimulation
from .meta_features import FeatureSpace
from sklearn.ensemble import RandomForestRegressor
from .config.meta_features import meta_features
from .config.path import path_meta_dataset
import warnings 
from sklearn.metrics.cluster._unsupervised import (
    davies_bouldin_score,
    silhouette_score,
)

class Surrogate:
    def __init__(self):
        """ Induces the Surrogate Model for the problem space"""
        # path configs here
        self.problem_space = ProblemSpace()
        self.simulations = ClusteringSimulation()
        self.features = FeatureSpace()
        
    def populate_problem_space(self, sample_size=None, keep=False):
        self.problem_space.synthesize_clustering_problems(sample_size=sample_size, keep=keep)
        
    def simulate_solutions(self, solutions_size=25, mean=0.5, std_dev=0.19):
        self.simulations.algorithm_space(solutions_size, mean, std_dev) 

    def extract_metafeatures(self):
        self.features.mf_clustering_algos()
        self.features.generate_meta_dataset()
    
    def build_model(self, seed=30):
        df_surrogate = pd.read_csv(f"{path_meta_dataset}/Surrogate_MetaDataset_sv6.csv")
        data = df_surrogate[meta_features]

        x_train, y_train = data.values[:, :-1], data.values[:, -1]
        rf_regressor = RandomForestRegressor(random_state=seed, n_estimators=100, n_jobs=-1)
        rf_regressor.fit(x_train, y_train)
        return rf_regressor

class SurrogateScorer:
    def __init__(self, model, meta_features, cvi=[silhouette_score, davies_bouldin_score]) -> None:
        self.model = model
        self.meta_features = meta_features
        self.cvi = cvi

    def __call__(self, estimator, X):
        try:
            warnings.filterwarnings('ignore')
            cluster_labels = estimator.fit_predict(X)
            mf = self.meta_features.copy()
            mf.extend([score(X,cluster_labels) for score in self.cvi])
            surrogate_score = self.model.predict([mf])[0]
            return surrogate_score if len(set(cluster_labels)) > 1 else -float('inf') 
        except Exception as e:
            raise TypeError(f"{e}")