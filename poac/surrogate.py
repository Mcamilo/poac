import pandas as pd
from .problem_space import ProblemSpace
from .clustering_simulation import ClusteringSimulation
from .meta_features import FeatureSpace
from sklearn.ensemble import RandomForestRegressor
from .config.meta_features import meta_features
from .config.path import path_meta_dataset

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