import pandas as pd
from os import listdir, makedirs, remove
from os.path import isfile, join, dirname, abspath, exists
from .config.path import path_clustering_problems, path_simulations, path_default
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import davies_bouldin_score, silhouette_score, adjusted_rand_score
import random
import numpy as np

class ClusteringSimulation:
    def __init__(self):
        """ Induces the Surrogate Model for the problem space"""
        print("[Clustering Simulation]>> Listing clustering problems...")

        if not exists(path_simulations):
            makedirs(path_simulations)
        
        self._get_clustering_problems_data()
        self._algorithm_space()

    def _get_clustering_problems_data(self):
        print("[Clustering Simulation]>> Listing clustering problems...")
        self.clustering_problems = [f for f in listdir(path_clustering_problems) if isfile(join(path_clustering_problems, f))]
        print("Total:"+str(len(self.clustering_problems)))


    def _add_noise_to_categorical_values(self, categorical_list, noise_level=0.1, categories=None):
        """
        Add noise to a list of categorical ordinal values.

        Parameters:
        - categorical_list: List of categorical ordinal values.
        - noise_level: The level of noise to introduce (a float between 0 and 1).
        - categories: List of possible categories (if None, it's inferred from the input data).

        Returns:
        - A list with added noise to the original values.
        """
        if not categories:
            categories = list(set(categorical_list))
        noisy_list = []
        for value in categorical_list:
            if random.random() < noise_level:
                # Introduce noise by randomly selecting a different category
                noisy_value = random.choice(categories)
            else:
                noisy_value = value
            noisy_list.append(noisy_value)

        return noisy_list

    def _algorithm_space(self):
        """ Produces random solutions from a list of clustering algorithms for the clustering problems 
        """
        print("[Clustering Simulation]>> Simulating solutions from algo list")
        scaler = MinMaxScaler()
        num_values = 25
        mean = 0.5
        std_dev = 0.19
        for idx, file_name in enumerate(self.clustering_problems):
            print(str(idx), "/", str(len(self.clustering_problems)))
            print(file_name)
            file_path = join(path_clustering_problems, file_name)
            data = pd.read_csv(file_path)
            labels_true = list(data['y'])
            X = scaler.fit_transform(data.drop('y',axis=1))
            n_clusters = int(file_name.split('-')[1].replace("clusters",""))
            normal_values = np.random.normal(mean, std_dev, num_values)
            solutions_output = join(path_simulations,file_name)
            if exists(solutions_output):
                remove(solutions_output)
            
            datasets = []
            for noise_level in normal_values:
                try:
                    extended_categories = list(set(labels_true))
                    random_increment = random.randint(0, 10)
                    max_element = max(extended_categories)
                    new_max = max_element + random_increment
                    for i in range(max_element + 1, new_max):
                        extended_categories.append(i)
                    noisy_data = self._add_noise_to_categorical_values(labels_true, noise_level, extended_categories)
                    sil = silhouette_score(X, noisy_data)
                    dbs = davies_bouldin_score (X, noisy_data)
                    ari = adjusted_rand_score(labels_true, noisy_data)
                    rep = len(set(noisy_data))
                    
                    datasets.append([file_name,np.round(sil, decimals=2),np.round(dbs, decimals=2),np.round(ari, decimals=2),abs(rep-n_clusters),rep,n_clusters])
                    # print(f"[Clustering Simulation]>> {datasets}")
                except Exception as e:
                    print(f"{e} for >>>> {file_name}")
                    exit()
            pd.DataFrame(datasets, columns=['filename','sil','dbs','ari','cluster_diff','cluster_predicted','cluster_true']).to_csv(solutions_output, index=False)