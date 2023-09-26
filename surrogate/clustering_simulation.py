import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, dirname, abspath, exists
from .config.path import path_clustering_problems, path_simulations
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import davies_bouldin_score, silhouette_score
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
        self.onlyfiles = [f for f in listdir(path_clustering_problems) if isfile(join(path_clustering_problems, f))]
        print("Total:"+str(len(self.onlyfiles)))

    def _algorithm_space(self):
        """ Produces random solutions from a list of clustering algorithms for the clustering problems 
        """
        print("[Clustering Simulation]>> Simulating solutions from algo list")

        for idx, file_name in enumerate(self.onlyfiles):
            print(str(idx), "/", str(len(self.onlyfiles)))
            print(file_name)
            
            # alterar
            file_path = join(path_clustering_problems, file_name)
            X = pd.read_csv(file_path)
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            n_clusters = int(file_name.split('-')[1].replace("clusters",""))
            print("Num Clusters:", n_clusters)
            
            diff = 8
            if (n_clusters < (diff+2)): #para ter no mínimo 2 clusters
                diff = n_clusters-2
            for rep in range(n_clusters-diff, n_clusters+diff, 1):
                cluster_algo = [AgglomerativeClustering(n_clusters=rep, metric='euclidean', linkage='ward'),
                                KMeans(n_clusters=rep, n_init="auto"),
                                KMedoids(n_clusters=rep),
                                MiniBatchKMeans(n_clusters=rep, batch_size=10,n_init="auto")]

                c = random.choice(cluster_algo)
                cluster_labels = c.fit_predict(X)
                sil = silhouette_score(X, cluster_labels)
                dbs = davies_bouldin_score (X, cluster_labels)
                
                datasets = ([file_name]+[type(c).__name__]+[np.round(sil, decimals=2)]+[np.round(dbs, decimals=2)]+[abs(rep-n_clusters)])
                
                pd.DataFrame(datasets).T.to_csv(join(path_simulations,file_name), mode='a', index=False, header=False)
                