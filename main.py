from datetime import datetime
from surrogate.problem_space import ProblemSpace
from surrogate.clustering_simulation import ClusteringSimulation
from surrogate.meta_features import FeatureSpace
from surrogate.surrogate import Surrogate
# from optimization.tpot import Tpot

# TODO - Service to plot/analize data
# TODO - add data layers (raw/stage/processing/production)

if __name__ == "__main__":
    print(f"Starting time: {datetime.now()}")
    try:
        # ProblemSpace(sample_n=5)
        ClusteringSimulation()
        # Surrogate()
        # TPOT()
    except Exception as e:
        print(e)
    print(f"Ending time: {datetime.now()}")
