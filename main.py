from datetime import datetime
from surrogate.synthesize import Synthesize
from surrogate.meta_features import MetaFeatures
from surrogate.problem_space import ProblemSpace
from surrogate.surrogate import Surrogate
# from optimization.tpot import Tpot

# TODO - Service to plot/analize data
# TODO - add data layers (raw/stage/processing/production)

if __name__ == "__main__":
    print(f"Starting time: {datetime.now()}")
    try:
        Synthesize(sample_n=5)
        MetaFeatures()
        # ProblemSpace()
        # Surrogate()
        # TPOT()
    except Exception as e:
        print(e)
    print(f"Ending time: {datetime.now()}")
