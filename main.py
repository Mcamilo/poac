from datetime import datetime
from surrogate.synthesize import Synthesize
from surrogate.meta_features import MetaFeatures
from surrogate.surrogate import Surrogate
# from optimization.tpot import Tpot

# TODO - Service to plot data

if __name__ == "__main__":
    print(f"Starting time: {datetime.now()}")
    try:
        Synthesize()
        # MetaFeatures()
        # Surrogate()
        # TPOT()
    except Exception as e:
        print(e)
    print(f"Ending time: {datetime.now()}")
