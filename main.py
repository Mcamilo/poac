from datetime import datetime
import warnings
import poac
# import optimization 

# from optimization.tpot import Tpot

# TODO - deal with the data handling (file or memory)
# TODO - create a service to plot/visualize data

if __name__ == "__main__":
    print(f"Starting time: {datetime.now()}")
    try:
        warnings.simplefilter("ignore")
        surrogate = poac.Surrogate()
        surrogate.populate_problem_space(sample_size=5, keep=False)
        surrogate.simulate_solutions()
        surrogate.extract_metafeatures()
        surrogate_model = surrogate.build_model()
        
        # TODO - validate surrogate model
        # TODO - optimization module

    except Exception as e:
        print(e)
    print(f"Ending time: {datetime.now()}")
