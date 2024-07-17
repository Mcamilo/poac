# Problem-Oriented AutoML in Clustering
Problem-oriented AutoML in Clustering (PoAC), a novel approach that automatically generates complete machine learning pipelines for human-defined clustering problems in a data-driven manner.


## Example

```python
from datetime import datetime
import warnings
import poac
import joblib

if __name__ == "__main__":
    print(f"Starting time: {datetime.now()}")
    try:
        warnings.simplefilter("ignore")
        surrogate = poac.Surrogate()
        # Problem Space Design (Step 1)
        surrogate.populate_problem_space(sample_size=5, keep=False)
        # Feature Space Mapping (Step 2)
        surrogate.simulate_solutions()
        surrogate.extract_metafeatures()
        # Surrogate Modeling (Step 3)
        surrogate_model = surrogate.build_model()
        # tpot-clustering(surrogate_model)
        
        # joblib.dump(surrogate_model, 'optimization/tpot/models/       random_forest_model.joblib')
        
    except Exception as e:
        print(e)
    print(f"Ending time: {datetime.now()}")

```