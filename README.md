Package information: [![Python 3.10](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: LGPL v3](https://img.shields.io/badge/license-LGPL%20v3-blue.svg)](http://www.gnu.org/licenses/lgpl-3.0)

<p align="center">
<img src="https://raw.githubusercontent.com/Mcamilo/poac/master/images/poac-logo.png" width=300 />
</p>


## Overview

**Problem-oriented AutoML in Clustering (PoAC)** is a flexible and powerful framework designed to enhance the automation of clustering tasks within the AutoML landscape. PoAC leverages meta-learning and surrogate modeling to optimize clustering pipelines, offering a flexible approach that allows customization of meta-features, Clustering Validation Indices (CVIs).

## Features

- **Problem Space Generation:** Synthesize labeled clustering datasets through combinatorial analysis of dataset archetype parameters.
- **Clustering Simulations:** Create partitionings with multiple noise levels, calculate CVIs, and similarity metrics to simulate clustering performance.
- **Feature Space Construction:** Extract meta-features from the problem space datasets and combine them with the CVIs and similarity metrics to build a comprehensive meta-database.
- **Surrogate Modeling:** Train a regression model as a surrogate to predict the quality of clustering pipelines, enabling task-agnostic optimization across various clustering scenarios.
- **Clustering pipeline synthesis:** Seamlessly integrate the trained surrogate model with popular AutoML frameworks like TPOT to enhance clustering evaluations.

## Installation

To get started with PoAC, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-repo/PoAC.git
   cd PoAC
   ```

It’s recommended to use a virtual environment to manage dependencies.

2. Create a virtual environment:
    ```bash
    python3 -m venv poac-env
    source poac-env/bin/activate  # On Windows, use `poac-env\Scripts\activate`
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
We have divided the PoAC framework into two main stages: **Training of the Surrogate Model** and the **Pipeline Synthesis**. While the framework is designed to guide users through these stages sequentially, it is flexible enough to allow users to execute individual modules based on their specific needs. Additionally, PoAC comes with a pre-trained default surrogate model, enabling users to quickly start synthesizing and optimizing clustering pipelines without the need for training a new model.

### 1. Surrogate Model

```python
import poac
import joblib

surrogate = poac.Surrogate()

# Start by defining the problem space, where you synthesize clustering datasets:
surrogate.populate_problem_space(sample_size=5, keep=False)
# Simulate clustering partitionings with varying levels of noise:
surrogate.simulate_solutions()
# Extract meta-features and combine with CVIs and similarity metrics
surrogate.extract_metafeatures()
# Train the surrogate model
surrogate_model = surrogate.build_model()

# Optionally, save the surrogate model
joblib.dump(surrogate_model, 'optimization/tpot/models/random_forest_model.joblib')
```


### 2. Pipeline Synthesis

```python
import poac
from sklearn.datasets import load_breast_cancer

# Example of using PoAC with TPOT
data = load_breast_cancer().data
optimizer = poac.Optimizer(data)

sv6light_meta_features = ['attr_ent.sd','sparsity.sd', 'cov.mean','var.mean','eigenvalues.mean','sparsity.mean', 'wg_dist.sd', 'iq_range.mean','sil','dbs']
code, pipeline, labels = optimizer.synthesize(generations=3,population_size=5,meta_features=sv6light_meta_features)
```

## Results

In our experiments, integrating the PoAC surrogate model into TPOT achieved a mean Adjusted Rand Index (ARI) of 70% across 100 synthetic datasets. The model's flexibility and robustness make it suitable for a wide range of clustering tasks and AutoML applications.

## Contributing

We welcome contributions to PoAC! Please fork the repository, create a new branch, and submit a pull request. For major changes, please open an issue to discuss your proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PoAC in your research, please cite our paper:

```
@inproceedings{
silva2024benchmarking,
title={Benchmarking Auto{ML} Clustering Frameworks},
author={Matheus Camilo da Silva and Biagio Licari and Gabriel Marques Tavares and Sylvio Barbon Junior},
booktitle={AutoML Conference 2024 (ABCD Track)},
year={2024},
url={https://openreview.net/forum?id=RzUKJnph1g}
}
```
