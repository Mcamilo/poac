from os.path import isfile, join, dirname, abspath, exists

path_default = join(
    dirname(dirname(dirname(abspath(__file__)))), "data"
)

path_clustering_problems = join(path_default, "clustering_problems")

path_meta_dataset = join(path_default,"meta_dataset")
path_simulations = join(join(path_default,"meta_dataset"), "simulations")
path_meta_features = join(join(path_default,"meta_dataset"), "meta_features")


path_model = join(
    dirname(dirname(dirname(abspath(__file__)))), "models"
)