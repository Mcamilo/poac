from os.path import isfile, join, dirname, abspath, exists

path_default = join(
    dirname(dirname(dirname(abspath(__file__)))), "data"
)

path_clustering_problems = join(path_default, "clustering_problems")

path_meta_dataset = join(path_default, "meta_dataset")

path_simulations = join(path_default, "simulations")
