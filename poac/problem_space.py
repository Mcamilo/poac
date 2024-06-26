from .config.synth_combinations import combinations_default_config
from repliclust import set_seed, Archetype, DataGenerator
import itertools
from os.path import join, dirname, abspath, exists
from os import makedirs, rmdir
from shutil import rmtree
import pandas as pd
import random

path_default = join(
    dirname(dirname(abspath(__file__))), "data/clustering_problems"
)

# TODO
# substitute pandas for pyarrow

class ProblemSpace:
    def __init__(self, data_path=path_default, config=combinations_default_config):
        """Synthesize clustering problems
        :param data_path: Path to save the synthesized clustering problems
        :param config: A config dict to synthesize different combinations of clustering problems
        """
        print("[ProblemSpace]>> Starting...")
        self.PATH = data_path
        self.combinations_config = config
        
    # TODO - optimize this to only generate combinations as long as the n_samples
    def __generate_combinations(self, sample_size=None):
        print("[ProblemSpace]>> Generating combinations...")
        """Generates a dict with combinations of different ranges of parameters for the synthesis of data
        """
        combinations = [
            {key: value for key, value in zip(self.combinations_config.keys(), combo)}
            for combo in itertools.product(*self.combinations_config.values())
        ]
        
        if sample_size:
            combinations = combinations[:sample_size]

        return combinations

    def synthesize_clustering_problems(self, sample_size=None, keep=False):
        """ Randomily picks values from the combinations dict for each Archetype parameter 
        """
        print("[ProblemSpace]>> Generating clustering problems...")
        combinations = self.__generate_combinations(sample_size)
        
        if not keep:
                rmtree(self.PATH)
        
        if not exists(self.PATH):
                makedirs(self.PATH)
        
        for i, config in enumerate(combinations):
            set_seed(random.choice(range(1, 100)))
            print(f"[ProblemSpace]>> Config: {i}/{len(combinations)}")
            dim = random.choice(config["dim"])
            n_clusters = random.choice(config["n_clusters"])
            n_samples = random.choice(config["n_samples"])
            min_overlap = config["overlap_settings"]["min_overlap"]
            max_overlap = config["overlap_settings"]["max_overlap"]
            aspect_ref = config["aspect_ref"]
            aspect_maxmin = config["aspect_maxmin"]
            radius_maxmin = config["radius_maxmin"]
            distributions = config["distributions"]
            imbalance_ratio = random.choice(config["imbalance_ratio"])

            archetype = Archetype(
                dim=dim,
                n_clusters=n_clusters,
                n_samples=n_samples,
                min_overlap=min_overlap,
                max_overlap=max_overlap,
                aspect_ref=aspect_ref,
                aspect_maxmin=aspect_maxmin,
                radius_maxmin=radius_maxmin,
                imbalance_ratio=imbalance_ratio,
                distributions=distributions,
            )
            X, y, archetype = DataGenerator(archetype).synthesize(quiet=True)
            df = pd.DataFrame(X, columns=[f"x{ind}" for ind in range(X.shape[1])])
            df["y"] = y
            file_name = f"dim{dim}-clusters{n_clusters}-instances{n_samples}-overlap{min_overlap}-{max_overlap}-aspectref{aspect_ref}-aspectmaxmin{aspect_maxmin}-radius{radius_maxmin}-imbalance{imbalance_ratio}.csv"
            
            output_path = join(self.PATH, file_name)

            df.to_csv(output_path, index=False)
        print("[ProblemSpace]>> Done...")
