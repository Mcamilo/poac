from .config.synth_combinations import combinations_default_config
from repliclust import set_seed, Archetype, DataGenerator
import itertools
import os
import pandas as pd
import random

path_default = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

# TODO
# substitute pandas for pyarrow


class Synthesize:
    def __init__(self, dir=path_default, config=combinations_default_config):
        """Synthesize clustering problems
        :param dir: Path to save the synthesized clustering problems
        :param config: A config dict to synthesize different combinations of clustering problems
        """
        print("[Synthesize]>> Starting...")
        self.PATH = dir
        self.combinations_config = config
        self._generate_combinations()
        self._synthesize_clustering_problems()

    def _generate_combinations(self):
        print("[Synthesize]>> Generating combinations...")
        """Generates a dict with combinations of different ranges of parameters for the synthesis of data
        """
        self.combinations = [
            {key: value for key, value in zip(self.combinations_config.keys(), combo)}
            for combo in itertools.product(*self.combinations_config.values())
        ]

    def _synthesize_clustering_problems(self):
        """ Randomily picks values from the combinations dict for each Archetype parameter 
        """
        print("[Synthesize]>> Generating clustering problems...")
        for i, config in enumerate(self.combinations):
            set_seed(random.choice(range(1, 100)))
            print(f"[Synthesize]>> Config: {i}/{len(self.combinations)}")
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
            # Create the output directory if it doesn't exist
            if not os.path.exists(self.PATH):
                os.makedirs(self.PATH)
            output_path = os.path.join(self.PATH, file_name)

            df.to_csv(output_path, index=False)
        print("[Synthesize]>> Done...")
