import time
from data import get_input_data
from general_attack import attack
from selector_classes import *
import random
from itertools import product
from utils import *
from perturbation_functions import *
import pandas as pd
from tqdm import tqdm
from math import prod
from numpy import random as rand

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __contains__(self, item):
        return item in self.__dict__

    def __str__(self):
        return str(self.__dict__)

    def update(self, d):
        self.__dict__.update(d)

if __name__ == "__main__":
    seed_plus = 7
    # sets seeds for reproducibility
    random.seed(81238.2345+9235.893456*seed_plus)
    rand.seed(892358293+27493463*seed_plus)

    n_trials = 1

    path_selector_classes = {
        "Single": [SinglePairPathSelector],
        "Sets": [SetsPathSelector],
        "Multiple Pairs": [MultiPairPathSelector],
    }

    configuration_ranges = dict(
        graph_name = ["er", "ba", "ws", "Facebook"],
        weights = ['Poisson', 'Uniform', 'Equal'],
        experiment_type = ["Single", "Sets", "Multiple Pairs"],
        perturbation_function = [pathattack],
        global_budget = [1000],
        local_budget = [100],
        epsilon = [0.1],
        k = [2, 5],
        top_k = [1],
        max_iterations = [500],
    )

    results = []

    for trial_number in range(n_trials):
        for config_values in tqdm(product(*configuration_ranges.values()), desc="Configuration", total=prod(len(v) for v in configuration_ranges.values())): # position=0, leave=True, 
            config = Config(**dict(zip(configuration_ranges.keys(), config_values)))

            input_data = get_input_data(config.graph_name, weights=config.weights, experiment_type=config.experiment_type)
            config.update(input_data)
            
            if config.experiment_type == "Single":
                config.source, config.target = config.nodes
            elif config.experiment_type == "Sets":
                config.S, config.T = config.nodes
            elif config.experiment_type == "Multiple Pairs":
                config.pairs = config.nodes
            
            for path_selector_class in path_selector_classes[config.experiment_type]:
                config.path_selector = path_selector_class(config)
                original_distance = config.path_selector.distance(config.G)
                config.goal = original_distance * config.k + config.epsilon


                start_time = time.time()
                perturbations, stats_dict = attack(config)
                perturbations = {k:v for k,v in perturbations.items() if v != 0}
                time_taken = time.time() - start_time

                success = stats_dict["Final Distance"] >= config.goal
                status = stats_dict["Status"]
                print(f"\n{status}\n")

                del config.__dict__["G"]
                del config.__dict__["path_selector"]
                config.__dict__["Path Selector"] = path_selector_class.name
                config.perturbation_function = str(config.perturbation_function)

                results.append({
                    "Trial Number": trial_number,
                    "Original Distance": original_distance,
                    "Time Taken": time_taken,
                    "Perturbations": perturbations,
                    "Total Perturbation": sum(perturbations.values()) if perturbations is not None else None,
                    "Success": success,
                    **config.__dict__,
                    **stats_dict
                })

                pd.DataFrame.from_records(results).to_pickle(f"results.pkl")