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

## TODO: Get slack for the final system
## TODO: Get list of distances during experiment
## TODO: Fix data.py

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __contains__(self, item):
        return item in self.__dict__

    def __str__(self):
        return ", ".join(f"{key}: {value}" for key, value in self.__dict__.items())

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
        perturbation_function = [pathattack],
        global_budget = [10],
        local_budget = [100],
        epsilon = [0.1],
        k = [2, 5],
        top_k = [1, 50],
        max_iterations = [200],
    )

    condition_ranges = dict(
        graph_name = ["er", "ba", "ws"],
        weights = ['Poisson', 'Uniform', 'Equal'],
        experiment_type = ["Single", "Sets", "Multiple Pairs"],
    )

    results = []

    def run_experiment(config):

        original_distance = config.path_selector.distance(config.G)
        config.goal = original_distance * config.k + config.epsilon

        start_time = time.time()
        stats_dict = attack(config)
        time_taken = time.time() - start_time

        success = stats_dict["Final Distance"] >= config.goal
        status = stats_dict["Status"]
        config_iter.set_postfix_str(f"Last Status: {status}")

        result_dict = {
            "Original Distance": original_distance,
            "Time Taken": time_taken,
            "Success": success,
            **stats_dict
        }

        return result_dict
        

    def iterate_over_ranges(d):
        kv_iterators = []
        for key, values in d.items():
            kv_iterators.append([(key, value) for value in values])
        return map(dict, product(*kv_iterators))

    for trial_number in range(n_trials):
        print(f"\n----[ Trial {trial_number} ]----\n")
        for condition_dict in tqdm(iterate_over_ranges(condition_ranges), desc="Conditions", total=prod(len(v) for v in condition_ranges.values())):
            config = Config(**condition_dict)
            # print(f"\nCurrent Conditions: {config}")

            input_data_dict = get_input_data(config.graph_name, weights=config.weights, experiment_type=config.experiment_type)

            config.update(input_data_dict)

            if config.experiment_type == "Single":
                config.source, config.target = config.nodes
            elif config.experiment_type == "Sets":
                config.S, config.T = config.nodes
            elif config.experiment_type == "Multiple Pairs":
                config.pairs = config.nodes

            config_iter = tqdm(iterate_over_ranges(configuration_ranges), desc="Configurations", total=prod(len(v) for v in configuration_ranges.values()), position=1)
            for config_dict in config_iter:
                config.update(config_dict)
                
                for path_selector_class in path_selector_classes[config.experiment_type]:
                    config.path_selector = path_selector_class(config)


                    result_dict = run_experiment(config)

                    results.append({
                        "Trial Number": trial_number,
                        **condition_dict,
                        **config_dict,
                        **result_dict
                    })

                    pd.DataFrame.from_records(results).to_pickle(f"results.pkl")