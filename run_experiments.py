import time
from data import get_graph, get_nodes
from general_attack import attack
import random
from itertools import product
import pandas as pd
from tqdm import tqdm
from math import prod
from numpy import random as rand
from settings import *
from multiprocessing import Process, Manager, Queue, active_children

## TODO: Get slack for the final system
## TODO: Get list of distances during experiment
## TODO: Fix data.py

class Config:
    # Object to hold all the configuration information for an experiment
    # Just for readability

    def __init__(self, kwargs):
        self.__dict__.update(kwargs)

    def __contains__(self, item):
        return item in self.__dict__

    def __str__(self):
        return ", ".join(f"{key}: {value}" for key, value in self.__dict__.items())

    def update(self, d):
        self.__dict__.update(d)

def iterate_over_ranges(d):
    # Given a dictionary of lists, return the cartesian product as a list of dictionaries

    kv_iterators = []
    for key, values in d.items():
        kv_iterators.append([(key, value) for value in values])
    return map(dict, product(*kv_iterators))

def run_experiment(config_dict, G_dict, queue=None):
    # Given a configuration, run the experiment and return the results
    config = Config(config_dict)

    config.G = G_dict["G"].copy()

    config.path_selector = path_selector_classes[config.experiment_type](config)
    config.perturber = perturber_classes[config.perturber_class](config)

    start_time = time.time()
    stats_dict = attack(config)
    time_taken = time.time() - start_time

    original_distance = config.path_selector.distance(config.G)

    result_dict = {
        "Original Distance": original_distance,
        "Time Taken": time_taken,
        "Success": stats_dict["Final Distance"] >= config.path_selector.goal,
        **stats_dict,
        **{k:v for k,v in config.__dict__.items() if k not in ["G", "path_selector", "perturber"]},
    }

    if queue is not None:
        queue.put(result_dict)

    return result_dict

if __name__ == "__main__":
    seed_plus = 7
    # sets seeds for reproducibility
    random.seed(81238.2345+9235.893456*seed_plus)
    rand.seed(892358293+27493463*seed_plus)

    results = []

    processes = []
    queue = Queue(maxsize=n_processes)

    pbar = tqdm("Progress", total=total_experiments)

    for condition_dict in iterate_over_ranges(condition_ranges):
        config = Config(condition_dict)

        # Get the graph and the nodes for testing
        G = get_graph(config)
        all_nodes = get_nodes(G, config)
        manager = Manager()
        G_dict = manager.dict({"G": G})

        for nodes in all_nodes:
            config.nodes = nodes
            if config.experiment_type == "Single":
                config.source, config.target = nodes
            elif config.experiment_type == "Sets":
                config.S, config.T = nodes
            elif config.experiment_type == "Multiple Pairs":
                # print("hm")
                config.pairs = nodes

            # print("HMMM", config.experiment_type, len(config.experiment_type))
            # print("AHHHHHHH", config.pairs)

            for config_dict in iterate_over_ranges(configuration_ranges):
                # For each possible configuration, update the config object and run the experiment
                config.update(config_dict)
                for trial_index in range(config.n_trials):
                    if sum(process.is_alive() for process in processes)>n_processes or (len(processes) == total_experiments and len(results) < total_experiments):
                        results.append(queue.get(timeout=600))
    
                        pd.DataFrame.from_records(results).to_pickle(f"results.pkl")
                        
                        pbar.update(1)

                    process = Process(target=run_experiment, args=(config.__dict__.copy(), G_dict, queue))
                    processes.append(process)
                    process.start()
                    # process.join() ## REMOVE