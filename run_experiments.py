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
from multiprocessing import Process, Manager, Queue, Lock, active_children
from time import sleep

## TODO: Fix data.py
## TODO: Fix case for undirected graphs

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

def run_experiment(config_dict, G, queue=None, solver_lock=None):
    sleep(random.random()*5)
    # Given a configuration, run the experiment and return the results
    config = Config(config_dict)

    if use_multithreading:
        config.G = G[(config.graph_name, config.weights)]
    else:
        config.G = G

    config.path_selector = path_selector_classes[config.experiment_type](config)
    config.perturber = perturber_classes[config.perturber_class](config)

    start_time = time.time()
    stats_dict = attack(config, solver_lock)
    time_taken = time.time() - start_time

    original_distance = config.path_selector.distance(config.G)

    result_dict = {
        "Original Distance": original_distance,
        "Time Taken": time_taken,
        "Success": stats_dict["Status"] == "Success",
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

    if use_multithreading:
        processes = []
        queue = Queue(maxsize=n_processes)
        solver_lock = Lock()
        manager = Manager()
        G_dict = manager.dict()

    pbar = tqdm("Progress", total=total_experiments)

    for condition_index, condition_dict in enumerate(iterate_over_ranges(condition_ranges)):
        config = Config(condition_dict)
        config.condition_index = condition_index
        config.use_multithreading = use_multithreading

        # Get the graph and the nodes for testing
        G = get_graph(config)
        all_nodes = get_nodes(G, config)

        if use_multithreading: 
            G_dict[(config.graph_name, config.weights)] = G
            

        for nodes in all_nodes:
            config.nodes = nodes
            if config.experiment_type == "Single":
                config.source, config.target = nodes
            elif config.experiment_type == "Sets":
                config.S, config.T = nodes
            elif config.experiment_type == "Multiple Pairs":
                config.pairs = nodes

            for configuration_index, config_dict in enumerate(iterate_over_ranges(configuration_ranges)):
                # For each possible configuration, update the config object and run the experiment
                config.configuration_index = configuration_index
                config.update(config_dict)
                for _ in range(config.n_trials):
                    if use_multithreading:
                        process = Process(target=run_experiment, args=(config.__dict__.copy(), G_dict, queue, solver_lock))
                        processes.append({"Process": process, "Config": config.__dict__.copy()})
                        process.start()

                        while len(active_children()) >= n_processes:
                            results.append(queue.get())
                            pd.DataFrame.from_records(results).to_pickle(output_path)
                            pbar.update(1)
                    else:
                        results.append(run_experiment(config.__dict__.copy(), G))
                        pd.DataFrame.from_records(results).to_pickle(output_path)
                        pbar.update(1)
    if use_multithreading:
        while len(results) < total_experiments:
            results.append(queue.get())
            pbar.update(1)
        pd.DataFrame.from_records(results).to_pickle(output_path)

        sleep(5)
        
        failed = []
        for process in processes:
            assert not process["Process"].is_alive()
            if process["Process"].exitcode != 0:
                process["Config"]["Exit Code"] = process["Process"].exitcode
                failed.append(process["Config"])
        pd.DataFrame.from_records(failed).to_pickle(failed_path)
