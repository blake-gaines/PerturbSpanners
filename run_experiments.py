import time
from DataSets import DataSets
import networkx as nx
from data import get_input_data
import numpy as np
from general_attack import attack
from selector_classes import *
import random
from itertools import product
from selector_functions import *
from perturbation_functions import *
import pandas as pd
from tqdm import tqdm
from math import prod
from collections import OrderedDict
import pickle

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
    random.seed(7)

    weighted = False

    # G = DataSets.get_directed_networkx_graph(dataset=DataSets.FACEBOOK, lcc=False)
    # if weighted:
    #     nx.set_edge_attributes(G, {e: random.randint(1,10) for e in G.edges()}, "weight")
    # else:
    #     nx.set_edge_attributes(G, 1, "weight")

    # print("Graph Information")
    # print('nodes: {} \nedges: {}\n'.format(G.number_of_nodes(), G.number_of_edges()))

    # # pairs = [[2565,999],[2254,996],[2308,65],[3067,2743],[1259,933],
    # #         [2310,1934],[2130,402],[2569,1577],[2272,857],[1808,936],
    # #         [3452,371],[2818,1670],[2000,87],[1969,1286],[2733,26],
    # #         [963,423],[3285,2789],[1041,414],[3414,3051],[1888,1715]]
    # pairs = []
    # while len(pairs) < 10:p
    #     a,b = random.choices(list(G.nodes()),k=2)
    #     if nx.has_path(G, a, b):
    #         pairs.append((a,b))
    # print("Node Pairs:", pairs)

    n_trials = 1

    path_selector_classes = {
        "Single": [SinglePairPathSelector],
        "Sets": [SetsPathSelector],
        "Multiple Pairs": [MultiPairPathSelector],
    }

    configuration_ranges = dict(
        experiment_type = ["Single", "Sets", "Multiple Pairs"],
        # experiment_type = ["Single"],
        perturbation_function = [pathattack],
        global_budget = [1000],
        local_budget = [100],
        epsilon = [0.1],
        k = [2],
        top_k = [1],
        max_iterations = [100],
        path_selector_class = path_selector_classes,
    )

    # input_data = get_input_data("er", n_trials=n_trials)

    results = []

    for trial_number in range(n_trials):
        for config_values in tqdm(product(*configuration_ranges.values()), desc="Configuration", total=prod(len(v) for v in configuration_ranges.values())): # position=0, leave=True, 
            config = Config(**dict(zip(configuration_ranges.keys(), config_values)))

            input_data = get_input_data("er", experiment_type=config.experiment_type)
            config.update(input_data)
            
            if config.experiment_type == "Single":
                config.source, config.target = config.nodes
            elif config.experiment_type == "Sets":
                config.S, config.T = config.nodes
            elif config.experiment_type == "Multiple Pairs":
                config.pairs = config.nodes

            original_path_length = nx.shortest_path_length(config.G, config.source, config.target, weight="weight")
            config.goal = original_path_length * config.k + config.epsilon
            # print(f"Original Path Length: {original_path_length} | Goal: {goal}")
            
            print("Config:", config)
            for path_selector_class in path_selector_classes[config.experiment_type]:
                config.path_selector = path_selector_class(config)

                # print("Selector:", config.path_selector)


                start_time = time.time()
                perturbations, stats_dict = attack(config)
                perturbations = {k:v for k,v in perturbations.items() if v != 0}
                time_taken = time.time() - start_time

                # print("Cost: ", sum(perturbations.values()))
                success = stats_dict["Final Distance"] >= config.goal
                # print("SUCCESS" if success else "FAILURE")
                # if not success: print("FAILURE")
                status = stats_dict["Status"]
                print(f"\n{status}\n")

                # print("STATS:", {k:v for k,v in stats_dict.items() if "Times" not in k})

                del config.__dict__["G"]
                del config.__dict__["path_selector"]
                config.__dict__["Path Selector"] = path_selector_class.name
                config.perturbation_function = str(config.perturbation_function)

                results.append({
                    "Trial Number": trial_number,
                    "Original Path Length": original_path_length,
                    "Time Taken": time_taken,
                    "Perturbations": perturbations,
                    "Total Perturbation": sum(perturbations.values()) if perturbations is not None else None,
                    "Success": success,
                    **config.__dict__,
                    **stats_dict
                })
                # print("\n\n================================\n\n")

                pd.DataFrame.from_records(results).to_pickle("results.pkl")