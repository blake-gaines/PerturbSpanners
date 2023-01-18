import time
from DataSets import DataSets
import networkx as nx
from data import get_graph
import numpy as np
from general_attack import attack
from selector_classes import *
import random
from itertools import product
from selector_functions import *
from perturbation_functions import *
import pandas as pd
from tqdm import tqdm

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __contains__(self, item):
        return item in self.__dict__

    def __str__(self):
        return str(self.__dict__)

if __name__ == "__main__":
    random.seed(7)

    weighted = False

    G = DataSets.get_directed_networkx_graph(dataset=DataSets.FACEBOOK, lcc=False)
    if weighted:
        nx.set_edge_attributes(G, {e: random.randint(1,10) for e in G.edges()}, "weight")
    else:
        nx.set_edge_attributes(G, 1, "weight")

    print("Graph Information")
    print('nodes: {} \nedges: {}\n'.format(G.number_of_nodes(), G.number_of_edges()))

    # node_pairs = [[2565,999],[2254,996],[2308,65],[3067,2743],[1259,933],
    #         [2310,1934],[2130,402],[2569,1577],[2272,857],[1808,936],
    #         [3452,371],[2818,1670],[2000,87],[1969,1286],[2733,26],
    #         [963,423],[3285,2789],[1041,414],[3414,3051],[1888,1715]]
    node_pairs = []
    while len(node_pairs) < 3:
        a,b = random.choices(list(G.nodes()),k=2)
        if nx.has_path(G, a, b):
            node_pairs.append((a,b))
    print("Node Pairs:", node_pairs)

    n_trials = 1

    path_selector_classes = [
    # (SinglePairPathSelector, dict(name="Ours", update_every_iteration=False, weight="weight")),
    their_selector,
    # (SinglePairPathSelector, dict(name="Random Shortest Paths", generator_function=random_shortest_paths, update_every_iteration=False, weight="weight")),
    edge_centrality_selector,
    # SinglePairPathSelector(name="One Sided Random", generator_function=random_one_sided, update_every_iteration=False, weight="weight"),
    ]

    configuration_ranges = dict(
        path_selector_class = path_selector_classes,
        perturbation_function = [pathattack],
        global_budget = [1000],
        local_budget = [100],
        epsilon = [0.1],
        k = [2, 3],
        node_pair = node_pairs,
        top_k = [1, 50],
        max_iterations = [100],
    )

    results = []

    for trial_number in range(n_trials):
        for config_values in tqdm(product(*configuration_ranges.values()), position=0, leave=True):
            config = Config(**dict(zip(configuration_ranges.keys(), config_values)))
            config.G = G
            # print("\n========\nConfig:", config)

            config.source, config.target = config.node_pair
            original_path_length = nx.shortest_path_length(G, config.source, config.target, weight="weight")
            config.goal = original_path_length * config.k + config.epsilon
            # print(f"Original Path Length: {original_path_length} | Goal: {goal}")
            
            print("Config:", config)
            config.path_selector = config.path_selector_class(c=config)

            print("Selector:", config.path_selector)


            start_time = time.time()
            perturbations, stats_dict = attack(config)
            perturbations = {k:v for k,v in perturbations.items() if v != 0}
            time_taken = time.time() - start_time

            # print("Cost: ", sum(perturbations.values()))
            if stats_dict["Final Distance"] >= config.goal: 
                print("\n\nSUCCESS")
            else:
                print("\n\nFAIL")
            print("STATS:", {k:v for k,v in stats_dict.items() if "Times" not in k})
            
            results.append({
                "Trial Number": trial_number,
                "Original Path Length": original_path_length,
                "Time Taken": time_taken,
                "Perturbations": perturbations,
                "Total Perturbation": sum(perturbations.values()) if perturbations is not None else None,
                **config.__dict__,
                **stats_dict
            })
            print("\n\n================================\n\n")

        pd.DataFrame.from_records(results).to_pickle("results.pkl")