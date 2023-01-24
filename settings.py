from perturbation_classes import *
from selector_classes import *
from math import prod

# Which selectors to test
path_selector_classes = {
    "Single": SinglePairPathSelector,
    "Sets": SetsPathSelector,
    "Multiple Pairs": MultiPairPathSelector,
}

# Which perturbers to test
perturber_classes = {
    "PathAttack": PathAttack,
    "GreedyFirst": GreedyFirst,
    "MinFirst": GreedyMin,
}

# Hyperparameter Ranges
configuration_ranges = dict(
    # path_selector_class = list(path_selector_classes.keys()),
    perturber_class = list(perturber_classes.keys()),
    global_budget = [1000],
    local_budget = [None],
    epsilon = [0.1],
    k = [5],
    top_k = [1],
    max_iterations = [200],
    max_paths = [300]
)

# Experimental condition ranges
condition_ranges = dict(
    # graph_name = ["Facebook", "er", "ba", "ws"],
    graph_name = ["Facebook", "Cora", "LastFM_Asia"],
    # graph_name = ["RoadNet"],
    weights = ['Uniform', 'Poisson', 'Equal',],
    experiment_type = list(path_selector_classes.keys()),
    n_nodes_per_experiment = [20],
    n_experiments = [50],
    n_trials = [3],
    min_path_length = [5],
)

use_multithreading = True
n_processes = 64
output_path = "results.pkl"
failed_path = "failed_experiments.pkl"

total_experiments = prod(len(v) for v in configuration_ranges.values()) * prod(len(v) for v in condition_ranges.values()) * condition_ranges["n_trials"][0] * condition_ranges["n_experiments"][0]