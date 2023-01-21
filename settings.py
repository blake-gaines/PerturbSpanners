from perturbation_classes import *
from selector_classes import *
from math import prod

# Which selectors to test
path_selector_classes = {
    "Single": SinglePairPathSelector,
    # "Sets": SetsPathSelector,
    "Multiple Pairs": MultiPairPathSelector,
}

# Which perturbers to test
perturber_classes = {
    "PathAttack": PathAttack,
    "GreeedyFirst": GreedyFirst,
    "MinFirst": GreedyMin,
}

# Hyperparameter Ranges
configuration_ranges = dict(
    path_selector_class = list(path_selector_classes.keys()),
    perturber_class = list(perturber_classes.keys()),
    global_budget = [1000],
    local_budget = [100],
    epsilon = [0.1],
    k = [3,5],
    top_k = [1],
    max_iterations = [200],
)

# Experimental condition ranges
condition_ranges = dict(
    # graph_name = ["Facebook", "er", "ba", "ws"],
    graph_name = ["Facebook"],
    weights = ['Poisson', 'Uniform', 'Equal'],
    experiment_type = ["Single", "Multiple Pairs"],
    n_nodes_per_experiment = [10, 100],
    n_experiments = [20],
    n_trials = [5],
    min_path_length = [5],
)

n_processes = 8

total_experiments = prod(len(v) for v in configuration_ranges.values()) * prod(len(v) for v in condition_ranges.values())