from tqdm import tqdm
import time
import networkx as nx
from math import isclose

class State:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Make every path between s and t have length of at least goal
def attack(c, solver_lock=None):
    add_times = []
    perturb_times = []

    G = c.G.copy()
    # TODO: Merge edge perturbations for undirected graphs
    if type(G) == nx.Graph:
        G = G.to_directed(as_view=True)

    state = State(
        G_prime=G.copy(),
        perturbation_dict=dict(),
        paths = set(),
        all_path_edges = set(),
    )

    perturbation_result = dict()

    status= "Fail: Unknown"
    for i in range(c.max_iterations):
        assert all([state.G_prime.edges[pair]["weight"] == G.edges[pair]["weight"]+state.perturbation_dict[pair] for pair in state.perturbation_dict]), [(state.G_prime.edges[pair]["weight"], G.edges[pair]["weight"], state.perturbation_dict[pair]) for pair in state.perturbation_dict]
        # Add Paths
        add_start_time = time.time()
        new_paths = c.path_selector.get_next(state=state)
        add_times.append(time.time() - add_start_time)

        # Update State
        if not new_paths:
            status = "Fail: No Paths Returned By Selector"
            break
        else:
            state.paths.update(set(new_path for new_path, _ in new_paths))
            for new_path, _ in new_paths: 
                state.all_path_edges.update(zip(new_path[:-1], new_path[1:]))

        # Perturb Graph
        c.perturber.add_paths(new_paths)
        # if c.use_multithreading: 
        #     solver_lock.acquire()
        perturb_start_time = time.time()
        perturbation_result = c.perturber.perturb()
        perturb_times.append(time.time() - perturb_start_time)
        # if c.use_multithreading: 
        #     solver_lock.release()

        if perturbation_result["Perturbation Failure"]:
            status = "Fail: Failure in Perturber"
            break

        # Reset G_prime
        for edge, perturbation in state.perturbation_dict.items():
            state.G_prime.edges[edge[0], edge[1]]["weight"] = G.edges[edge[0], edge[1]]["weight"]

        state.perturbation_dict = perturbation_result["Perturbation Dict"]

        # Perturb Graph
        for edge, perturbation in state.perturbation_dict.items():
            state.G_prime.edges[edge[0], edge[1]]["weight"] += perturbation

        # print([(nx.path_weight(G, path, "weight"), nx.path_weight(state.G_prime, path, "weight")) for path, goal in state.paths])

        # Check if we are done
        if c.path_selector.check_if_done(state=state):
            status = "Success"
            break

    if status != "Success" and c.max_iterations == i+1:
        status = "Fail: Max Iterations"

    stats_dict = {
        "Number of Paths": len(state.paths),
        "Number of Edges": len(state.all_path_edges),
        "Add Times": add_times,
        "Perturb Times": perturb_times,
        "Iterations": i+1,
        "Final Distance": c.path_selector.distance(state.G_prime),
        "Status": status,
        **perturbation_result
    }

    return stats_dict