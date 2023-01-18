from tqdm import tqdm
import time

# from selector_functions import random_shortest_paths

class State:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Make every path between s and t have length of at least goal
def attack(c):
    # path_selector = random_shortest_paths(G, path_selector.source, path_selector.target, goal, weight="weight")

    add_times = []
    perturb_times = []

    

    state = State(
        G_prime=c.G.copy(),
        perturbation_dict=dict(),
        paths = set(),
        all_path_edges = set(),
        current_distance = c.path_selector.distance(c.G),
    )

    pbar = tqdm(range(c.max_iterations), position=1, leave=False)
    for i in pbar:
            # print("Adding")
            add_start_time = time.time()
            # new_paths = next(path_selector, None)
            new_paths = c.path_selector.get_next(state=state)
            if not new_paths:
                break
            else:
                state.paths.update(new_paths)
                for new_path in new_paths: 
                    state.all_path_edges.update(zip(new_path[:-1], new_path[1:]))
            add_times.append(time.time() - add_start_time)

            # print("Perturbing")
            perturb_start_time = time.time()
            state.perturbation_dict = c.perturbation_function(c.G, state.paths, state.all_path_edges, c.goal, c.global_budget, c.local_budget)
            perturb_times.append(time.time() - perturb_start_time)

            if not state.perturbation_dict:
                print("Failed to Perturb")
                break

            G_prime = c.G.copy()
            for edge, perturbation in state.perturbation_dict.items():
                G_prime.edges[edge[0], edge[1]]["weight"] += perturbation

            current_distance = c.path_selector.distance(G_prime)
            pbar.set_description(f"    Current Distance: {current_distance} | Goal: {c.goal}")
            
            if current_distance >= c.goal:
                break

            if c.path_selector.update_every_iteration:
                c.path_selector.update_graph(G_prime)

    stats_dict = {
        "Number of Paths": len(state.paths),
        "Number of Edges": len(state.all_path_edges),
        "Add Times": add_times,
        "Perturb Times": perturb_times,
        "Iterations": i+1,
        "Final Distance": current_distance,
    }

    return state.perturbation_dict, stats_dict