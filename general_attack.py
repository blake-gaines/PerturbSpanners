from tqdm import tqdm
import time

# from selector_functions import random_shortest_paths

# Make every path between s and t have length of at least goal
def attack(conditions):
    # path_selector = random_shortest_paths(G, path_selector.source, path_selector.target, goal, weight="weight")
    paths = set()
    all_path_edges = set()

    add_times = []
    perturb_times = []

    current_distance = path_selector.distance(G)

    G_prime = G.copy()

    pbar = tqdm(range(max_iterations), position=1)
    for i in pbar:
            # print("Adding")
            add_start_time = time.time()
            # new_paths = next(path_selector, None)
            new_paths = path_selector.get_next(G=G_prime, current_distance=current_distance)
            if not new_paths:
                break
            else:
                paths.update(new_paths)
                for new_path in new_paths: 
                    all_path_edges.update(zip(new_path[:-1], new_path[1:]))
            add_times.append(time.time() - add_start_time)

            # print("Perturbing")
            perturb_start_time = time.time()
            perturbation_dict = perturbation_function(G, paths, all_path_edges, goal, global_budget, local_budget)
            perturb_times.append(time.time() - perturb_start_time)

            if not perturbation_dict:
                break

            G_prime = G.copy()
            for edge, perturbation in perturbation_dict.items():
                G_prime.edges[edge[0], edge[1]]["weight"] += perturbation

            current_distance = path_selector.distance(G_prime)
            pbar.set_description(f"    Current Distance: {current_distance} | Goal: {goal}")
            
            if current_distance >= goal:
                break

            if path_selector.update_every_iteration:
                path_selector.update_graph(G_prime)

    stats_dict = {
        "Number of Paths": len(paths),
        "Number of Edges": len(all_path_edges),
        "Add Times": add_times,
        "Perturb Times": perturb_times,
        "Iterations": i+1,
        "Final Distance": current_distance,
    }

    return perturbation_dict, stats_dict