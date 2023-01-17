from tqdm import tqdm
import time

# Make every path between s and t have length of at least goal
def attack(G, selector, perturbation_function, global_budget, local_budget, goal=float("inf"), max_iterations=500):
    paths = set()
    all_path_edges = set()

    add_times = []
    perturb_times = []

    pbar = tqdm(range(max_iterations), position=1)
    for i in pbar:
            add_start_time = time.time()
            new_paths = next(selector, None)
            if not new_paths:
                break
            else:
                paths.update(new_paths)
                for new_path in new_paths: 
                    all_path_edges.update(zip(new_path[:-1], new_path[1:]))
            add_times.append(time.time() - add_start_time)

            perturb_start_time = time.time()
            perturbation_dict = perturbation_function(G, paths, all_path_edges, global_budget, local_budget)
            perturb_times.append(time.time() - perturb_start_time)

            if not perturbation_dict:
                break

            G_prime = G.copy()
            for edge, perturbation in perturbation_dict.items():
                G_prime.edges[edge[0], edge[1]]["weight"] += perturbation

            current_distance = selector.get_current_distance(G_prime)
            pbar.set_description(f"    Current Distance: {current_distance} ")
            
            if current_distance >= goal:
                break

            if selector.update_every_iteration:
                selector.update_graph(G_prime)

    stats_dict = {
        "Number of Paths": len(paths),
        "Number of Edges": len(all_path_edges),
        "Add Times": add_times,
        "Perturb Times": perturb_times,
        "Iterations": i+1,
        "Final Distance": current_distance,
    }

    return perturbation_dict, stats_dict