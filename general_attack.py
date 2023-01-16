import types
from tqdm import tqdm
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import time

class PathSelector:
    def __init__(self, name, generator_function, update_every_iteration=False, **generator_function_kwargs):
        self.name = name
        self.generator_function = generator_function
        self.update_every_iteration = update_every_iteration

        self.generator = None
        self.source = None
        self.target = None
        self.generator_function_kwargs = generator_function_kwargs

    def initialize_generator(self, G, source, target):
        self.source = source
        self.target = target
        self.generator = self.generator_function(G, source, target, **self.generator_function_kwargs)
    
    def update_graph(self, new_graph):
        self.generator = self.generator_function(new_graph, self.source, self.target, **self.generator_function_kwargs)

    def __iter__(self):
        if not self.generator:
            raise Exception("Generator not initialized. Call update_generator first.")
        return self.generator

    def __next__(self):
        return next(self.generator)

    def __repr__(self):
        return self.name

# Minimize the perturbations of weights of path edges so that all paths have length at least goal
def solve(G, paths, all_path_edges, goal, global_budget, local_budget, write_model=False, verbose=False):
    assert type(G) == nx.DiGraph, "Input must be an nx.DiGraph"

    num_nodes = G.number_of_nodes()

    # Initialize Model
    m = gp.Model("Spanner Attack")
    if not verbose: m.setParam('OutputFlag', 0)

    # Initialize Variables
    d = m.addVars(all_path_edges, vtype=GRB.INTEGER, name="d", lb=0, ub=local_budget) # Perturbations for each edge. Vtype may be GRB.CONTINUOUS, but constaints need to change
    


    # The perturbed distance must be more than k times the original distance on every path
    m.addConstrs((gp.quicksum((G.edges[i, j]["weight"]+d[i,j]) for i,j in zip(path[:-1], path[1:])) >= goal for path in paths), name="attack") 
    # The total of our perturbations must be under the global budget
    m.addConstr(d.sum() <= global_budget, name="budget") 

    # Minimize the perturbation
    m.setObjective(d.sum(), sense=GRB.MINIMIZE) 

    if write_model: m.write("final model.lp")

    # Solve
    m.optimize()

    # Check if the model was infeasible
    if m.Status == 3:
        print("Model Infeasible")
        return None
        # raise Exception("Model was infeasible")
    
    # Return Perturbations
    return { k : v.X for k,v in d.items() }

# Make every path between s and t have length of at least goal
def attack(G, source, target, goal, selector, global_budget, local_budget, top_k=1, max_iterations=500):
    paths = set()
    all_path_edges = set()

    # G_prime = G.copy()

    selector.initialize_generator(G, source, target)

    add_times = []
    solve_times = []

    pbar = tqdm(range(max_iterations), position=1)
    for i in pbar:

            add_start_time = time.time()
            for _ in range(top_k):
                new_path = next(selector, None)
                if new_path is None: break

                paths.add(tuple(new_path))
                all_path_edges.update(zip(new_path[:-1], new_path[1:]))
            add_times.append(time.time() - add_start_time)

            solve_start_time = time.time()
            perturbation_dict = solve(G, paths, all_path_edges, goal, global_budget, local_budget)
            solve_times.append(time.time() - solve_start_time)

            if perturbation_dict is None:
                return None, None

            G_prime = G.copy()
            for edge, perturbation in perturbation_dict.items():
                G_prime.edges[edge[0], edge[1]]["weight"] += perturbation

            shortest_path_length = nx.shortest_path_length(G_prime, source=source, target=target, weight='weight')
            pbar.set_description(f"    Current Shortest Path Length: {shortest_path_length} | Goal: {goal} ")

            if shortest_path_length >= goal:
                break

            if selector.update_every_iteration:
                selector.update_graph(G_prime)

    if shortest_path_length < goal:
        print("Max Iterations Reached Without Solution")
        return None, None

    stats_dict = {
        "Number of Paths": len(paths),
        "Number of Edges": len(all_path_edges),
        "Add Times": add_times,
        "Solve Times": solve_times,
        "Mean Add Time": sum(add_times)/len(add_times),
        "Mean Solve Time": sum(solve_times)/len(solve_times),
        "Iterations": i+1,
    }

    return perturbation_dict, stats_dict