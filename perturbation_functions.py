import networkx as nx
import gurobipy as gp
from gurobipy import GRB



# Minimize the perturbations of weights of path edges so that all paths have length at least goal
def pathattack(G, paths, all_path_edges, goal, global_budget, local_budget, write_model=False, verbose=False):
    assert type(G) == nx.DiGraph, "Input must be an nx.DiGraph"

    # Initialize Model
    m = gp.Model("Spanner Attack")
    if not verbose: m.setParam('OutputFlag', 0)

    # Initialize Variables
    d = m.addVars(all_path_edges, vtype=GRB.CONTINUOUS, name="d", lb=0, ub=local_budget) # Perturbations for each edge. Vtype can be changed to GRB.INTEGER if desired

    # The perturbed distance must be more than k times the original distance on every path
    # print([[(G.edges[i, j]["weight"]) for i,j in zip(path[:-1], path[1:])] for path in paths])
    # print([[d[i,j] for i,j in zip(path[:-1], path[1:])] for path in paths])
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
        return dict()

    return { k : v.X for k,v in d.items() }