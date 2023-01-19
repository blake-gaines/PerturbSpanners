import gurobipy as gp
from gurobipy import GRB

# Minimize the perturbations of weights of path edges so that all paths have length at least goal
def pathattack(G, paths, all_path_edges, goal, global_budget, local_budget, write_model=False, verbose=False):
    result_dict = {}
    with gp.Env(empty=True) as env:
        if not verbose: env.setParam('OutputFlag', 0)
        env.start()

        # Initialize Model
        m = gp.Model("Spanner Attack", env=env)

        # Initialize Variables
        d = m.addVars(all_path_edges, vtype=GRB.CONTINUOUS, name="d", lb=0, ub=local_budget) # Perturbations for each edge. Vtype can be changed to GRB.INTEGER if desired

        # The perturbed distance must be more than k times the original distance on every path
        path_constraints = m.addConstrs((gp.quicksum((G.edges[i, j]["weight"]+d[i,j]) for i,j in zip(path[:-1], path[1:])) >= goal for path in paths), name="attack") 
        # The total of our perturbations must be under the global budget
        global_budget = m.addConstr(d.sum() <= global_budget, name="global_budget") 

        # Minimize the perturbation
        m.setObjective(d.sum(), sense=GRB.MINIMIZE) 

        if write_model: m.write("final model.lp")

        # Solve
        m.optimize()

        result_dict["LP Status"] = m.status

        if m.status == GRB.OPTIMAL:
            result_dict["Perturbation Dict"] = { k : v.X for k,v in d.items() if v.X != 0}
            result_dict["Total Perturbations"] = m.objVal
            result_dict["Global Budget Slack"] = global_budget.Slack
        elif m.status == GRB.INFEASIBLE:
            m.computeIIS()
            result_dict["IIS_paths"] = [path for path in paths if path_constraints[path].IISConstr]
            
    return result_dict