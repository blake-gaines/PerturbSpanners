import networkx as nx
import random
from collections import deque
from collections import defaultdict

# Get subgraph containing only paths with length less than goal
# Assumes positive edge weights
# Returns read-only view of original graph
def get_P_graph(G, goal, dist_from_s, dist_to_t):
    selected_edges = []
    for a,b in G.edges():
        if a not in dist_from_s or b not in dist_to_t:
            continue
        s_to_a = dist_from_s[a]
        a_to_b = G.edges[a,b]["weight"]
        b_to_t = dist_to_t[b]

        shortest_path_length = s_to_a + a_to_b + b_to_t

        if shortest_path_length <= goal:
            selected_edges.append((a,b))

    return G.edge_subgraph(selected_edges)

def restrict_graph(G, source, target, goal, weight="weight"):
    dist_from_s, _ = nx.single_source_dijkstra(G, source, cutoff=goal, weight=weight)
    dist_to_t, _ = nx.single_source_dijkstra(G.reverse(), target, cutoff=goal, weight=weight)

    P_Graph = get_P_graph(G, goal, dist_from_s, dist_to_t)

    return dist_from_s, dist_to_t, P_Graph

# Select random path between s and t from the P graph through (a,b) by randomly walking backwards and forwards
def select_random_path(G, source, target, node_pair, dist_from_s, dist_to_t, max_path_length, counts):
    a, b = node_pair
    path = [b, a]

    # walk back toward the source, taking only valid steps
    cost=0
    while path[-1] != source:
        neighbors =[node for node in G.predecessors(path[-1]) if dist_from_s[node]+G.edges[node, path[-1]]["weight"]+cost <= max_path_length-dist_to_t[b]]
        # weights = [dist_from_s[node]**0.5 if node != source else 1 for node in neighbors]
        # weight_sum = max(sum(weights), 1)
        # weights = [weight/weight_sum for weight in weights]
        weights = [1]*len(neighbors)

        choice = random.choices(neighbors, weights=weights, k=1)[0]
        counts[choice] = counts[choice] + 1
        cost += G.edges[choice, path[-1]]["weight"]
        path.append(choice)
    path.reverse()

    # walk forward toward the target, taking only valid steps
    while path[-1] != target:
        neighbors = [node for node in G.successors(path[-1]) if cost+G.edges[path[-1], node]["weight"]+dist_to_t[node] <= max_path_length]
        
        # weights = [dist_to_t[node]**0.5 if node != target else 1 for node in neighbors]
        # weight_sum = max(sum(weights), 1)
        # weights = [weight/weight_sum for weight in weights]
        weights = [1]*len(neighbors)

        choice = random.choices(neighbors, weights=weights, k=1)[0]
        counts[choice] = counts[choice] + 1
        choice = random.choice(neighbors)
        cost += G.edges[path[-1], choice]["weight"]
        path.append(choice)

    return path

def random_paths(G, source, target, goal, weight="weight"):
    dist_from_s, dist_to_t, P_Graph = restrict_graph(G, source, target, goal, weight=weight)

    print('P_Graph nodes: {} \nP_Graph edges: {}\n'.format(P_Graph.number_of_nodes(), P_Graph.number_of_edges()))

    all_edges = set(P_Graph.edges())

    counts = defaultdict(lambda: 1)

    while all_edges:
        edge_choice = random.choice(list(all_edges))
        path = select_random_path(P_Graph, source, target, edge_choice, dist_from_s, dist_to_t, goal, counts)
        all_edges = all_edges.difference(set(zip(path[:-1], path[1:])))
        yield path

def shortest_through_edge(G, source, target, edge_choice, weight="weight"):
    a, b = edge_choice
    path = nx.shortest_path(G, source, a, weight=weight)
    path.extend(nx.shortest_path(G, b, target, weight=weight))
    return path

def random_shortest_paths(G, source, target, goal, weight="weight"):
    _, _, P_Graph = restrict_graph(G, source, target, goal, weight=weight)
    P_Graph = P_Graph.copy()

    print('P_Graph nodes: {} \nP_Graph edges: {}\n'.format(P_Graph.number_of_nodes(), P_Graph.number_of_edges()))

    centrality_dict = nx.edge_betweenness_centrality(P_Graph, weight=weight)
    edge_list = list(centrality_dict.keys())
    edge_list.sort(key=centrality_dict.get, reverse=True)

    for edge_choice in edge_list:
        # if edge_choice not in P_Graph.edges():
        #     continue
        try:
            path = shortest_through_edge(P_Graph, source, target, edge_choice, weight=weight)
        except:
            continue
        P_Graph.remove_edge(*edge_choice)
        # P_Graph.remove_edges_from(zip(path[:-1], path[1:]))
        yield path

def random_one_sided(G, source, target, goal, weight="weight"):
    dist_to_t, _ = nx.single_source_dijkstra(G.reverse(), target, cutoff=goal, weight=weight)

    counts = defaultdict(lambda: 1)

    while True:
        path = [source]
        cost = 0
        while path[-1] != target:
            neighbors = [node for node in G.successors(path[-1]) if node in dist_to_t and cost+G.edges[path[-1], node]["weight"]+dist_to_t[node] <= goal]

            weights = [counts[node]**-2 for node in neighbors]
            weight_sum = sum(weights)
            weights = [weight/weight_sum for weight in weights]

            choice = random.choices(neighbors, weights=weights, k=1)[0]

            counts[choice] = counts[choice] + 1
            cost += G.edges[path[-1], choice]["weight"]
            path.append(choice)
        yield path