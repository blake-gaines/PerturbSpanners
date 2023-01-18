import networkx as nx
import itertools
from selector_functions import *

class PathSelector:
    name = "Path Selector"
    update_every_iteration = True

    def __init__(self, c, update_every_iteration=False, filter_func=lambda x: True, **kwargs):
        self.top_k = c.top_k
        self.generator = None
        self.filter_func = filter_func

    def __iter__(self):
        self.raise_if_not_initialized()
        return self

    def distance(self):
        raise NotImplementedError
    
    def initialize_generator(self):
        raise NotImplementedError

    def get_next(self, state):
        return list(map(tuple, itertools.islice(filter(self.filter_func, self.generator), self.top_k)))

    def __repr__(self):
        return self.name

    def raise_if_not_initialized(self):
        if not self.generator:
            raise Exception("Generator not initialized. Call initialize_generator first.")

class SinglePairPathSelector(PathSelector):
    name = "Shortest Path Selector"
    generator_function = nx.shortest_simple_paths
    generator_function_kwargs = {"weight": "weight"}

    def __init__(self, c, **generator_function_kwargs):
        super().__init__(c)
        self.source = c.source
        self.target = c.target
        self.update_graph(c.G)
    
    def update_graph(self, new_graph):
        self.generator = self.generator_function(new_graph, self.source, self.target)#, **self.generator_function_kwargs)

    def distance(self, G):
        return nx.shortest_path_length(G, self.source, self.target, weight="weight")

def get_set_config(c):
    c.source = "s_ghost"
    c.target = "t_ghost"
    G = c.G.copy()
    G.add_node(c.source)
    G.add_node(c.target)
    for s in c.S:
        c.G.add_edge(c.source, s, weight=0)
    for t in c.T:
        c.G.add_edge(t, c.target, weight=0)
    return c

class SetsPathSelector(SinglePairPathSelector):
    name = "Set Path Selector"

    def __init__(self, c):
        new_c = get_set_config(c)
        super().__init__(get_set_config(new_c))

    def get_next(self, state):
        return [path[1:-1] for path in super(self).get_next(state)]

class MultiPairPathSelector(PathSelector):
    name = "Multi Pairs Selector"
    def __init__(self, c, path_selector_func=None, path_selectors=None, update_every_iteration=True, top_k=1, **selector_func_kwargs):
        super().__init__(c, update_every_iteration)

        self.generator = None
        self.path_selectors = path_selectors

        if path_selector_func is not None:
            self.path_selectors = []
            for source, target in c.pairs:
                self.path_selectors.append(self.path_selector_func(name=f"{self.name}: from {source} to {target}", **selector_func_kwargs))
                self.path_selectors[-1].initialize_generator(c.G, source, target)
        elif path_selectors is None:
            raise Exception("Either path_selector_func or path_selectors must be provided.")
        self.generator = self.combine_generators(self.path_selectors)

    def combine_generators(self, generators):  
        i = 0
        while generators:
            next_paths = next(generators[i])
            if next_paths:
                yield next_paths
            else:
                generators.pop(i)
            i = (i + 1) % len(generators)
    
    def update_graph(self, new_graph):
        for selector in self.path_selectors:
            selector.update_graph(new_graph)
        self.generator = self.combine_generators(self.path_selectors)

    def distance(self, G):
        return min([selector.distance(G) for selector in self.path_selectors])

    def get_next(self, state):
        self.raise_if_not_initialized()
        return (path for path_set in itertools.islice(self.generator, self.top_k) for path in path_set)

class their_selector(SinglePairPathSelector):
    name = "Their Selector"

class our_selector(SinglePairPathSelector):
    name = "Our Selector"
    update_every_iteration = False

class edge_centrality_selector(SinglePairPathSelector):
    name = "Edge Centrality Selector"
    update_every_iteration=False

    def __init__(self, c, weight="weight"):
        super().__init__(c)
        _, _, P_Graph = restrict_graph(c.G, c.source, c.target, c.goal, weight=weight)
        self.P_Graph = P_Graph.copy()

        self.top_k = c.top_k
        centrality_dict = nx.edge_betweenness_centrality(P_Graph, weight=weight)
        self.edge_list = list(centrality_dict.keys())
        self.edge_list.sort(key=centrality_dict.get, reverse=True)
        self.i = 0

    def get_next(self, state):
        paths = []
        for _ in range(self.top_k):
            edge_choice = self.edge_list[self.i]
            path = shortest_through_edge(state.G_prime, self.source, self.target, edge_choice, weight="weight")
            paths.append(tuple(path))
            self.i = (self.i + 1) % len(self.edge_list)
        return paths


