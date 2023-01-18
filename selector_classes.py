import networkx as nx
import itertools
from selector_functions import *

class PathSelector:
    def __init__(self, name, update_every_iteration=False, top_k=1):
        self.name = name
        self.update_every_iteration = update_every_iteration
        self.top_k = top_k
        self.generator = None

    def __iter__(self):
        self.raise_if_not_initialized()
        return self

    def distance(self):
        raise NotImplementedError
    
    def initialize_generator(self):
        raise NotImplementedError

    def get_next(self, filter_func=lambda x: True, **kwargs):
        self.raise_if_not_initialized()
        return list(map(tuple, itertools.islice(filter(filter_func, self.generator), self.top_k)))

    def __repr__(self):
        return self.name

    def raise_if_not_initialized(self):
        if not self.generator:
            raise Exception("Generator not initialized. Call initialize_generator first.")

class SinglePairPathSelector(PathSelector):
    def __init__(self, G, source, target, name="Shortest Path Selector", generator_function=nx.shortest_simple_paths, update_every_iteration=True, top_k=1, **generator_function_kwargs):
        super().__init__(name, update_every_iteration, top_k)
        self.generator_function = generator_function
        self.generator_function_kwargs = generator_function_kwargs

        self.source = source
        self.target = target
        self.update_graph(G)
    
    def update_graph(self, new_graph):
        self.generator = self.generator_function(new_graph, self.source, self.target, **self.generator_function_kwargs)

    def distance(self, G):
        return nx.shortest_path_length(G, self.source, self.target, weight="weight")

class SetsPathSelector(SinglePairPathSelector):
        def __init__(self, G, S, T, name="Shortest Path Selector", generator_function=nx.shortest_simple_paths, update_every_iteration=True, top_k=1, **generator_function_kwargs):
            super().__init__(name, update_every_iteration, top_k=top_k)
            self.generator_function = generator_function
            self.generator_function_kwargs = generator_function_kwargs
            source = "s_ghost"
            target = "t_ghost"
            G.add_node(source)
            G.add_node(target)
            for s in S:
                G.add_edge(source, s, weight=0)
            for t in T:
                G.add_edge(t, target, weight=0)
            self.source = source
            self.target = target
            self.update_graph(G)

        def get_next(self, **kwargs):
            return [path[1:-1] for path in super(self).get_next(**kwargs)]

class MultiPairPathSelector(PathSelector):
    def __init__(self, name, G, pairs, path_selector_func=None, path_selectors=None, update_every_iteration=True, top_k=1, **selector_func_kwargs):
        super().__init__(name, update_every_iteration, top_k)

        self.generator = None
        self.path_selectors = path_selectors
        self.generator_function_kwargs = None

        if path_selector_func is not None:
            self.path_selectors = []
            for source, target in pairs:
                self.path_selectors.append(self.path_selector_func(name=f"{self.name}: from {source} to {target}", **selector_func_kwargs))
                self.path_selectors[-1].initialize_generator(G, source, target)
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

    def get_next(self, **kwargs):
        self.raise_if_not_initialized()
        return (path for path_set in itertools.islice(self.generator, self.top_k) for path in path_set)

class edge_centrality_selector(SinglePairPathSelector):
    def __init__(self, G, source, target, goal, name, weight="weight", top_k=1):
        super().__init__(G, source, target, name=name, update_every_iteration=False, top_k=top_k)
        _, _, P_Graph = restrict_graph(G, source, target, goal, weight=weight)
        self.P_Graph = P_Graph.copy()

        self.top_k = top_k
        centrality_dict = nx.edge_betweenness_centrality(P_Graph, weight=weight)
        self.edge_list = list(centrality_dict.keys())
        self.edge_list.sort(key=centrality_dict.get, reverse=True)
        self.i = 0

    def get_next(self, G, current_dist):
        paths = []
        for _ in range(self.top_k):
            edge_choice = self.edge_list[self.i]
            path = shortest_through_edge(G, self.source, self.target, edge_choice, weight="weight")
            paths.append(tuple(path))
            self.i = (self.i + 1) % len(self.edge_list)
        return paths