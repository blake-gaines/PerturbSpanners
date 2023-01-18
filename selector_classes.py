import networkx as nx
import itertools
from selector_functions import *
import copy

class PathSelector:
    name = "Path Selector"
    update_every_iteration = True

    def __init__(self, c, filter_func=None):
        self.top_k = c.top_k
        self.filter_func = filter_func

    def __iter__(self):
        # self.raise_if_not_initialized()
        return self

    def distance(self):
        raise NotImplementedError
    
    def initialize_generator(self):
        raise NotImplementedError

    def get_next(self, state):
        # self.raise_if_not_initialized()
        return list(map(tuple, itertools.islice(filter(self.filter_func, self.generator), self.top_k)))

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    # def raise_if_not_initialized(self):
    #     if not self.generator:
    #         raise Exception("Generator not initialized. Call initialize_generator first.")

class SinglePairPathSelector(PathSelector):
    name = "Shortest Path Selector"
    generator_function_kwargs = {"weight": "weight"}

    def __init__(self, c, **generator_function_kwargs):
        super().__init__(c)
        self.source = c.source
        self.target = c.target
        self.update_graph(c.G)

    def generator_function(self, G, source, target, **kwargs):
        return nx.shortest_simple_paths(G, source, target, **kwargs)
    
    def update_graph(self, new_graph):
        self.generator = self.generator_function(new_graph, self.source, self.target, **self.generator_function_kwargs)

    def distance(self, G):
        return nx.shortest_path_length(G, self.source, self.target, weight="weight")

class SetsPathSelector(SinglePairPathSelector):
    name = "Set Path Selector"

    def get_set_config(self, c):
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

    def __init__(self, c):
        new_c = self.get_set_config(c)
        super().__init__(new_c)

    def get_next(self, state):
        return [path[1:-1] for path in super().get_next(state)]

class MultiPairPathSelector(PathSelector):
    name = "Multi Pairs Selector"
    def __init__(self, c, path_selector_func=SinglePairPathSelector, path_selectors=None, selector_func_kwargs=dict()):
        super().__init__(c)

        self.generator = None
        self.path_selectors = path_selectors

        if path_selectors is None:
            self.path_selectors = []
            for source, target in c.pairs:
                c_copy = copy.copy(c)
                c_copy.source = source
                c_copy.target = target
                self.path_selectors.append(path_selector_func(c_copy, **selector_func_kwargs))
                self.path_selectors[-1].name = f"{self.name}: from {source} to {target}"
        else:
            self.path_selectors = path_selectors

    def combine_generators(self, state):  
        i = 0
        generators = self.path_selectors.copy()
        while generators:
            next_paths = self.path_selectors[i].get_next(state)
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
        # self.raise_if_not_initialized()
        generator = self.combine_generators(state)
        # ah = list((path for path_set in itertools.islice(generator, self.top_k) for path in path_set))
        # print("AHA", ah)
        # return ah
        # print([path for path_set in ah for path in path_set])
        # print("AHA", list((path for path_set in itertools.islice(self.generator, self.top_k*len(self.path_selectors)) for path in path_set)))
        return list((tuple(path) for path_set in itertools.islice(generator, self.top_k) for path in path_set))

##################################

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
        self.base = 0
        self.prev_distance = None

    def get_next(self, state):
        paths = []
        
        self.i = 0
        if state.current_distance == self.prev_distance:
            self.base += 1

        for j in range(len(self.edge_list)-self.i):
            edge_choice = self.edge_list[self.base + self.i]
            # if edge_choice in state.all_path_edges:
            #     if j-1 == self.base:
            #         self.base += 1
            #     continue
            path = tuple(shortest_through_edge(state.G_prime, self.source, self.target, edge_choice, weight="weight"))
            if path not in state.paths:
                paths.append(tuple(path))
                if len(paths) == self.top_k:
                    break
            self.i = (self.i + 1) % (len(self.edge_list) - self.base)
        return paths


