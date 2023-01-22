import networkx as nx
import itertools
from utils import *
import copy
from math import isclose, ceil

class PathSelector:
    # General Path Selector class
    name = "Path Selector"

    def __iter__(self):
        return self

    def distance(self):
        raise NotImplementedError
    
    def initialize_generator(self):
        raise NotImplementedError

    def get_next(self, state):
        raise NotImplementedError

    def check_if_done(self, state):
        raise NotImplementedError

class SinglePairPathSelector(PathSelector):
    # Path selector for single pairs of nodes, by default equivalent to PATHATTACK
    name = "Shortest Path Selector"

    def __init__(self, c):
        self.c = c
        self.goal = c.k*self.distance(c.G)+c.epsilon
    
    def get_next(self, state):
        if self.c.top_k == 1:
            return [(tuple(nx.shortest_path(state.G_prime, self.c.source, self.c.target, weight="weight")), self.goal)]
        else:
            return [(tuple(path), self.goal) for path in itertools.islice(nx.shortest_simple_paths(state.G_prime, self.c.source, self.c.target, weight="weight"), self.c.top_k)]

    def check_if_done(self, state):
        current_distance = self.distance(state.G_prime)
        return current_distance >= self.goal or isclose(state.current_distance, self.goal)

    def distance(self, G):
        return nx.shortest_path_length(G, self.c.source, self.c.target, weight="weight")

class SetsPathSelector(SinglePairPathSelector):
    # Path selector for two sets of nodes
    name = "Set Path Selector"

    def get_set_config(self, c):
        # Create a new config with a new graph, with a ghost source/target connected to each set

        c.source = "s_ghost"
        c.target = "t_ghost"
        G = c.G.copy()
        G.add_node(c.source)
        G.add_node(c.target)
        for s in c.S:
            G.add_edge(c.source, s, weight=0)
        for t in c.T:
            G.add_edge(t, c.target, weight=0)
        c.G = G
        return c

    def __init__(self, c):
        # Reset the configuration and reinitialize
        new_c = self.get_set_config(c)
        super().__init__(new_c)

    def get_next(self, state):
        # Trim the ghost nodes from the paths
        return [(path[1:-1], goal) for path, goal in super().get_next(state)]

class MultiPairPathSelector(PathSelector):
    # Path selector for multiple pairs of nodes, cycles through per-pair selectors

    name = "Multi Pairs Selector"
    def __init__(self, c, path_selector_func=SinglePairPathSelector, path_selectors=None, selector_func_kwargs=dict()):

        self.c = c

        self.generator = None
        self.path_selectors = path_selectors

        # Per-pair selectors can either be passed in or created using the path_selector_func argument (default: SinglePairPathSelector)
        if path_selectors is None:
            self.path_selectors = []
            for source, target in c.pairs:
                c_copy = copy.copy(c)
                c_copy.source = source
                c_copy.target = target
                c_copy.top_k = ceil(c.top_k/len(c.pairs))
                self.path_selectors.append(path_selector_func(c_copy, **selector_func_kwargs))
                self.path_selectors[-1].name = f"{self.name}: from {source} to {target}"
        else:
            self.path_selectors = path_selectors
        
        self.goal = 0
        self.c.top_k *= len(c.pairs)

    def combine_generators(self, state):  
        # Combine the generators of the per-pair selectors into a single generator
        i = 0
        generators = self.path_selectors.copy()
        while len(generators)>0:
            i = i % len(generators)
            next_paths = self.path_selectors[i].get_next(state)
            # print("Next", next_paths)
            # print("Weights", [nx.path_weight(state.G_prime, path, weight="weight") for path in next_paths])
            next_paths = [path for path in next_paths if nx.path_weight(state.G_prime, path["path"], weight="weight") <= self.path_selectors[i].goal]
            if next_paths:
                yield next_paths
                i = i + 1
            else:
                generators.pop(i)

    def distance(self, G):
        # Return the minimum distance of the per-pair selectors, equal to the distance between the sets
        return min([path_selector.distance(G) for path_selector in self.path_selectors])

    def check_if_done(self, state):
        return all([path_selector.check_if_done(state) for path_selector in self.path_selectors])

    def get_next(self, state):
        next_paths = [selector.get_next(state)[0] for selector in self.path_selectors if not selector.check_if_done(state)]
        return next_paths
        # generator = self.combine_generators(state)
        # return list((tuple(path) for path_set in itertools.islice(generator, self.c.top_k) for path in path_set)) # TODO: Fix

##################################

class our_selector(SinglePairPathSelector):
    name = "Our Selector"
    def __init__(self, c):
        self.c = c
        self.generator = nx.shortest_simple_paths(c.G, self.c.source, self.c.target)

    def get_next(self, state):
        return list(map(tuple, itertools.islice(self.generator, self.c.top_k)))

class random_walk_selector(SinglePairPathSelector):
    name = "Random Walk Selector"

    def get_next(self, state):
        return list(map(tuple, itertools.islice(random_one_sided(self.c.G, self.c.source, self.c.target), self.c.top_k)))
