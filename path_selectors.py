import networkx as nx
import itertools

class PathSelector:
    def __init__(self, name, goal, update_every_iteration=False, top_k=1):
        self.name = name
        self.update_every_iteration = update_every_iteration
        self.top_k = top_k
        self.generator = None
        self.goal = goal

    def __iter__(self):
        self.raise_if_not_initialized()
        return self

    def distance(self):
        raise NotImplementedError
    
    def initialize_generator(self):
        raise NotImplementedError

    def __next__(self, filter_func=lambda x: True):
        self.raise_if_not_initialized()
        return map(tuple, itertools.islice(filter(filter_func, self.generator), self.top_k))

    def __repr__(self):
        return self.name

    def raise_if_not_initialized(self):
        if not self.generator:
            raise Exception("Generator not initialized. Call initialize_generator first.")

class SinglePairPathSelector:
    def __init__(self, name="Shortest Path Selector", generator_function=nx.shortest_simple_paths, update_every_iteration=True, top_k=1, **generator_function_kwargs):
        super(self).__init__(name, update_every_iteration, top_k=top_k)
        self.generator_function = generator_function
        self.generator = None
        self.source = None
        self.target = None
        self.generator_function_kwargs = generator_function_kwargs

    def initialize_generator(self, G, source, target):
        self.source = source
        self.target = target
        self.update_graph(G)
    
    def update_graph(self, new_graph):
        self.generator = self.generator_function(new_graph, self.source, self.target, **self.generator_function_kwargs)

    def distance(self, G):
        return nx.shortest_path_length(G, self.source, self.target, weight="weight")

class SetsPathSelector(SinglePairPathSelector):
        def initialize_generator(self, G, S, T):
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

        def __next__(self):
            return [path[1:-1] for path in super(self).__next__()]

class MultiPairPathSelector:
    def __init__(self, name, path_selectors=None, update_every_iteration=False, top_k=1):
        super(self).__init__(name, update_every_iteration, top_k)

        self.generator = None
        self.path_selectors = path_selectors
        self.generator_function_kwargs = None

    def initialize_generator(self, G, pairs, path_selector_func, **selector_kwargs):
        self.path_selectors = []
        for source, target in pairs:
            self.path_selectors.append(self.path_selector_func(name=f"{self.name}: from {source} to {target}", **self.selector_kwargs))
            self.path_selectors[-1].initialize_generator(G, source, target)
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

    def __next__(self):
        self.raise_if_not_initialized()
        return (path for path_set in itertools.islice(self.generator, self.top_k) for path in path_set)