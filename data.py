import networkx as nx
import pickle as pkl
import numpy as np
import scipy as sp
import pandas as pd
from numpy import random as rand
import random

class DataSets:
    base_path = 'data/'
    CORA =     {'path': 'cora/cora.cites', 'sep': '\t', 'name': 'Cora'}  # ~ 5K edges 2K nodes
    FACEBOOK = {'path': 'facebook/facebook_combined.txt', 'sep': ' ', 'name': 'Facebook'} # ~88K edges 4K nodes
    HEPTH = {'path': 'HepTh/ca-HepTh.txt', 'sep': '\t', 'name': 'High Energy Physics - Theory collaboration network'}
    ROADNET = {'path': 'roadNet-PA/roadNet-PA.txt', 'sep': '\t', 'name': 'Pennsylvania road network'}

    # returns an networkx graph object representing the dataset
    # if lcc is true, it returns only largest connected component
    @classmethod    
    def get_undirected_networkx_graph(cls, dataset, lcc=True):
        path = cls.base_path + dataset['path']
        separator = dataset['sep']
        edgelist = pd.read_csv(path, sep=separator, names=['target', 'source'], comment='#')
        G = nx.from_pandas_edgelist(edgelist)
        if lcc == True:
            gs = [G.subgraph(c) for c in nx.connected_components(G)]
            G = max(gs, key=len)
        return G
    
    @classmethod   
    def get_directed_networkx_graph(cls, dataset, lcc=True):
        path = cls.base_path + dataset['path']
        separator = dataset['sep']
        edgelist = pd.read_csv(path, sep=separator, names=['target', 'source'], comment='#')
        G = nx.from_pandas_edgelist(edgelist,source='source', target='target', create_using=nx.DiGraph())
        if lcc == True:
            gs = [G.subgraph(c) for c in nx.connected_components(G)]
            G = max(gs, key=len)
        return G

    
    @classmethod
    def get_df_lcc(cls, dataset, lcc=True):
        path = cls.base_path + dataset['path']
        separator = dataset['sep']
        edgelist = pd.read_csv(path, sep=separator, names=['target', 'source'], comment='#')
        G = nx.from_pandas_edgelist(edgelist)
        if lcc == True:
            gs = [G.subgraph(c) for c in nx.connected_components(G)]
            G = max(gs, key=len)
        df = nx.to_pandas_edgelist(G)
        df.columns = [0,1]
        return df
    
    @classmethod
    def get_df(cls, dataset):
        path = cls.base_path + dataset['path']
        separator = dataset['sep']
        df = pd.read_csv(path, sep=separator, header=None, comment='#')
        return df

# From https://github.com/bamille1/PATHATTACK/blob/main/data.py
# add_weights: add weights to an unweighted networkx graph via a specified method
#    Inputs:    G - an unweighted, undirected networkx graph
#               weights - a string denoting the distribution from which to
#                   draw weights (Poisson, Uniform, or Equal)
#   Outputs:    None (G is modified directly)
def add_weights(G, weights):

    '''
        Add weights to a graph
        Input : 
            G       : nx Graph object - Input graph
            weights : String - Poisson | Uniform
    '''

    nWeights = G.number_of_edges()
    
    if weights == 'Poisson':
        # draw weights from a Poisson distribution
        w = 1+rand.poisson(size=(nWeights))
    elif weights == 'Uniform':
        # draw weights from a uniform distribution
        # w = 1+rand.randint(41, size=(nWeights))
        w = 1+rand.uniform(size=(nWeights))
    else:
        # make all weights equal (1)
        w = np.ones((nWeights))

    # assign each weight to an edge
    ctr = 0
    for e in G.edges:
        G.edges[e]['weight'] = w[ctr]
        ctr += 1

def get_nodes(G, config):
    nodes = list(G.nodes())

    if len(nodes) > 500000:
        source = random.choice(nodes)
        nodes = list(nx.single_source_dijkstra(G, source, cutoff=100, weight="weight")[0].keys())


    if config.experiment_type == "Single":
        experiments = set()
        while len(experiments) < config.n_experiments:
            a,b = random.choices(nodes,k=2)
            if nx.has_path(G, a, b) and nx.shortest_path_length(G, a, b, weight="weight") > config.min_path_length:
                experiments.add((a,b))

    elif config.experiment_type == "Multiple Pairs":
        experiments = set()
        while len(experiments) < config.n_experiments:
            experiment = set()
            while len(experiment) < config.n_nodes_per_experiment // 2:
                a,b = random.choices(nodes,k=2)
                if nx.has_path(G, a, b) and nx.shortest_path_length(G, a, b, weight="weight") > config.min_path_length:
                    experiment.add((a,b))
            experiments.add(tuple(experiment))
    elif config.experiment_type == "Sets":
        experiments = set()
        while len(experiments) < config.n_experiments:
            S, T = [], []
            while len(S) < config.n_nodes_per_experiment // 2:
                a,b = random.choices(nodes,k=2)
                if nx.has_path(G, a, b) and nx.shortest_path_length(G, a, b, weight="weight") > config.min_path_length:
                    S.append(a), T.append(b)
            experiments.add((tuple(S),tuple(T)))
    else:
        print(config.experiment_type)
        raise NotImplementedError

    return experiments

def get_graph(config):
    if config.graph_name == 'Facebook':
        G = DataSets.get_directed_networkx_graph(dataset=DataSets.FACEBOOK, lcc=False)
    elif config.graph_name == 'RoadNet':
        G = DataSets.get_directed_networkx_graph(dataset=DataSets.ROADNET, lcc=False)
    elif config.graph_name == 'er':
        G = nx.erdos_renyi_graph(10000, .001, directed=True)
    elif config.graph_name == 'ba':
        G = nx.barabasi_albert_graph(10000, 5)
    elif config.graph_name == 'ws':
        G = nx.watts_strogatz_graph(10000, 10, 0.01)
    else:
        raise ValueError(f"Unknown Graph {config.graph_name}")
    add_weights(G, config.weights)
    return G