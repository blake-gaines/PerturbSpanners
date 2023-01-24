import networkx as nx
import numpy as np
import pandas as pd
from numpy import random as rand
import random
import os

class DataSets:
    base_path = os.path.dirname(os.path.realpath(__file__)) + '/'

    # ~ 5K edges 2K nodes
    CORA = {
        'path' : 'cora/cora.cites',
        'sep' : '\t',
        'name' : 'Cora',
        'cluster_size' : 7,
    }
    CORA_NPZ = {
        'path' : 'cora/cora.npz',
        'sep' : None,
        'name' : 'Cora_NPZ',
        'cluster_size' : 7,
    }
    POLBLOGS_NPZ = {
        'path' : 'polblogs/polblogs.npz',
        'sep' : None,
        'name' : 'Polblogs_NPZ',
        'cluster_size' : 2,
    }
    CITESEER_NPZ = {
        'path' : 'citeseer/citeseer.npz',
        'sep' : None,
        'name' : 'Citeseer_NPZ',
        'cluster_size' : 6,
    }
    # ~88K edges 4K nodes
    FACEBOOK = {
        'path' : 'facebook/facebook_combined.txt',
        'sep' : ' ',
        'name' : 'Facebook',
        'cluster_size' : 68,#-1,
    }

    # ~ 184K edges 33K nodes
    ENRON_EMAILS = {
        'path' : 'enron/email-Enron.txt',
        'sep' : '\t',
        'name' : 'Enron Emails',
        'cluster_size' : -1,
    }

    GRQC = {
        'path' : 'grqc/ca-GrQc.txt',
        'sep' : '\t',
        'name' : 'General Relativity & Quantum Cosmology collab network',
        'cluster_size' : -1,
    }

    # Nodes: 9,877 Edges: 51,971
    HEPTH = {
        'path' : 'HepTh/ca-HepTh.txt',
        'sep' : '\t',
        'name' : 'High Energy Physics - Theory collaboration network',
        'cluster_size' : 4,#-1,
    }

    # Nodes 7,624, Edges 27,806
    # https://snap.stanford.edu/data/feather-lastfm-social.html
    ASIA_LAST_FM = {
        'path' : 'lastfm_asia/lastfm_asia_edges.csv',
        'sep' : ',',
        'name' : 'LastFM Asia Social Network',
        'cluster_size' : 18,#-1,
        
    }

    # Nodes: 12,008, Edges: 237,010
    # http://snap.stanford.edu/data/ca-HepPh.html
    CA_HEPPH = {
        'path' : 'ca-HepPh/CA-HepPh.txt',
        'sep' : '\t',
        'name' : 'High Energy Physics - Phenomenology collaboration network',
        'cluster_size' : -1,
    }

    # Nodes: 58,228, Edges: 214,078
    # http://snap.stanford.edu/data/loc-Brightkite.html
    LOC_BRIGHTKITE = {
        'path' : 'loc-brightkite/Brightkite_edges.txt',
        'sep' : '\t',
        'name' : 'Brightkite',
        'cluster_size' : -1,
    }

    # Nodes: 334,863, Edges: 925,872
    # http://snap.stanford.edu/data/com-Amazon.html
    AMAZON_CO_PURCHASING = {
        'path' : 'amazon/com-amazon.ungraph.txt',
        'sep' : '\t',
        'name' : 'Amazon - Co-purchasing Network',
        'cluster_size' : -1,
    }

    # Nodes: 50,515, Edges: 819,306
    # http://snap.stanford.edu/data/gemsec-Facebook.html
    GEMSEC_FACEBOOK_VERIFIED_NETWORKS = {
        'path' : 'gemsec/artist_edges.csv',
        'sep' : ',',
        'name' : 'Facebook - Graph Embedding with Self Clustering',
        'cluster_size' : -1,
    }

    # Nodes: 317,080 Edges: 1,049,866
    # http://snap.stanford.edu/data/com-DBLP.html
    DBLP_COLLABORATION_NETWORK = {
        'path' : 'dblp/com-dblp.ungraph.txt',
        'sep' : '\t',
        'name' : 'DBLP - Collaboration Network',
        'cluster_size' : -1,
    }

    # returns an networkx graph object representing the dataset
    # if lcc is true, it returns only largest connected component
    @classmethod
    def get_undirected_networkx_graph(cls, dataset, lcc=True):
        path = cls.base_path + dataset['path']
        separator = dataset['sep']
        if (cls.npz_check(dataset)):
            edgelist = cls.npz_to_df(dataset, ['target', 'source'])
        else:
            edgelist = pd.read_csv(path, sep=separator, names=['target', 'source'], comment='#')
        G = nx.from_pandas_edgelist(edgelist)
        if lcc == True:
            gs = [G.subgraph(c) for c in nx.connected_components(G)]
            G = max(gs, key=len)
        return G
    
    @classmethod
    def get_df_lcc(cls, dataset, lcc=True):
        path = cls.base_path + dataset['path']
        separator = dataset['sep']
        if (cls.npz_check(dataset)):
            edgelist = cls.npz_to_df(dataset, ['target', 'source'])
        else:
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
        if (cls.npz_check(dataset)):
            return cls.npz_to_df(dataset)
        path = cls.base_path + dataset['path']
        separator = dataset['sep']
        df = pd.read_csv(path, sep=separator, header=None, comment='#')
        return df

    @classmethod
    def npz_check(cls, _dataset):
        if (_dataset['path'].split('.')[-1] == 'npz'):
            return True
        return False

    '''
    npz dict:
    idx_to_attr
    attr_indices
    attr_shape
    idx_to_node
    adj_shape
    adj_indptr
    adj_data
    labels
    attr_data
    adj_indices
    attr_indptr
    idx_to_class
    attr_text
    '''
    @classmethod
    def npz_to_df (cls, _dataset, _columns = [0, 1]):
        npz = np.load(cls.base_path + _dataset['path'], allow_pickle = True)
        data = []
        for x in range(0, npz['adj_shape'][1]):
            for y in npz['adj_indices'][npz['adj_indptr'][x]:npz['adj_indptr'][x + 1]]:
                data.append([x, y])
        return pd.DataFrame(data, columns = _columns)

def get_graph(config):
    if config.graph_name == 'Facebook':
        G = DataSets.get_directed_networkx_graph(dataset=DataSets.FACEBOOK, lcc=False)
    elif config.graph_name == 'Cora':
        G = DataSets.get_directed_networkx_graph(dataset=DataSets.CORA, lcc=False)
    elif config.graph_name == 'LastFM_Asia':
        G = DataSets.get_directed_networkx_graph(dataset=DataSets.ASIA_LAST_FM, lcc=False)
    elif config.graph_name == 'ca-HepTH':
        G = DataSets.get_directed_networkx_graph(dataset=DataSets.HEPTH, lcc=False)
    elif config.graph_name == 'Citeseer':
        G = DataSets.get_directed_networkx_graph(dataset=DataSets.CITESEER_NPZ, lcc=False)
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