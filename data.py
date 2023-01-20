"""
Script to create input data for experiments outlined in “PATHATTACK: Attacking 
shortest paths in complex networks” by Benjamin A. Miller, Zohair Shafi, 
Wheeler Ruml, Yevgeniy Vorobeychik, Tina Eliassi-Rad, and Scott Alfeld at 
ECML/PKDD 2021.

This material is based upon work supported by the United States Air Force under
Air  Force  Contract  No.  FA8702-15-D-0001  and  the  Combat  Capabilities  
Development Command Army Research Laboratory (under Cooperative Agreement Number
W911NF-13-2-0045).  Any  opinions,  findings,  conclusions  or  recommendations
expressed in this material are those of the authors and do not necessarily 
reflect theviews of the United States Air Force or Army Research Laboratory.

Copyright (C) 2021
Benjamin A. Miller [1], Zohair Shafi [1], Wheeler Ruml [2],
Yevgeniy Vorobeychik [3],Tina Eliassi-Rad [1], and Scott Alfeld [4]

[1] Northeastern Univeristy
[2] University of New Hampshire
[3] Washington University in St. Louis
[4] Amherst College

The software is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work
"""
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


# from algorithms import *

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
        w = 1+rand.poisson(20, (nWeights))
    elif weights == 'Uniform':
        # draw weights from a uniform distribution
        w = 1+rand.randint(41, size=(nWeights))
    else:
        # make all weights equal (1)
        w = np.ones((nWeights))

    # assign each weight to an edge
    ctr = 0
    for e in G.edges:
        G.edges[e]['weight'] = w[ctr]
        ctr += 1

def get_input_data(graph_name, weights="Equal", n_trials=1, experiment_type="Single", num_nodes=10):

    assert(weights in ['Poisson', 'Uniform', 'Equal'])
    # print(weights, ' weights')
    if graph_name == 'er':    # Erdos-Renyi graph
        G = []
        for ii in range(n_trials):
            Gtemp = nx.erdos_renyi_graph(16000, .00125)
            add_weights(Gtemp, weights)
            G.append(Gtemp)
    elif graph_name == 'ba':   # Barabasi-Albert graph
        G = []
        for ii in range(n_trials):
            Gtemp = nx.barabasi_albert_graph(16000, 10)
            add_weights(Gtemp, weights)
            G.append(Gtemp)
    elif graph_name == 'ws':   # Watts-Strogatz graph
        G = []
        for ii in range(n_trials):
            Gtemp = nx.watts_strogatz_graph(n=16000, k=20, p=0.02)
            add_weights(Gtemp, weights)
            G.append(Gtemp)
    # elif graph_name == 'kron':   # Kronecker graph
    #     G = []
    #     for ii in range(n_trials):
    #         Gtemp = kronecker_graph(16000, 0.00125)
    #         for n in Gtemp.nodes():
    #             if Gtemp.has_edge(n, n):
    #                 Gtemp.remove_edge(n, n)
    #         add_weights(Gtemp, weights)
    #         G.append(Gtemp)
    elif graph_name == 'lattice':   # lattice graph
        G = []
        Gtemp = nx.Graph(nx.adjacency_matrix(nx.grid_2d_graph(285, 285)))
        for ii in range(n_trials):
            add_weights(Gtemp, weights)
            G.append(Gtemp.copy())
    elif graph_name == 'complete':   # Complete graph
        G = []
        Gtemp = nx.complete_graph(565)
        for ii in range(n_trials):
            add_weights(Gtemp, weights)
            G.append(Gtemp.copy())
    elif graph_name == 'wiki':   # Wikispeedia Graph
        G = []
        with open('./wikispeediaGraph.pkl', 'rb') as f:
            Gtemp = pkl.load(f)
        for ii in range(n_trials):
            add_weights(Gtemp, weights)
            G.append(Gtemp.copy())
    elif graph_name == 'as':   # Autonomous System Graph
        G = []
        with open('./autonomousSystemGraph.pkl', 'rb') as f:
            Gtemp = pkl.load(f)
        for ii in range(n_trials):
            add_weights(Gtemp, weights)
            G.append(Gtemp.copy())
    elif graph_name == 'PARoad':   # Pennsylvania Road Network
        G = []
        with open('./roadNet-PA.pkl', 'rb') as f:
            Gtemp = pkl.load(f)
        for ii in range(n_trials):
            add_weights(Gtemp, weights)
            G.append(Gtemp.copy())
    elif graph_name == 'Facebook':
        Gtemp = DataSets.get_directed_networkx_graph(dataset=DataSets.FACEBOOK, lcc=False)
        input_dict = dict()
        nodes = []
        while len(nodes) < (num_nodes if experiment_type != "Single" else 1):
            print(len(nodes), end='\r', flush=True)
            a,b = random.choices(list(Gtemp.nodes()),k=2)
            if nx.has_path(Gtemp, a, b):
                nodes.append((a,b))
        if experiment_type == "Single":
            nodes = (nodes[0][0], nodes[0][1])
        if experiment_type == "Sets":
            nodes = [[x[0] for x in nodes], [x[1] for x in nodes]]
        add_weights(Gtemp, weights)
        input_dict['G'] = Gtemp
        input_dict['nodes'] = nodes
        # print("input_dict: ", input_dict)
        return input_dict

        
    else:
        print('invalid graph name')
        raise
    # print('done making graphs', flush=True)


    # file name to save input data
    # if inputDir: filename = inputDir+'/'+graph_name+'_'+weights+'Weights_part_'+str(seed_plus)+'_'+str(n_trials)+'trials.pkl'

    input_list = []
    if graph_name not in ['er', 'ba', 'kron', 'ws']:
        lcc = list(max(nx.connected_components(G[0]), key=len))
    for ii in range(n_trials):
        # print('   graph ', ii)
        if graph_name in ['er', 'ba', 'kron', 'ws']:
            lcc = list(max(nx.connected_components(G[ii]), key=len))

        # for lattice-like networks
        if graph_name in ['lattice', 'PARoad']:
            # random node from largest connected component
            s = rand.choice(lcc)
            n = len(G[ii])
            A = nx.adjacency_matrix(G[ii])
            A += sp.diags(np.ones(n)) #add diagonal to adjacency matrix

            # initialize vector as an indicator for selected node
            v = np.zeros((n))
            v[s] = 1
            # spread over 49 hops
            for _ in range(49):
                v = A @ v
                
            # find nodes exactly 50 hops away
            neighbors = np.nonzero(v)[0]
            v = A @ v
            nHopNeighbors = np.setdiff1d(np.nonzero(v)[0], neighbors)
            # target is one of these nodes (randomly selected)
            t = rand.choice(nHopNeighbors)

            # only consider paths within the induced subgraph of nodes within
            # 60 hops of the source
            # for _ in range(10):
            #     v = A @ v
            # sg = np.nonzero(v)[0]
            # Gpath = nx.subgraph(G[ii], sg)
        else:  # other (non-lattice-like) networks)
            # choose two nodes at random from the largest connected component
            terminals = rand.choice(lcc, size=(2 if experiment_type == "Single" else num_nodes), replace=False)
            if experiment_type == "Single":
                nodes = terminals
            if experiment_type == "Multiple Pairs":
                nodes = list(zip(terminals[:num_nodes//2], terminals[num_nodes//2:]))
            elif experiment_type == "Sets":
                nodes = (set(terminals[:num_nodes//2]), set(terminals[num_nodes//2:]))
            terminals = rand.choice(lcc, size=2, replace=False)
            s = terminals[0]
            t = terminals[1]
            # Gpath = G[ii] # consider paths over the entire graph


        # save input information: Graph (with weights), source and 
        # target
        input_dict = dict()
        input_dict['G'] = G[ii]
        input_dict['nodes'] = nodes
        # input_list.append(input_dict)

    # # write file
    # if inputDir:
    #     with open(filename, 'wb') as f:
    #         pkl.dump(input_list, f)

    return input_dict

