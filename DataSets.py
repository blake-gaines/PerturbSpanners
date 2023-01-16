import pandas as pd
import networkx as nx

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
