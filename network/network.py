# -*- coding: utf-8 -*-
import os
import re
import pickle
import math
import networkx as nx
from networkx import Graph
import pandas as pd
import numpy as np
import scipy as sp
from sklearn import preprocessing
from base.hash_json import hashlib_json


class BiGraph(Graph):
    # the networkx package version is 2.0
    def __init__(self):
        super().__init__()
        self.userLE = None
        self.itemLE = None
        self.userlist = []
        self.itemlist = []
        self.user_size = 0
        self.item_size = 0
        self.node_size = 0
        self.neighbors_dict = dict()
        self.neighbors_of_neighbors_dict = dict()

    def filter_nodes(self, data, value):
        nodeview = list(self.nodes(data=data))
        filtered_nodeview = filter(lambda x: x[1] == value, nodeview)
        nodes = list(map(lambda x: x[0], filtered_nodeview))
        return nodes

    def node2id(self):
        self.userLE = preprocessing.LabelEncoder()
        self.itemLE = preprocessing.LabelEncoder()
        self.userLE.fit(self.userlist)
        self.itemLE.fit(self.itemlist)

    @property
    def nodelist(self):
        userlist = self.filter_nodes('label', 'user')
        itemlist = self.filter_nodes('label', 'item')
        self.userlist = sorted(userlist)
        self.itemlist = sorted(itemlist)
        nodelist = self.userlist + self.itemlist
        self.user_size = len(self.userlist)
        self.item_size = len(self.itemlist)
        self.node_size = len(nodelist)
        return nodelist

    @property
    def adj_matrix(self, nodelist=None):
        if not nodelist:
            nodelist = self.nodelist
        adj_matrix = nx.adjacency_matrix(self, nodelist=nodelist)
        return adj_matrix

    def dict_of_neighbors(self):
        """
    	neighbors_dict: is the direct neighbors of every node
    	neighbors_of neighbors_dict: is the indirect neighbors of every node
    	"""
        neighbors_dict = dict()  # its value's type is list
        for node in self.nodes():
            neighbors_dict[node] = list(self.neighbors(node))  # list of neighbors
        neighbors_of_neighbors_dict = dict()  # its value's type is list
        for node in self.nodes():
            indirect_neighbors = set(self.nodes())  # init with all nodes
            for neig_node in neighbors_dict[node]:
                indirect_neighbors.intersection_update(neighbors_dict[neig_node])  # update the indirect neighbors
            neighbors_of_neighbors_dict[node] = list(indirect_neighbors)  # transfer to list
        return neighbors_dict, neighbors_of_neighbors_dict

    def get_neighbors(self, graph_id):
        # hash_graph = hashlib_json(self.neighbors_dict)  # identity the graph with network structure
        path = 'cache/graph_%s.pkl' % graph_id
        if os.path.exists(path):
            f = open(path, 'rb')
            neighbors, neighbors_of_neighbors = pickle.load(f)
            f.close()
        else:
            neighbors, neighbors_of_neighbors = self.dict_of_neighbors()
            f = open(path, 'wb')
            pickle.dump((neighbors, neighbors_of_neighbors), f)  # 将两个邻居字典作为一个元组存入序列化文件
            f.close()
        self.neighbors_dict = neighbors
        self.neighbors_of_neighbors_dict = neighbors_of_neighbors
        return neighbors, neighbors_of_neighbors

    def hash_graph(self):
        return

    def __eq__(self, other):
        return

    def load_graph(self):
        graph_ids = list(filter(lambda x: re.search('graph_id_', x), os.listdir('cache/')))
        graphs = [nx.read_gpickle(graph_id) for graph_id in graph_ids]
        return graphs

    def dump_graph(self):
        nx.write_gpickle(self, 'cache/graph_id_')

    def neighborhood_sim(self, x, y):
        neighbors = self.neighbors_dict
        neighbors_of_neighbors = self.neighbors_of_neighbors_dict
        rx = set(neighbors[x])  # set(x)
        rry = neighbors_of_neighbors[y]  # set(y)
        cn = rx & rry
        tn = rx | rry
        cn_size = float(len(cn))
        jc = cn_size / len(tn) if len(tn) > 0 else 0  # Jaccard (JC)
        aa = 0.0
        ra = 0.0
        for z in cn:
            if len(neighbors[z]) == 1:
                ra += 1.0 / len(neighbors[z])  # Resource Allocation（RA)
                continue
            aa += 1.0 / math.log(len(neighbors[z]))  # Adamic-Adar(AA)
            ra += 1.0 / len(neighbors[z])  # Resource Allocation（RA)
        pa = len(rx) * len(rry)  # Preferential Attachment(PA)
        cs = cn_size / math.sqrt(pa) if pa > 0 else 0  # Cosine/Salton
        lhn = cn_size / pa if pa > 0 else 0  # Leicht-Holme-Newman Index
        hpi = cn_size / min(len(rx), len(rry)) if pa > 0 else 0  # Hub Promoted Index(HPI)
        hdi = cn_size / max(len(rx), len(rry)) if max(len(rx), len(rry)) > 0 else 0  # Hub Depressed Index(HDI)
        sorence = cn_size * 2.0 / (len(rx) + len(rry)) if (len(rx) + len(rry)) > 0 else 0  # Sorence index
        cp = cn_size / len(rry) if len(rry) > 0 else 0  # Condition probability(CP)
        return cn_size, jc, aa, ra, pa, cs, lhn, hpi, hdi, sorence, cp

    def common_neighbors(self, x, y):
        neighbors = self.neighbors_dict
        neighbors_of_neighbors = self.neighbors_of_neighbors_dict
        rx = set(neighbors[x])  # set(x)
        rry = neighbors_of_neighbors[y]  # set(y)
        return len(rx & rry)

    def shortest_distance(self, x, y):
        try:
            sd = nx.shortest_path_length(self, x, y)  # SD
        except Exception as e:
            sd = 99
        return sd

    def local_path(self, all_data, alpha=0.001):
        array_data = np.array(all_data)
        # cn=[(self.common_neighbors(G,x[0],x[1])+self.common_neighbors(G,x[1],x[0]))/2.0 for x in array_data]
        cn = [self.common_neighbors(x[0], x[1]) for x in array_data]
        A = self.adj_matrix
        rows = self.userLE.transform(all_data['id'])
        cols = self.itemLE.transform(all_data['app'])
        cube_adj_mat = ((A ** 3)[rows, cols + self.user_size]).flat
        lp = pd.Series(cn) + alpha * pd.Series(cube_adj_mat)
        sm_data = all_data.copy()
        sm_data.insert(loc=3, column='LP', value=lp)
        return sm_data

    def local_shortest_path(self, all_data):
        array_data = np.array(all_data)
        sd = [self.shortest_distance(x[0], x[1]) for x in array_data]
        A = self.adj_matrix
        user_num = len(self.userlist)
        Dijkstra = np.zeros(A.shape)
        rows = self.userLE.transform(all_data['id'])
        cols = self.itemLE.transform(all_data['app'])
        Dijkstra[rows, user_num + cols] = sd
        maxD = np.max(Dijkstra)
        sim = Dijkstra
        k = 0
        while ((2 * k + 1) <= maxD):
            sim[Dijkstra == (2 * k + 1)] += 0.5 / np.array((np.power(A, 2 * k + 1)[Dijkstra == (2 * k + 1)]).flat)
            k += 1
        return sim

    def add_sim_to_edges(self, sim, all_data, feature='feat'):
        bias = 0
        if sim.shape == (self.user_size, self.item_size):
            bias = 0
        elif sim.shape[0] == (self.user_size + self.item_size):
            bias = self.user_size
        rows = self.userLE.transform(all_data['id'])
        cols = self.itemLE.transform(all_data['app'])
        series = pd.Series(sim[rows, cols + bias].flat)
        sm_data = all_data.copy()
        sm_data.insert(loc=3, column=feature, value=series)
        return sm_data

    def random_walk_with_restart(self, c):
        A = self.adj_matrix
        deg = np.repeat(A.sum(axis=1, dtype=np.float), A.shape[1], axis=1)  # degree matrix
        P = A / deg  # Transfer probability matrix
        #     P[np.isnan(P)]=0
        P = sp.sparse.csr_matrix(P)
        a = sp.sparse.eye(A.shape[0]) - c * P.T
        A2 = a[:self.user_size, self.user_size:]  # block A2, top right corner
        A3 = a[self.user_size:, :self.user_size]  # block A3, bottom left corner
        A4 = a[self.user_size:, self.user_size:]  # block A4, bottom right corner
        del P, a
        B4 = sp.sparse.linalg.inv(A4 - A3 * A2)
        B2 = -A2 * B4
        B3 = -B4 * A3
        del A2, A3, A4
        sim = (1 - c) * (B2 + B3.T)
        del B2, B3, B4
        return sim

    def random_walk_with_resource_redistribution(self, c):
        A = self.adj_matrix
        M = A.sum()
        deg = np.repeat(A.sum(axis=1, dtype=np.float), A.shape[1], axis=1)
        P = np.asarray(A / deg)  #
        #     P[np.isnan(P)]=0
        Qy = np.asarray(deg / M)
        a = np.zeros(A.shape)
        a[P.T > 0] = P.T[P.T > 0] * Qy[P.T > 0]
        a = sp.sparse.eye(A.shape[0]) - c * sp.sparse.csr_matrix(a)
        A2 = a[:self.user_size, self.user_size:]  # block A2, top right corner
        A3 = a[self.user_size:, :self.user_size]  # block A3, bottom left corner
        A4 = a[self.user_size:, self.user_size:]  # block A4, bottom right corner
        del deg, P, Qy, a
        B4 = sp.sparse.linalg.inv(A4 - A3 * A2)
        B2 = -A2 * B4
        B3 = -B4 * A3
        del A2, A3, A4
        sim = (1 - c) * (B2 + B3.T)
        del B2, B3, B4
        return sim







