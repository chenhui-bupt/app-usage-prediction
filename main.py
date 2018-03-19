# -*- coding: utf-8 -*-
import itertools
import random

import numpy as np
import pandas as pd

from network.network import BiGraph

data1 = pd.read_csv('../datasets/nfp-small/data0117.csv', encoding='gbk', usecols=['id', 'app'])
data2 = pd.read_csv('../datasets/nfp-small/data0118.csv', encoding='gbk', usecols=['id', 'app'])
# data3 = pd.read_csv('../datasets/nfp-small/data0119.csv', encoding='gbk', usecols=['id', 'app'])
# data4 = pd.read_csv('../datasets/nfp-small/data0120.csv', encoding='gbk', usecols=['id', 'app'])


def get_edges(data):
    groups = data.groupby(['id', 'app']).groups
    edges = {g for g in groups}   
    return edges


def label_from_nextday(nextday_edges):
    userlist = list(map(lambda x: x[0], nextday_edges))
    applist = list(map(lambda x: x[1], nextday_edges))
    full_edges = set(itertools.product(userlist, applist))
    nextday_unedges = full_edges - nextday_edges
    nextday_unedges = random.sample(nextday_unedges, len(nextday_edges))  # random sample
    pos_data = pd.DataFrame(list(nextday_edges), columns=['id', 'app'])
    neg_data = pd.DataFrame(nextday_unedges, columns=['id', 'app'])
    pos_data['link'] = 1
    neg_data['link'] = 0
    all_data = pd.concat([pos_data, neg_data], ignore_index=True)
    return all_data


def gen_graph(data):
    G = BiGraph()
    userlist = data['id']
    applist = data['app']
    edges = get_edges(data)
    G.add_nodes_from(userlist, label='user')
    G.add_nodes_from(applist, label='item')
    G.add_edges_from(edges)
    return G


def neighbor_based_sim(G, all_data):
    arr_data = np.array(all_data)
    sim = [G.neighborhood_sim(x[0], x[1]) + G.neighborhood_sim(G, x[1], x[0]) for x in arr_data]
    cols = ['CN', 'JC', 'AA', 'RA', 'PA', 'CS', 'LHN', 'HP', 'HD', 'SI', 'CP', 'CN_2', 'JC_2', 'AA_2', 'RA_2',
            'PA_2', 'CS_2', 'LHN_2', 'HP_2', 'HD_2', 'SI_2', 'CP_2']
    sim_data = pd.DataFrame(sim, columns=cols)
    sim_data = pd.concat([all_data, sim_data], 1)
    return sim_data


edges1 = get_edges(data1)
edges2 = get_edges(data2)
# edges3 = get_edges(data3)
print(len(edges1))
all_data1 = label_from_nextday(edges2)
# all_data2 = label_from_nextday(edges3)
print(len(all_data1))
G1 = gen_graph(data1)
# G2 = gen_graph(data2)
print(list(G1.nodes)[:3])
A1 = G1.adj_matrix
# A2 = G2.adj_matrix
print(A1.shape)
# sim_data1 = neighbor_based_sim(G1, all_data1)
# sim_data2 = neighbor_based_sim(G2, all_data2)

G1.get_neighbors(1)
keys = list(G1.neighbors_dict.keys())

