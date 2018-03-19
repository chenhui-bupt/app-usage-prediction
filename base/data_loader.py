# -*- coding: utf-8 -*-
import pandas as pd
import os
import itertools, random
from network.network import BiGraph


def get_edges(data):
    groups = data.groupby(['id', 'app']).groups
    edges = {g for g in groups}
    return edges

# 每天的边，密度， 连续出现不变的边， 不变率
def data_description(edges1, edges2):
    return len(edges1), len(edges1)/float(25413*55), len(edges1 & edges2), len(edges1 & edges2)/float(len(edges1))

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

def load_data():
    data1 = pd.read_csv('./input/data0117.csv', encoding='gbk', usecols=['id', 'app'])
    data2 = pd.read_csv('./input/data0118.csv', encoding='gbk', usecols=['id', 'app'])
    data3 = pd.read_csv('./input/data0119.csv', encoding='gbk', usecols=['id', 'app'])
    data4 = pd.read_csv('./input/data0120.csv', encoding='gbk', usecols=['id', 'app'])
    edges1 = get_edges(data1)
    edges2 = get_edges(data2)
    edges3 = get_edges(data3)
    all_data1 = label_from_nextday(edges2)
    all_data2 = label_from_nextday(edges3)
    des = data_description(edges1, edges2)
    print("the number of edges is %s, the density is %s, the invariant usage history is %s, "
          "and the invariant rate is %s" % des)
    make_dirs()
    data = [data1, data2, data3, data4]
    all_data = [all_data1, all_data2]
    return data, all_data


def gen_graph(data, graph_id=0):
    G = BiGraph()
    G.set_graph_id(graph_id)
    userlist = data['id']
    applist = data['app']
    edges = get_edges(data)
    G.add_nodes_from(userlist, label='user')
    G.add_nodes_from(applist, label='item')
    G.add_edges_from(edges)
    return G


def make_dirs():
    dirs = ['node', 'path', 'randomwalk']
    base = './output'
    for dir in dirs:
        path = os.path.join(base, dir)
        if not os.path.exists(path):
            os.makedirs(path)

