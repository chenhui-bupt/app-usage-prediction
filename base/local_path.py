# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import pandas as pd


usernum = 0
appnum = 0
nodelist = []
userLE = None
appLE = None


def cn(G, x, y, neighbors, neighbors_of_neighbors):
    rx = set(neighbors[x])
    ry = neighbors_of_neighbors[y]
    cn = rx & ry
    return cn


def localPath(G, all_data, cn):
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    A3 = A**3
    rows = userLE.transform(all_data['id'])
    cols = appLE.transform(all_data['app'])
    a3 = pd.Series((A3[rows, cols+usernum]).flat)
    lp = cn+0.001*a3
    sm_data = all_data.copy()
    sm_data.insert(loc=3, column='lp', value=lp)
    return sm_data


def katz(A, alpha):
    w=np.linalg.eigvalsh(A.toarray())  #特征值
    beta=alpha/w.max()
    dim=A.shape[0]
    sim=np.linalg.inv(np.eye(dim)-beta*A) - np.eye(dim)
    return sim[:usernum, usernum:]


def LHNII(A, alpha):  # A是稀疏矩阵csr_matrix
    M = A.sum()/2  # 边的数目
    deg = np.repeat(1.0/A.sum(axis=1), A.shape[1], axis=1)
    D = np.diag(np.diag(deg))  # 度矩阵的逆矩阵
    maxeig = max(np.linalg.eigvalsh(A.todense()))  # 最大特征值
    temp = np.eye(A.shape[0])-(alpha/maxeig)*A
    temp = np.linalg.inv(temp)
    sim = 2*M*maxeig*D*temp*D  # 叉乘外积，注意
    return sim[:usernum, usernum:]


def add_sim_to_edges(sim, all_data, feature='feat'):
    if sim.shape == (usernum, appnum):
        bias = 0
    elif sim.shape[0] == (usernum+appnum):
        bias = usernum
    rows = userLE.transform(all_data['id'])
    cols = appLE.transform(all_data['app'])
    series = pd.Series(sim[rows, cols+bias].flat)
    sm_data = all_data.copy()
    sm_data.insert(loc=3, column=feature, value=series)
    return sm_data


def add_sim_to_edges(sim, all_data, feature='feat'):
    if sim.shape == (usernum, appnum):
        bias = 0
    elif sim.shape[0] == (usernum+appnum):
        bias = usernum
    rows = userLE.transform(all_data['id'])
    cols = appLE.transform(all_data['app'])
    series = pd.Series(sim[rows, cols+bias].flat)
    sm_data = all_data.copy()
    sm_data.insert(loc=3, column=feature, value=series)
    return sm_data