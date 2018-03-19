# -*- coding: utf-8 -*-
import os
import pickle
import math
import networkx as nx 


class BiGraph(nx.Graph):

    @property
    def adj_matrix(self, nodelist):
        return nx.adjacency_matrix(self, nodelist=nodelist)

    def dict_of_neighbors(self):
        """
    	neighbors: is the direct neighbors of every node
    	neighbors_of neighbors: is the indirect neighbors of every node
    	"""
        neighbors = {}  # its value's type is list
        for node in self.nodes():
            neighbors[node] = self.neighbors(node)
        neighbors_of_neighbors={} # its value's type is set
        for node in self.nodes():
            indirect_neighbors = set()
            for neig_node in neighbors[node]:
                indirect_neighbors.add(neighbors[neig_node])
            neighbors_of_neighbors[node] = indirect_neighbors
        self.neighbors = neighbors
        self.neighbors_of_neighbors = neighbors_of_neighbors
        return neighbors, neighbors_of_neighbors

    def get_neighbors(self):
        path = 'cache/neighbors_%s.pkl' % hash(self)
        if os.path.exists(path):
            f = open(path, 'rb')
            neighbors, neighbors_of_neighbors = pickle.load(f)
            f.close()
        else:
            neighbors, neighbors_of_neighbors = self.dict_of_neighbors()
            f = open(path, 'wb')
            pickle.dump((neighbors, neighbors_of_neighbors), f)  # 将两个邻居字典作为一个元组存入序列化文件
            f.close()
        return neighbors, neighbors_of_neighbors

    def neighborhood_sim(self, x, y):
        neighbors = self.neighbors
        neighbors_of_neighbors = self.neighbors_of_neighbors
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
        neighbors = self.neighbors
        neighbors_of_neighbors = self.neighbors_of_neighbors
        rx = set(neighbors[x])  # set(x)
        rry = neighbors_of_neighbors[y]  # set(y)
        return len(rx & rry)

    def localPath(G, A, all_data, neighbors, neighbors_of_neighbors, alpha=0.001):
        array_data = np.array(all_data)
        #     cn=[(CN(G,x[0],x[1],neighbors,neighbors_of_neighbors)+CN(G,x[1],x[0],neighbors,neighbors_of_neighbors))/2.0 for x in array_data]
        cn = [CN(G, x[0], x[1], neighbors, neighbors_of_neighbors) for x in array_data]
        A3 = A ** 3
        rows = userLE.transform(all_data['id'])
        cols = appLE.transform(all_data['app'])
        a3 = pd.Series((A3[rows, cols + usernum]).flat)
        lp = pd.Series(cn) + alpha * a3
        sm_data = all_data.copy()
        sm_data.insert(loc=3, column='LP', value=lp)
        return sm_data
