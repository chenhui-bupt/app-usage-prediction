import pickle
import json
from network import BiGraph


G = BiGraph()
userlist = ['u1', 'u2', 'u3']
applist = ['a1', 'a2']
edges = [('u1', 'a1'), ('u1', 'a2'), ('u2', 'a1'), ('u3', 'a2')]
G.add_nodes_from(userlist, label='user')
G.add_nodes_from(applist, label='item')
G.add_edges_from(edges)
print(G.filter_nodes('label', 'user'))
print(G.filter_nodes('label', 'item'))
print(G.adj_matrix)

print(G.dict_of_neighbors())
print(G.neighbors_dict)
hash_graph = hash(pickle.dumps(G.neighbors_dict))
print(hash_graph)

