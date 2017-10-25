# A trivial example, for now

import netcomp as nc
import networkx as nx

G = nx.complete_graph(10)
A = nx.adjacency_matrix(G)
R = nc.resistance_matrix(A)
