# A trivial example, for now

import netcomp as nc
import networkx as nx

# use networkx to build a sample graph
G0 = nx.complete_graph(10)
G1 = G0.copy()
G1.remove_edge(1, 0)
G2 = G0.copy()
G2.remove_node(0)
A, B, C = [nx.adjacency_matrix(G) for G in [G0, G1, G2]]

# matrix distances
nc.resistance_distance(A, B);
nc.edit_distance(A, B);
nc.deltacon0(A, B);
nc.vertex_edge_distance(A, B);
# spectral distances
nc.lambda_dist(A, C);
nc.lambda_dist(A, C, kind='adjacency', k=2);
nc.lambda_dist(A, C, kind='laplacian_norm');
# other distances
nc.resistance_distance(A, C, renormalized=True);
nc.netsimile(A, C);
# matrices w/o associated distances
nc.commute_matrix(A);
nc.conductance_matrix(A);
