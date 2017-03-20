"""
***************
Exact Distances
***************

Calculation of exact distances between graphs. Generally slow.
"""

import numpy as np
from numpy import linalg as la
from math import ceil
import networkx as nx

from netcomp.linalg import (renormalized_res_mat,resistance_matrix,
                            fast_bp,laplacian_matrix,normalized_laplacian_eig,
                            _flat,_pad,_eigs)
from netcomp.distance import get_features,aggregate_features
from netcomp.exception import InputError


def vertex_edge_overlap(A1,A2):
    """Vertex-edge overlap. Basically a souped-up edit distance, but in similarity
    form. The VEO similarity is defined as

        VEO(G1,G2) = (|V1&V2| + |E1&E2|) / (|V1|+|V2|+|E1|+|E2|)

    where |S| is the size of a set S and U&T is the union of U and T.

    Parameters
    ----------
    A1, A2 : NumPy matrices
        Adjacency matrices of graphs to be compared

    Returns
    -------
    sim : float
        The similarity between the two graphs
    

    References
    ----------

    """
    try:
        [G1,G2] = [nx.from_scipy_sparse_matrix(A) for A in [A1,A2]]
    except AttributeError:
        [G1,G2] = [nx.from_numpy_matrix(A) for A in [A1,A2]]
    V1,V2 = [set(G.nodes()) for G in [G1,G2]]
    E1,E2 = [set(G.edges()) for G in [G1,G2]]
    V_overlap = len(V1|V2) # set union
    E_overlap = len(E1|E2)
    sim = (V_overlap + E_overlap) / (len(V1)+len(V2)+len(E1)+len(E2))
    return sim

def vertex_edge_distance(A1,A2):
    """Vertex-edge overlap transformed into a distance via

        D = (1-VEO)/VEO

    which is the inversion of the common distance-to-similarity function

        sim = 1/(1+D).

    Parameters
    ----------
    A1, A2 : NumPy matrices
        Adjacency matrices of graphs to be compared

    Returns
    -------
    dist : float
        The distance between the two graphs
    """
    sim = vertex_edge_overlap(A1,A2)
    dist = (1-sim)/sim
    return dist

    
def lambda_dist(A1,A2,k=None,p=2,kind='laplacian'):
    """The lambda distance between graphs, which is defined as

        d(G1,G2) = norm(L_1 - L_2)

    where L_1 is a vector of the top k eigenvalues of the appropriate matrix
    associated with G1, and L2 is defined similarly.

    Parameters
    ----------
    A1, A2 : NumPy matrices
        Adjacency matrices of graphs to be compared

    k : Int
        The number of eigenvalues to be compared. Eigenvalues are sorted from
        greatest to least. We require k <= min(N1,N2) where N1 and N2 are the
        sizes of A1 and A2 respectively.

    p : non-zero Float
        The p-norm is used to compare the resulting vector of eigenvalues.
    
    kind : String , in {'laplacian','laplacian_norm','adjacency'}
        The matrix for which eigenvalues will be calculated.

    Returns
    -------
    dist : float
        The distance between the two graphs

    Notes
    -----
    The norm can be any p-norm; by default we use p=2. If p<0 is used, the
    result is not a mathematical norm, but may still be interesting and/or
    useful.

    References
    ----------

    See Also
    --------
    netcomp.linalg._eigs
    normalized_laplacian_eigs

    """
    # ensure valid k
    n1,n2 = [A.shape[0] for A in [A1,A2]]
    N = min(n1,n2) # minimum size between the two graphs
    if k is None:
        k = ceil(N/10)
    if N < k:
        raise InputError('k must be strictly less than the minimum size of the'
                         ' two input graphs.')
    if kind is 'laplacian':
        # form matrices
        L1,L2 = [laplacian_matrix(A) for A in [A1,A2]]
        # get eigenvalues, ignore eigenvectors
        evals1,evals2 = [_eigs(L,k=k)[0] for L in [L1,L2]]
    elif kind is 'laplacian_norm':
        # use our function to graph evals of normalized laplacian
        evals1,evals2 = [normalized_laplacian_eig(A,k=k)[0] for A in [A1,A2]]
    elif kind is 'adjacency':
        evals1,evals2 = [_eigs(A,k=k)[0] for A in [A1,A2]]
    else:
        raise InputError("Invalid type, choose from 'laplacian', "
                         "'laplacian_norm', and 'adjacency'.")
    dist = la.norm(evals1-evals2,ord=p)
    return dist


def netsimile(A1,A2):
    """NetSimile distance between two graphs.

    Parameters
    ----------
    A1, A2 : SciPy sparse array
        Adjacency matrices of the graphs in question.
    
    Returns
    -------
    d_can : Float
        The distance between the two graphs.

    Notes
    -----
    NetSimile works on graphs without node correspondence. Graphs to not need to
    be the same size.

    See Also
    --------

    References
    ----------
    """ 
    feat_A1,feat_A2 = [get_features(A) for A in [A1,A2]]
    agg_A1,agg_A2 = [aggregate_features(feat) for feat in [feat_A1,feat_A2]]
    # calculate Canberra distance between two aggregate vectors
    d_can = (np.abs(agg_A1-agg_A2) / (agg_A1 + agg_A2)).sum()
    return d_can
    
    
def res_dist(A1,A2,p=2,renormalized=False,attributed=False):
    """Compare two graphs using resistance distance (possibly renormalized).

    Parameters
    ----------
    A1, A2 : NumPy Matrices
        Adjacency matrices of graphs to be compared.

    p : float
        Function returns the p-norm of the flattened matrices.

    renormalized : Boolean, optional (default = False)
        If true, then renormalized resistance distance is computed.

    attributed : Boolean, optional (default=False)
        If true, then the resistance distance PER NODE is returned.

    Returns
    -------
    dist : float of numpy array
        The RR distance between the two graphs. If attributed is True, then
        vector distance per node is returned.

    Notes
    -----
    The distance is calculated by assuming the nodes are in correspondence, and
    any nodes not present are treated as isolated by renormalized resistance.

    References
    ----------

    See Also
    --------
    resistance_matrix
    """
    # Calculate resistance matricies and compare
    if renormalized:
        # pad smaller adj. mat. so they're the same size
        n1,n2 = [A.shape[0] for A in [A1,A2]]
        N = max(n1,n2)
        A1,A2 = [_pad(A,N) for A in [A1,A2]]
        R1,R2 = [renormalized_res_mat(A) for A in [A1,A2]]
    else:
        R1,R2 = [resistance_matrix(A) for A in [A1,A2]]
    distance_vector = np.sum((R1-R2)**p,axis=1)
    if attributed:
        return distance_vector**(1/p)
    else:
        return np.sum(distance_vector)**(1/p)

def deltacon0_dist(A1,A2,eps=None):
    """DeltaCon0 distance between two graphs. The distance is the Frobenius norm
    of the element-wise square root of the fast belief propogation matrix.

    Parameters
    ----------
    A1, A2 : NumPy Matrices
        Adjacency matrices of graphs to be compared.

    Returns
    -------
    dist : float
        DeltaCon0 distance between graphs.

    References
    ----------

    See Also
    --------
    fast_bp
    """
    S1,S2 = [fast_bp(A,eps=eps) for A in [A1,A2]]
    dist = la.norm(np.sqrt(_flat(S1)) - np.sqrt(_flat(S2)))
    return dist
