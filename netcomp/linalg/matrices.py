"""
********
Matrices
********

Matrices associated with graphs. Also contains linear algebraic helper functions.
"""
from scipy import sparse as sps
import numpy as np


######################
## Helper Functions ##
######################


def _flat(D):
    """Helper function for flattening column or row matrices, as well as
    arrays"""
    try:
        d_flat = D.A1
    except AttributeError:
        d_flat = np.array(D).flatten()
    return d_flat


def _pad(A,N):
    """Pad A with so A.shape is (N,N)"""
    n,_ = A.shape
    if n>=N:
        return A
    else:
        side = np.zeros((n,N-n))
        A_pad = np.concatenate([A,side],axis=1)
        bottom = np.zeros((N-n,N))
        A_pad = np.concatenate([A_pad,bottom])
        return A_pad
    


########################
## Matrices of Grpahs ##
########################


def degree_matrix(A):
    """Diagonal degree matrix of graph with adjacency matrix A

    Parameters
    ----------
    A : matrix
        Adjacency matrix

    Returns
    -------
    D : SciPy sparse matrix
        Diagonal matrix of degrees.
    """
    n,m = A.shape
    degs = _flat(A.sum(axis=1))
    D = sps.spdiags(degs,[0],n,n,format='csr')
    return D


def laplacian_matrix(A,normalized=False):
    """Diagonal degree matrix of graph with adjacency matrix A

    Parameters
    ----------
    A : matrix
        Adjacency matrix
    normalized : Bool, optional (default=False)
        If true, then normalized laplacian is returned.

    Returns
    -------
    L : SciPy sparse matrix
        Combinatorial laplacian matrix.
    """
    n,m = A.shape
    D = degree_matrix(A)
    L = D - A
    if normalized:
        degs = _flat(A.sum(axis=1))
        rootD = sps.spdiags(np.power(degs,-1/2), [0], n, n, format='csr')
        L = rootD*L*rootD
    return L  
