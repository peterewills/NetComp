"""
**********
Eigenstuff
**********

Functions for calculating eigenstuff of graphs.
"""

from scipy import sparse as sps
from numpy import linalg as la
import numpy as np

# irrelevant until I implement sanity checks (See below)
from netcomp.exception import InputError
from netcomp.linalg.matrices import _flat,_eps

######################
## Helper Functions ##
######################


def _eigs(A,k):
    """ Helper function. Runs numpy.linalg.eig if k is n or n-1, and runs
    scipy.sparse.linalg.eigs otherwise. Note that n = A.shape[0]."""
    n,_ = A.shape
    # calculation of eigvals depends on k compared to n
    if k > n:
        raise InputError('k greater than n.')
    elif k in [n-1,n]:
        try:
            A = A.todense()
        except AttributeError:
            pass
        # use numpy
        evals,evecs = la.eig(A)
        # sort dem eigenvalues
        inds = np.argsort(evals)[::-1]
        evals = evals[inds[:k]]
        evecs = evecs[:,inds[:k]]
        return evals,evecs
    else:
        try:
            # if A is made of ints, we can't do eigs() below
            A = A.asfptype()
        except AttributeError:
            pass
        evals,evecs = sps.linalg.eigs(A,k=k)
        # sps.linalg.eigs returns the k smallest, sorted from largest to
        # smallest. So we reverse the ordering to match the above.
        return evals[::-1],evecs[:,::-1]


#####################
##  Get Eigenstuff ##
#####################

def normalized_laplacian_eig(A,k=None):
    """Return the eigenstuff of the normalized Laplacian matrix of graph
    associated with adjacency matrix A.

    Calculates via eigenvalues if 

    K = D^(-1/2) A D^(-1/2)

    where `A` is the adjacency matrix and `D` is the diagonal matrix of
    node degrees. Since L = I - K, the eigenvalues and vectors of L can 
    be easily recovered.

    Parameters
    ----------
    A : NumPy matrix
        Adjacency matrix of a graph

    k : Int
        Number of eigenvalues to return. By default, all are returned.

    Returns
    -------
    lap_evals : NumPy array
       Eigenvalues of L

    evecs : NumPy matrix
       Columns are the eigenvectors of L

    Notes
    -----
    This way of calculating the eigenvalues of the normalized graph laplacian is
    more numerically stable than simply forming the matrix L = I - K and doing
    numpy.linalg.eig on the result. This is because the eigenvalues of L are
    close to zero, whereas the eigenvalues of K are close to 1.

    References
    ----------

    See Also
    --------
    nx.laplacian_matrix
    nx.normalized_laplacian_matrix
    """
    n,m = A.shape
    ##
    ## TODO: implement checks on the adjacency matrix
    ##
    degs = _flat(A.sum(axis=1))
    # the below will break if
    inv_root_degs = [d**(-1/2) if d>_eps else 0 for d in degs]
    inv_rootD = sps.spdiags(inv_root_degs, [0], n, n, format='csr')
    # build normalized diffusion matrix
    K = inv_rootD*A*inv_rootD
    if k is None:
        k = n
    evals,evecs = _eigs(K,k)
    lap_evals = 1-evals
    return np.real(lap_evals),np.real(evecs)
