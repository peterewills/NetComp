"""
NetComp
=======

    NetComp is a Python package for comparing networks using pairwise distances,
    and for performing anomaly detection on a time series of networks. It is
    built on top of the NetworkX package.

Using
-----

    Just use it!

"""

import sys
if sys.version_info[0] < 3:
    m = "Python 3.x required (%d.%d detected)."
    raise ImportError(m % sys.version_info[:2])
del sys

from netcomp.linalg import *
import netcomp.linalg

from netcomp.distance import *
import netcomp.distance

def in_the_right_package():
    print('Yup!')
