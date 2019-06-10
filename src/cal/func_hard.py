"""
Collection of phenomenological
strain hardening functions
"""
import numpy as np
exp=np.exp
def func_voce2(eps,a,b0,c,b1):
    """
    sigma = a-b0*exp(-c*eps)+b1*eps

    Arguments
    --------
    eps,a,b0,c,b1
    """
    return a-b0*exp(-c*eps)+b1*eps

def func_voce(eps,a,b,c):
    """
    sigma = a-b*exp(c*eps)

    Arguments
    ---------
    eps,a,b,c
    """
    return a-b*exp(c*eps)

def func_swift(eps,k,eps_0,n):
    """
    sigma = k * (eps+eps_0) **n

    Arguments
    ---------
    eps,k,eps_0,n
    """
    return k * (eps+eps_0) **n
