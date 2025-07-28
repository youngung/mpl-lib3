import numpy as np
import sys
ijv = np.zeros((2,6),dtype='int')
ijv[0,0],ijv[1,0] = 0,0
ijv[0,1],ijv[1,1] = 1,1
ijv[0,2],ijv[1,2] = 2,2
ijv[0,3],ijv[1,3] = 1,2
ijv[0,4],ijv[1,4] = 0,2
ijv[0,5],ijv[1,5] = 0,1

vij = np.zeros((3,3),dtype='int')
vij[0,0] = 0
vij[1,1] = 1
vij[2,2] = 2
vij[1,2] = 3
vij[0,2] = 4
vij[0,1] = 5

# lower triangle
vij[2,1] = 3
vij[2,0] = 4
vij[1,0] = 5

def voigt(aux33,aux6,iopt=0):
    """
    iopt 0: aux33 (Cart.) -> aux6  (Voigt)
    returns aux6

    iopt 1: aux6  (Voigt) -> aux33 (Cart.)
    returns aux33
    """
    if iopt==0:
        rst6=np.zeros(6)
        for iv in range(6):
            i0,j0=ijv[:,iv]
            rst6[iv]=aux33[i0,j0]
        return rst6
    elif iopt==1:
        rst33=np.zeros((3,3))
        for iv in range(6):
            i0,j0=ijv[:,iv]
            rst33[i0,j0]=aux6[iv]
            rst33[j0,i0]=aux33[i0,j0]
        return rst33
    else:
        sys.stderr(f'** Error: unexpected iopt <{iopt}> was given\n')
        sys.exit(-1)
