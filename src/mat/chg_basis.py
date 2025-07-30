import numpy as np
sqrt=np.sqrt

def kasemer(cart=None,s=None,iopt=None):
    """
    Following the convention used by Kasemer and Dawson.
    https://doi.org/10.48550/arXiv.2502.19531

    Arguments
    ---------
    cart
    s
    iopt

    Returns
    -------
    if iopt==0: returns s (vectorized 5-D deviatoric quantity)
    if iopt==1: returns sig (cartesian 3x3 quantity)
    """
    if iopt==0:
        rst=np.zeros(5)
        rst[0]=sqrt(0.5)*(cart[0,0]-cart[1,1])
        rst[1]=sqrt(1.5)* cart[2,2]
        rst[2]=sqrt(2)  * cart[1,2]
        rst[3]=sqrt(2)  * cart[0,2]
        rst[4]=sqrt(2)  * cart[0,1]
        return rst
    elif iopt==1:
        rst=np.zeros((3,3))
        ## normal components
        rst[0,0]=0.5**0.5*s[0]-6**(-0.5)*s[1]
        rst[2,2]=1.5**(-0.5)*s[1]
        rst[1,1]=-rst[0,0]-rst[2,2]

        ## shear components
        rst[1,2]=1./sqrt(2)*s[2]
        rst[2,1]=rst[1,2]

        rst[0,2]=1./sqrt(2)*s[3]
        rst[2,0]=rst[0,2]

        rst[0,1]=1./sqrt(2)*s[4]
        rst[1,0]=rst[0,1]

        return rst


def kocks(a,iopt):
    """
    Convention used in Kocks, Canova, Jonas, Acta Metallurgica, Vol 31 (1983)
    https://doi.org/10.1016/0001-6160(83)90186-4

    iopt0: Convert tensor represented by 3x3 matrix to 5-D vector
    iopt1: Convert tensor represented by 5-D vector to 3x3 matrix

    Arguments
    ---------
    a
    iopt  0: a is 2nd order tensor represented by 3x3 matrix form
          1: a is 2nd order tensor represented by 5-D vector form

    Returns
    -------
    rst  0: rst is 2nd order tensor represented by 5-D vector form
         1: rst is 2nd order tensor represented by 3x3 matrix form
    """
    if iopt==0:
        rst=np.zeros(5)
        rst[0]=(a[0,0]-a[1,1])/2.
        rst[1]=1.5*a[2,2]
        rst[2]=a[1,2]
        rst[3]=a[2,0]
        rst[4]=a[0,1]
    elif iopt==1:
        rst=np.zeros((3,3))
        ## normal components
        rst[0,0]= a[0]-1./3.*a[1]
        rst[1,1]=-a[0]-1./3.*a[1]
        rst[2,2]=2./3.*a[1]
        ## shear components
        rst[1,2]=a[2]
        rst[2,1]=rst[1,2]

        rst[2,0]=a[3]
        rst[0,2]=rst[2,0]

        rst[0,1]=a[4]
        rst[1,0]=rst[0,1]

    return rst

def chg_basis(ce2=None, c2=None, ce4=None, c4=None, iopt=0, kdim=5):
    """
    parameter(sqr2 = sqrt(2.))
    parameter(rsq2 = 1./sqrt(2.))
    parameter(rsq3 = 1./sqrt(3.))

    ce2 = array(kdim)
    c2 = array(3,3)
    ce4 = array(kdim,kdim)
    c4 = array(3,3,3,3)

    iopt = 1:
        ce2  -- >  c2
    iopt = 2:
        c2   -- >  ce2e
    iopt = 3:
        ce4  -- > c4
    iopt = 4:
        c4   -- > ce4
    """
    sqr2 = sqrt(2.)
    rsq2 = 1./sqrt(2.)
    rsq3 = 1./sqrt(3.)
    rsq6 = 1./sqrt(6.)

    b = np.resize(np.array((0.)), (3,3,6))
    if type(ce2) == type(None): ce2 = np.resize(np.array((0.)), (kdim))
    if type(c2)  == type(None): c2  = np.resize(np.array((0.)), (3,3))
    if type(ce4) == type(None): ce4 = np.resize(np.array((0.)), (kdim,kdim))
    if type(c4)  == type(None): c4  = np.resize(np.array((0.)), (3,3,3,3))

    b[0][0][1] = -rsq6
    b[1][1][1] = -rsq6
    b[2][2][1] = 2.*rsq6

    b[0][0][0] = -rsq2
    b[1][1][0] = rsq2

    b[1][2][2] = rsq2
    b[2][1][2] = rsq2

    b[0][2][3] = rsq2
    b[2][0][3] = rsq2

    b[0][1][4] = rsq2
    b[1][0][4] = rsq2

    b[0][0][5] = rsq3
    b[1][1][5] = rsq3
    b[2][2][5] = rsq3

    if iopt == 0:
        return b

    elif iopt == 1:
        if True:
            return np.tensordot(ce2[:kdim],b[:,:,:kdim],axes=((0),(2)))
        else:
            for i in range(3):
                for j in range(3):
                    c2[i][j] = 0.0
                    for n in range(kdim):
                        c2[i][j] = c2[i][j]+ce2[n]*b[i][j][n]
            return c2

    elif iopt == 2:
        if True:
            return np.tensordot(c2[:kdim],b[:,:,:kdim],axes=((0,1),(0,1)))
        else:
            for n in range(kdim):
                ce2[n] = 0.0
                for i in range(3):
                    for j in range(3):
                        ce2[n] = ce2[n] + c2[i][j]*b[i][j][n]
            return ce2
    elif iopt == 3:
        if False:
            return
        else:
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            c4[i][j][k][l] = 0.
                            for n in range(kdim):
                                for m in range(kdim):
                                    c4[i][j][k][l] = c4[
                                        i][j][k][l] + ce4[n][m] * b[
                                        i][j][n] * b[k][l][m]
            return c4
    elif iopt == 4:
        if False:
            return
        else:
            for n in range(kdim):
                for m in range(kdim):
                    ce4[n][m] = 0.
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                for l in range(3):
                                    ce4[n][m] = ce4[
                                        n][m]+c4[i][j][k][l]* b[
                                        i][j][n]*b[k][l][m]
            return ce4
