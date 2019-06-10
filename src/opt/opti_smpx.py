## optimization based on simplex method implemented in scipy
from scipy.optimize import fmin
import numpy as np
def smplx_bcc_uniax_voce(parameters=[80,70,340,20],
                         exp_fn='mat/Bst/avgstr_000.txt',maxiter=2,xtol=0.001):
    from . import objfs
    objf=objfs.BCC_uniax_voce

    # import matplotlib.pyplot as plt
    # fig=plt.figure(1,figsize=(4,3));ax=fig.add_axes((0.25,0.25,0.7,0.7))
    # dat=np.loadtxt(exp_fn).T
    # ax.plot(dat[0],dat[1]

    exp_dat=np.loadtxt(exp_fn).T
    sx_fn = open('vpsc7.in').readlines()[10].split('\n')[0]

    print('sx_fn:',sx_fn)

    xopt = fmin(objf,
                parameters,
                args=(exp_dat,sx_fn),
                xtol=xtol,maxfun=maxiter,retall=True)
    return xopt

def smplx_bcc_uniax_voce_sat(parameters=[80,70,340],thet0=0.,
                             exp_fn='mat/Bst/avgstr_000.txt',maxiter=2,xtol=0.001):
    from . import objfs
    objf=objfs.BCC_uniax_voce_sat

    import matplotlib.pyplot as plt
    fig=plt.figure(1,figsize=(4,3));ax=fig.add_axes((0.25,0.25,0.7,0.7))
    dat=np.loadtxt(exp_fn).T
    ax.plot(dat[0],dat[1],'rx',label='EXP uni')

    exp_dat=np.loadtxt(exp_fn).T
    sx_fn = open('vpsc7.in').readlines()[10].split('\n')[0]

    print('sx_fn:',sx_fn)

    xopt = fmin(objf,
                parameters,
                args=(exp_dat,thet0,sx_fn),
                xtol=xtol,maxfun=maxiter,retall=True)
    return xopt


def smplx_bcc_c3_voce(parameters=[80,70,340,20],
                      exp_fn='mat/Bst/EXP_BULGE_JINKIM.txt',maxiter=2,xtol=0.001):
    from . import objfs
    objf=objfs.BCC_c3_voce

    import matplotlib.pyplot as plt
    fig=plt.figure(1,figsize=(4,3));ax=fig.add_axes((0.25,0.25,0.7,0.7))
    dat=np.loadtxt(exp_fn).T
    ax.plot(dat[0][::10],dat[1][::10],'rx',label='EXP bulge')

    exp_dat=np.loadtxt(exp_fn).T
    sx_fn = open('vpsc7.in').readlines()[10].split('\n')[0]

    print('sx_fn:',sx_fn)

    xopt = fmin(objf,
                parameters,
                args=(exp_dat,sx_fn),
                xtol=xtol,maxfun=maxiter,retall=True)


    dat=np.loadtxt('STR_STR.OUT',skiprows=1).T
    ax.plot(-dat[4],-dat[10],'r-',label='VPSC')

    ax.legend(loc='best')
    fig.savefig('smplx_bcc_c3_voce.pdf')
    return xopt



def smplx_bcc_c3_voce_sat(parameters=[80,70,340],thet0=0.,
                          exp_fn='mat/Bst/EXP_BULGE_JINKIM.txt',maxiter=2,xtol=0.001):
    """
    thet1 is fixed to be zero in this hardening model
    """
    from . import objfs
    objf=objfs.BCC_c3_voce_sat

    import matplotlib.pyplot as plt
    fig=plt.figure(1,figsize=(4,3));ax=fig.add_axes((0.25,0.25,0.7,0.7))
    dat=np.loadtxt(exp_fn).T
    ax.plot(dat[0][::10],dat[1][::10],'rx',label='EXP bulge')

    exp_dat=np.loadtxt(exp_fn).T
    sx_fn = open('vpsc7.in').readlines()[10].split('\n')[0]

    print('sx_fn:',sx_fn)
    xopt = fmin(objf,
                parameters,
                args=(exp_dat,thet0,sx_fn),
                xtol=xtol,maxfun=maxiter,retall=True)

    dat=np.loadtxt('STR_STR.OUT',skiprows=1).T
    ax.plot(-dat[4],-dat[10],'r-',label='VPSC')

    ax.legend(loc='best')
    fig.savefig('smplx_bcc_c3_voce_sat.pdf')
    return xopt
