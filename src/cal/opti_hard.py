"""
Optimize the hardening parameters for the given func.
"""
import numpy as np
from . import func_hard
from scipy.optimize import curve_fit

def main(func,eps,sigma,p0):
    """
    Arguments
    ---------
    func
    eps
    sigma
    p0

    Returns
    -------
    eps, fittedF, popt
    """
    popt,pcov=curve_fit(func,eps,sigma,check_finite=True)
    #ax.plot(xdat,func(xdat,*popt),label='Voce2 fit')
    return eps, func(eps,*popt), popt

def ex(fn='/Users/yj/Documents/ExpDat/IFSteel/Bulge/EXP_BULGE_JINKIM.txt'):
    """
    Running all of the funcs in func_hard to fit experimental from <fn>

    Argument
    --------
    fn
    """
    import matplotlib.pyplot as plt
    dat=np.loadtxt(fn).T
    xdat,ydat=dat

    funcs=[func_hard.func_voce2,func_hard.func_voce,func_hard.func_swift]
    plt.plot(xdat[::10],ydat[::10],'x',label='exp')

    for f in funcs:
        x,y,popt = main(f,xdat,ydat,None)
        plt.plot(x,y,label=f.__name__,alpha=0.5)
        print(f.__name__, popt)

    ax=plt.gca()
    ax.legend(loc='best')

## command line usage
if __name__=='__main__':
    ex()
