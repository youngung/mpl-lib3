import numpy as np
def smooth_slope(x,y,nbin=10):
    """
    A custom method to smooth a noisy data
    Particularly useful when obtaining 'slopes'
    such as R-values, hardening rate and their changes 
    with respect to time stamp.

    Using np.linslg.lstsq calculate the smoothed slopes
    of nosy x-y curve
    """
    # least square method
    lstsq = np.linalg.lstsq
    # making windows
    slopes = []
    for i in range(len(y)):
        i0 = i-nbin/2
        i1 = i+nbin/2
        # least square method
        if i0<0 or i1>len(y)-1:
            slopes.append(np.nan)
        else:
            Y = y[i0:i1]
            X = x[i0:i1]
            A=np.vstack([X,np.ones(len(X))]).T
            m,c = lstsq(A, Y)[0]
            slopes.append(m)
            
    slopes=np.array(slopes)
    # fit individual windows
    # return the slopes
    return slopes
