## swapping axes of the 'result' arrays
# for 1) better data processing
# and for 2) easing the visualization.
def swap6(array):
    """ [Np, Ns, 6] """
    return array.swapaxes(1,2).swapaxes(0,1)

def swap33(array):
    """ [Np, Ns, 3, 3] """
    return array.swapaxes(2,0).swapaxes(3,1)

def pp(result, np=None, ns=None):
    """ Process the resulting arrays and return them"""
    dbar = result[0] #; sbar = result[1]
    dave = result[1]; davp = result[2]
    dbartot = result[3]; scau = result[4]
    scauchy = result[5]; epstot = result[6]
    epstote = result[7]; texture = result[8]

    return swap6(dbar), swap6(dave), swap6(davp), swap6(dbartot), \
        swap6(scau), swap33(scauchy), swap33(epstot), swap6(epstote),\
        texture\

def slope(x, y, upy, lowy):
    """
    Returns slope of the given stress and strain in the range indicated
    """
    import numpy as np
    # getting proper lower and upper index w.r.t. upy and lowy
    i = 0
    while True:
        if y[i]>lowy:
            ind1 = i
            break
        else : i = i + 1
    i = 0
    while True:
        if y[i]>upy:
            ind2 = i
            break
        else: i = i + 1
    while True:
        if y[i]> (lowy+upy)/2.:
            ind3 = i
            break
        else: i = i + 1
        
    # print 'Indices =', ind1, ind2
    # fitting and obtain the slope

    if ind1 == ind2 :
        ind2 = ind1 + 10
    z = np.polyfit(x[ind1:ind2],y[ind1:ind2],1)
    """
    try:
        z = np.polyfit(x[ind1:ind2],y[ind1:ind2],1)
    except TypeError:
        print x[ind1:ind2], y[ind1:ind2]
    """
    # returns the slope
    return z[0], ind1, ind2, ind3

def slope1(x,y):
    import numpy as np
    return np.polyfit(x,y,1)[0]


def windowed_smooth(x,y,dex=10):
    import numpy as np
    rst = []
    for i in range(len(x)):
        iskip = False
        if i<dex: iskip = True
        if i+dex>len(x): iskip = True

        if not(iskip):
            s = slope1(x[i-dex:i+dex:],y[i-dex:i+dex])
            rst.append(s)
        else: rst.append(np.nan)

    rst = np.array(rst)
    return rst

"""
def _interpolate_
def _interpolate2_
"""
def _interpolate_(x, y, x0):
    """
    Estimates new y by linearly interpolate the x-y curve
    at the given point, x0
    x0: 1-d array
    """
    y0 = 0
    if x0>max(x) or x0<min(x):
        print('The given x0 is out of the allowed range')
        print('The given x0: %f'%x0)
        print('max(x): %f'%max(x), 'min(x): %f'%min(x))
        raise IOError

    for i in range(len(x)):
        if x[i]>x0: break

    x1 = x[i - 1]; x2 = x[i]
    y1 = y[i - 1]; y2 = y[i]
    ## slope
    slope = (y2 - y1) / (x2 - x1)

    ## linear interpolation of interest y0 point.
    y0 = slope*(x0-x1) + y1
    return y0

def interpolate_curve(x,y,xs):
    new_y = []
    for i in range(len(xs)):
        new_y.append(_interpolate2_(x,y,xs[i]))
    return new_y

def _interpolate2_(x,y,x0):
    """
    Estimates new y be linearly interpolate the x-y curve
    at the given point x0, even if x0 is out of the range.
    Whenever x0 is out of the x range, extra-polate the data.


    an example:
    # ========================================
    # expx : experimental x data
    # expy : experimental y data
    # y : simulated dat along y-axis
    # ----------------------------------------
    newy, oldy, diff = [], [], []
    for i in range(len(y)):
       dum  = interpolate(expx, epxy, y[i])
       newy.append(dum)
       oldy.append(y[i])
       diff.append(y[i] - dum)

    diff = np.array(diff)
    stdv = abs(diff).sum()/(len(diff)-1)
    """
    if x0 > max(x):
        x1 = x[-2]; x2 = x[-1]
        y1 = y[-2]; y2 = y[-1]
        ## slope
        if x2==x1:
            slope=0
            y0 = 0
            pass
        else:
            try: slope = (y2 - y1) / (x2 - x1)
            except:
                print('x1, x2: %f %f'%(x1,x2))
                raise IOError('Error')
        ## linear extrapolate
            y0 = slope * (x0 - x1) + y1

        return y0
    elif x0 < min(x):
        x1 = x[0]; x2 = x[1]
        y1 = y[0]; y2 = y[1]
        ## slope
        if x2==x1:
            slope=0
            y0 = 0
            pass
        else:
            try: slope = (y2 - y1) / (x2 - x1)
            except:
                print('x1, x2: %f %f'%(x1, x2))
                raise IOError('Error')
            y0 = slope * (x0 - x1) + y1
        return y0

        # call _interpolate_
    else: return _interpolate_(x,y,x0)
    pass




# End of lib.py file
