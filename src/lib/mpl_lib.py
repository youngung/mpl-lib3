import matplotlib.pyplot as plt

def fancy_legend(ax,loc='best',size=12,ncol=None,
                 nscat=2,
                 bbox_to_anchor=None):
    """
    MPL Legend with FancyBox
    """
    if type(ncol)==type(None):
        if type(bbox_to_anchor)==type(None):
            ax.legend(
                loc=loc,fancybox=True, framealpha=0.5,
                prop={'size':size},numpoints=nscat)
        else:
            ax.legend(
                loc=loc,fancybox=True, framealpha=0.5,
                prop={'size':size},numpoints=nscat,
                bbox_to_anchor=bbox_to_anchor)
    elif type(ncol).__name__=='int':
        if type(bbox_to_anchor)==type(None):
            ax.legend(loc=loc,fancybox=True, framealpha=0.5,
                      ncol=ncol,numpoints=nscat,
                      prop={'size':size})
        else:
            ax.legend(loc=loc,fancybox=True, framealpha=0.5,
                      ncol=ncol,numpoints=nscat,
                      prop={'size':size},
                      bbox_to_anchor=bbox_to_anchor)
    else:
        raise IOError('Unexpected type for ncol')


def ticks_bins_ax_u(axes,n=4):
    ticks_bins_ax(axes,axis='x',n=n)
    ticks_bins_ax(axes,axis='y',n=n)

def ticks_bins_ax(axes,axis='x',n=4):
    for i in range(len(axes)):
        ticks_bins(axes[i],axis=axis,n=n)

def ticks_bins(ax,axis='x',n=4):
    ax.locator_params(nbins=n,axis=axis)

def rm_lab(ax,axis='y'):
    if axis=='x': plt.setp(ax.get_xticklabels(), visible=False)
    if axis=='y': plt.setp(ax.get_yticklabels(), visible=False)

    if axis=='x': ax.set_xlabel('')
    if axis=='y': ax.set_ylabel('')

def rm_ax(ax,axis='x'):
    if axis=='x': ax.get_xaxis().set_visible(False)
    if axis=='y': ax.get_yaxis().set_visible(False)

def rm_inner(ax):
    ## delete inner axes ticks
    for i in range(len(ax)-1):
        rm_lab(ax[i+1],axis='x')
        rm_lab(ax[i+1],axis='y')

def rm_all_lab(ax):
    for i in range(len(ax)):
        rm_lab(ax[i],axis='x')
        rm_lab(ax[i],axis='y')

def wide_fig(
        ifig=None,uw=2.8, uh=3,nw=5, nh=1,nfig=None,
        w0=0.2, ws=0.7, w1=0.1,
        left = 0.05, right = 0.02,
        h0=0.2, hs=0.7, h1=0.1,
        down = 0.05, up = 0.02, iarange=False,
        useOffset=False):
    """
    Make a figure that has horizontally aligned sub axes

    Arguments
    ---------
    ifig : mpl figure number

    uw  : unit width for a subaxes
    uh  : unit height
    nw  : number of axes

    h0  : spacing h0 for a subaxes
    hs  : width of uniax axes
    h1  : h0+hs+h1 is the total heigth of an axes

    w0  : w0,ws, and w1 are corresponding values for width
    ws
    w1
    l   : left spacing in the figure canvas
    r   : right spacing in the figure canvas
    """
    fig = plt.figure(ifig,figsize=(uw*nw,uh*nh))

    hsum = h0 + hs + h1
    h0 = h0/hsum; hs = hs/hsum; h1 = h1/hsum

    wsum = w0 + ws + w1
    w0 = w0/wsum; ws = ws/wsum; w1 = w1/wsum

    dw = (1. -left - right)/nw
    dh = (1. -up - down)/nh

    w0 = w0 * dw
    ws = ws * dw
    w1 = w1 * dw

    h0 = h0 * dh
    hs = hs * dh
    h1 = h1 * dh

    iax=-1
    if iarange==False:
        for j in range(nh):
            b = down + h0 + dh * j
            d = hs
            for i in range(nw):
                iax=iax+1
                a = left + w0 + dw * i
                c = ws
                if nfig!=None and nfig==len(fig.axes): pass
                else:
                    fig.add_axes([a,b,c,d]) # w0, h0, w, h
                    cax=fig.axes[iax] # current axes
                    cax.ticklabel_format(useOffset=useOffset)
                    ticks_bins(ax=cax,axis='x',n=4)
                    ticks_bins(ax=cax,axis='y',n=4)

    elif iarange:
        for j in range(nh):
            b = down+h0+dh*(nh-j-1)
            d = hs
            for i in range(nw):
                iax=iax+1
                a = left + w0 + dw * i
                c = ws
                if nfig!=None and nfig==len(fig.axes): pass
                else:
                    fig.add_axes([a,b,c,d]) # w0, h0, w, h
                    cax=fig.axes[iax] # current axes
                    ticks_bins(ax=cax,axis='x',n=4)
                    ticks_bins(ax=cax,axis='y',n=4)

    return fig

def axes3():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax=fig.add_subplot(111, projection='3d')
    return ax

def tune_xy_lim(axs):
    """ tune both x and y """
    tune_x_lim(axs,axis='x')
    tune_x_lim(axs,axis='y')

def tune_x_lim(axs,axis='x'):
    """ mpl.axs,  axis='x' or 'y' """
    X0 = None; X1 = None
    for i in range(len(axs)):
        if axis=='x': x0,x1 = axs[i].get_xlim()
        if axis=='y': x0,x1 = axs[i].get_ylim()
        if X0==None : X0=x0
        if X1==None : X1=x1
        if x0<X0    : X0=x0
        if x1>X1    : X1=x1
    for i in range(len(axs)):
        if axis=='x': axs[i].set_xlim(X0,X1)
        if axis=='y': axs[i].set_ylim(X0,X1)

def tune_x_lim_u(axs):
    """ Tune x and y to match the maximum accounting for both """
    mx = None; mn = None
    for i in range(len(axs)):
        x0,x1 = axs[i].get_xlim()
        y0,y1 = axs[i].get_ylim()
        if mx==None:
            if x1>=y1: mx=x1
            if y1> x1: mx=y1
        if mn==None:
            if x0<=y0: mn = x0
            if y1< y1: mn = y1
        if x0<=y0: _xn_ = x0
        if y0< x0: _xn_ = y0
        if x1>=y1: _xm_ = x1
        if y1> x1: _xm_ = y1
        if _xn_<mn: mn = _xn_
        if _xm_>mx: mx = _xm_
    for i in range(len(axs)):
        axs[i].set_xlim(mn,mx)
        axs[i].set_ylim(mn,mx)

def norm_cmap(mx,mn,val=None,cm_name='jet',inorm=False):
    """
    Arguments
    =========
    mx
    mn
    val
    cm_name = 'gist_rainbow'
            = 'jet'
    inorm

    Returns
    =======
    1. if val!=None: return cmap, m.to_rgba(val)
    2. if val==None and inorm==False, return cmap, m
    3. if val==None and inorm==True , return cmap, m, norm
    """
    import matplotlib    as mpl
    import matplotlib.cm as cm
    norm = mpl.colors.Normalize(vmin=mn,vmax=mx)
    cmap = cm.get_cmap(cm_name)
    m    = cm.ScalarMappable(norm=norm, cmap=cmap)
    if val==None and inorm==False: return cmap, m
    if val==None and inorm==True:  return cmap, m, norm
    else: return cmap, m.to_rgba(val)
    raise IOError

def add_cb(ax,cmap=None,spacing='proportional',filled=True,
           format='%2i',levels=None,colors=None,norm=None,
           ylab=None, xlab=None,lw=4):
    """
    Arguments
    ---------
    ax
    cmap=None
    spacing='proportional'
    filled=True,
    format='%2i'
    levels=None
    colors=None,norm=None,
    ylab=None
    xlab=None
    lw=4
    """
    import matplotlib as mpl
    import numpy as np
    if cmap==None: print('Warning: color map was not specified.')
    cb = mpl.colorbar.ColorbarBase(
        ax,cmap=cmap,spacing=spacing,
        norm=norm,filled=filled,format=format)
    ## cb.set_ticks(np.arange(0,90.01,15))
    if type(levels)!=type(None):
        cb.add_lines(
            levels=levels,colors=colors,
            linewidths=np.ones(len(colors))*lw,
            erase=True)

    if type(ylab)!=type(None): ax.set_ylabel(ylab,dict(fontsize=15))
    if type(xlab)!=type(None): ax.set_xlabel(xlab)

    return cb

# Topics: line, color, LineCollection, cmap, colorline, codex

'''
Defines a function colorline that draws a (multi-)colored 2D line with coordinates x and y.
The color is taken from optional data in z, and creates a LineCollection.

z can be:
- empty, in which case a default coloring will be used based on the position along the input arrays
- a single number, for a uniform color [this can also be accomplished with the usual plt.plot]
- an array of the length of at least the same length as x, to color according to this data
- an array of a smaller length, in which case the colors are repeated along the curve

The function colorline returns the LineCollection created, which can be modified afterwards.

See also: plt.streamplot
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

# Data manipulation:
def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

# Interface to LineCollection:
def colorline(ax, x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    ax.add_collection(lc)
    return lc

def clear_frame(ax=None):
    # Taken from a post by Tony S Yu
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

#-------------------------------------------------------------#
def color_map(mn=0,mx=1.,cmap='viridis'):
    """
    Color mapper that linearly translates values in (mn,mx)
    to colors determined by a chosen matplotlib color-map

    Arguments
    ---------
    mn
    mx
    cmap

    Return
    ------
    Color map function
    """
    cmap, m = norm_cmap(
        mn=mn,mx=mx,cm_name=cmap)
    return m.to_rgba
