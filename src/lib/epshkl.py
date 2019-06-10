## lanl formatted LAT STR plotter
from .diff_info import main as info
def main_ph(ref=None,ixy=0,ax=None):
    """
    ixy = 0
       phase elastic strain (x), macrostress (y)
    ixy = 1
       macrostress (x), phase elastic strain (y)
    ixy = 2
       macrostrain (x), phase elastic strain (y)

    macro values are given as ref from ax_ref method
    """
    from . import mpl_lib, dat_lib, axes_label
    import mech

    ax, ref = ax_ref(ax,ref,ixy)

    fdif = info(iopt=2)
    nph = len(fdif)

    ph_lat_str=[]
    for i in range(nph):
        f=mech.FlowCurve(name='phase_specific')
        f.get_pmodel_lat(fn='ph_str_%i.out'%(i+1))
        ph_lat_str.append(f)

        if ixy==0:
            y = ref
            x_l = ph_lat_str[i].epsilon[0,0]
            x_t = ph_lat_str[i].epsilon[1,1]
            y_l,x_l = dat_lib.trim(y,ref=x_l)
            y_t,x_t = dat_lib.trim(y,ref=x_t)
            x = x_l
            y = y_l
        elif ixy==1:
            x = ref
            y = ph_lat_str[i].epsilon[0,0]

        elif ixy==2:
            x = ref
            y = ph_lat_str[i].epsilon[0,0]
        else:
            raise IOError('Unavailable ixy')
        ax.plot(x,y)
        axes_label.__ph__(ax=ax,ft=15,iopt=ixy)

def main_hkl(ref=None,ixy=0,dr=2,ax=None):
    """
    ixy=0
        macrostress(x), lattice strain(y)
    ixy=1
        lattice strain(x), macrostress(y)
    ixy=2
        macrostrain (x), lattice strain(y)
    ixy=3
        lattice strain(x), macrostrain (y)
    """
    from . import mpl_lib, dat_lib, axes_label
    fdif = info(iopt=2)
    nph = len(fdif)

    ax, ref = ax_ref(ax,ref,ixy)

    ls = ['-','--','-.']
    c  = ['r','g','b','m']
    for iph in range(nph):
        hkls, chi_eta = read_dif(difile=fdif[iph])
        np = len(hkls)
        ax_ph = []
        for ip in range(np): # hkl
            hkl = hkls[ip]
            ndir = len(chi_eta[ip])
            for idir in range(ndir):
                if idir!=dr:
                    pass
                else:
                    lat = read_lat(hkl=hkl,difile=fdif[iph],iph=iph+1,
                                   idir=idir+1)
                    label='(%i%i%i)'%(hkl[0],hkl[1],hkl[2])
                    if ixy==0 or ixy==2:
                        x=ref
                        y=lat
                        x,y=dat_lib.trim(x,ref=y)
                    if ixy==1 or ixy==3:
                        x=lat
                        y=ref
                        y,x=dat_lib.trim(y,ref=x)
                    ax.plot(x,y,ls=ls[iph],color=c[ip])
        axes_label.__ehkl__(ax=ax,ft=15,iopt=ixy)
        pass

def main_dep(ref=None,ixy=0,ndir=2):
    from . import mpl_lib
    from . import dat_lib
    from . import axes_label
    fdif = info(iopt=2)
    nph = len(fdif)

    if ndir==None: nw=3
    elif ndir!=None: nw = ndir

    fig=mpl_lib.wide_fig(nw=nw,nh=nph,
                         w1=0,w0=0,
                         uh=3,
                         h0=0.25,h1=0.25,
                         left=0.15, right=0.05,
                         down=0.05, up=0.05)
    axs = fig.axes
    if ref==None:
        import mech
        macro = mech.FlowCurve(name='model')
        macro.get_model(fn='STR_STR.OUT')
        ref = macro.sigma[0,0]

    ax_dir=[]
    for iph in range(nph):
        hkls, chi_eta = read_dif(difile=fdif[iph])
        np = len(hkls)
        ax_ph = []
        for ip in range(np):
            hkl = hkls[ip]
            if ndir==None: ndir = len(chi_eta[ip])
            for idir in range(ndir):
                lat = read_lat(hkl=hkl,difile=fdif[iph],iph=iph+1,
                               idir=idir+1)
                iax = idir + nw*iph
                ax = axs[iax]
                if ip==0:
                    ax_ph.append(ax)
                if idir==0 and ip==0:
                    ax_dir.append(ax)

                label='(%i%i%i)'%(hkl[0],hkl[1],hkl[2])
                if ixy==0:
                    x=ref
                    y=lat
                    x,y=dat_lib.trim(x,ref=lat)
                if ixy==1:
                    x=lat
                    y=ref
                    y,x=dat_lib.trim(y,ref=lat)

                ax.plot(x,y,label=label)

                if idir==0 and ip==np-1:
                    ax.legend(loc='best',
                              fontsize=8,
                              fancybox=True,
                              framealpha=0.5)
                    axes_label.__ehkl__(ax=ax,iopt=ixy,ft=15)
        mpl_lib.tune_x_lim(ax_ph,axis='x')
        mpl_lib.tune_x_lim(ax_dir,axis='y')
        mpl_lib.rm_inner(ax_ph)

    for iax in range(len(axs)):
        axs[iax].grid('on')

def ax_ref(ax,ref,ixy):
    if ax==None:
        fig=mpl_lib.wide_fig(
            nw=1,nh=1,w1=0.8,w0=0.9,ws=5.,
            uh=3,uw=4,h0=0.25,h1=0.25,
            left=0.15, right=0.05,
            down=0.05, up=0.05)
        ax = fig.axes[0]
    if ref==None:
        import mech
        macro = mech.FlowCurve(name='model')
        macro.get_model(fn='STR_STR.OUT')
        if ixy==0 or ixy==1:
            ref = macro.sigma[0,0]
        if ixy==2 or ixy==3:
            ref = macro.epsilon[0,0]
    return ax, ref

def read_dif(difile='examples/ex07_TRIP/aust.dif'):
    from intstr import __difreader__ as difreader
    hkls, chi_eta,dum=difreader(filename=difile)
    return hkls, chi_eta

def read_lat(hkl=[1,1,1],
             difile='examples/ex07_TRIP/aust.dif',
             iph=1,
             idir=1):
    from intstr import epshkl
    accstr, epshkl, chi_eta = epshkl(
        hkl=hkl,difile=difile,iph=iph)
    chi_eta = chi_eta.tolist()
    c_e     = idir_to_ce(idir)
    ind     = chi_eta.index(c_e)
    return epshkl[ind]

def idir_to_ce(idir=1):
    if idir==1:   c_e = [90., 0.]
    elif idir==2: c_e = [90.,90.]
    elif idir==3: c_e = [ 0., 0.]
    else: raise IOError('Inappropriate idir')
    return c_e

def ce_to_idir(c_e=[90.,0.]):
    if    c_e==[90., 0.]: idir=1
    elif  c_e==[90.,90.]: idir=2
    elif  c_e==[ 0., 0.]: idir=3
    else: raise IOError('Inappropriate c_e')
    return idir
