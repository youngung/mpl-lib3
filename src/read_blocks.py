import numpy as np
def main(fn='STR_STR.OUT',skiprows=1):
    """
    Read (E)VPSC output files with string heads are inserted arbitrary.

    1) In case that no heads are inserted use np.loadtxt
    2) In case that head are inserted there are two options:
       2-1) Either completely ignore the appearance of the blocks
       2-2) Or, use the block as the indication of separable data blocks
    """
    try: 
        # in case that no heads are inserted:
        return np.loadtxt(fn,skiprows=skiprows).T
    except ValueError:
        return rb(fn,skiprows)

def read_tx(fn='TEX_PH1.OUT'):
    """
    Read a block of 'TEX_PH?.OUT'

    Arguments
    ---------
    'TEX_PHT1.OUT'

    Returns
    -------
    pxs   : pxs[nb,4,ngrs]
    ngrs
    """
    f=open(fn,'r')
    d=f.read()
    blocks=d.split('B ')
    blocks=blocks[1:]
    nb=len(blocks)
    px=[]

    MXGR=0
    ngrs=[]
    for ib in range(nb):
        b=blocks[ib]
        ngr=int(b.split('\n')[0])
        if ngr>MXGR:MXGR=ngr
        ngrs.append(ngr)

    pxs=np.zeros((nb,4,ngr))

    for ib in range(nb):
        b = blocks[ib]
        lines = b.split('\n')
        igr=0
        for il in range(len(lines)):
            grain=lines[il].split()
            if len(grain)==4:
                ph1,ph,ph2,wgt=list(map(float,grain))
                pxs[ib,:,igr]=[ph1,ph,ph2,wgt]
                igr=igr+1
    return pxs,ngrs

def rb(fn,skiprows):
    """
    """
    lines = open(fn,'r').readlines()[skiprows:]
    nrow = len(lines[0].split())
    ncol = 0

    i = 0
    D = []

    while True:
        try:
            dat = list(map(float,lines[i].split()))
        except ValueError: pass
        except IndexError: break
        else:
            if len(dat)!=0:
                if len(D)==0: ncol = len(dat)
                elif len(D)>0:
                    if ncol!=len(dat):raise IOError('Number of columns are not equal??')
                D.append(dat)
            elif len(dat)==0: break
        i = i + 1
    return np.array(D).T

def reader_nc(fn='sf_unload_ph1.out',nc=481):
    """
    Read lines that match the specified number of column (nc)
    """
    lines=open(fn,'r').readlines()
    rst_lines=[]

    for i in range(len(lines)):
        try: n0=len(list(map(float,lines[i].split())))
        except: 
            print('Line %i is not pure float'%i)
            pass
        else:
            if n0==nc:
                rst_lines.append(list(map(float,lines[i].split())))
            else:
                print('Line %i is not matched'%i) 
    return rst_lines
def _count_nc_(fn):
    f=open(fn,'r')
    l0=f.readline()
    l1=f.readline()
    l3=f.readline()
    f.close()
    n0 = len(l0.split())
    n1 = len(l1.split())
    nc = len(l3.split())
    return n0, n1, nc

def _block_starter_(fn='sf_unload_ph1.out',nc=481):
    lines=open(fn,'r').readlines()

    l0=[]
    for i in range(len(lines)):
        if len(lines[i].split())!=481:
            l0.append(i)
    return l0
            
def read_sf_scan(fn='sf_unload_ph1.out'):
    """
    fn ='sf_unload_ph1.out'
                 or
        'sf_ph1.out'          done during uniaxial loading
    """
    lines = open(fn,'r').readlines()
    n0,n1,nc= _count_nc_(fn)
    l0      = _block_starter_(fn,nc)

    # dat blocks
    blocks=[]
    for i in range(len(l0)-1):
        i0 = l0[i]
        i1 = l0[i+1]
        #print i0,i1
        blocks.append(lines[i0:i1])
    sig = []
    eps = []

    print('# of blocks:', len(blocks))

    for ib in range(len(blocks)):
        nstp = len(blocks[ib])-2
        sin2psi = list(map(float,blocks[ib][0].split()[7:]))
        sigma = np.zeros((6,nstp))
        ehkl  = np.zeros((nc-6-1,nstp))
        for istp in range(nstp):
            sigma[:,istp] = list(map(float,blocks[ib][istp+2].split()[1:7]))
            ehkl[:,istp]  = list(map(float,blocks[ib][istp+2].split()[7:]))
        sig.append(sigma)
        eps.append(ehkl)
    return sig,eps
