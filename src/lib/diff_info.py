def main(iopt=0):
    """
    returns names of files that contain
    initial condition of the simulation based on
    current 'EVPSC.IN' in the base folder
    """
    from saveout import find_files
    ftex,fsx,fdif_ = find_files(iopt=1)
    nph = len(ftex)
    print('%i phase(s) exist'%(nph))

    fdif=[]
    for i in range(len(fdif_)):
        fdif.append(fdif_[i][:-1])

    if iopt==0: return ftex
    if iopt==1: return fsx
    if iopt==2: return fdif
