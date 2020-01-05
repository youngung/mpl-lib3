## Find which computer I am on.
import os
pjoin = os.path.join

def find_vpsc_repo():
    """
    Find path to VPSC repository
    """
    if os.name=='posix':
        path_home = os.environ['HOME']
    elif os.name=='nt':
        path_home = os.environ['USERPROFILE']

        if os.environ['USERDOMAIN']=='DESKTOP-67P6BC3':
            return os.path.join(os.environ['USERPROFILE'],'repo','vpsc_plus_e')

    whereami = guessWhereami()

    ## test if repo/vpsc-fld-yld is present
    fn_vpsc_env = os.path.join(path_home,'.vpscyldfld')
    if os.path.isfile(fn_vpsc_env):
        with open(fn_vpsc_env,'r') as fo:
            lines = fo.read().split('\n')
            path_vpsc = lines[0].split('=')[1]
        if not os.path.isdir(path_vpsc):
            raise IOError('Could not find %s'%path_vpsc)
    else:
        if   whereami=='palmetto':
            path_vpsc=pjoin(path_home,'repo','vpsc-fld')
        elif whereami=='mac':
            path_vpsc=pjoin(path_home,'repo','vpsc','vpsc-dev-fld')
        elif whereami=='mbp':
            path_vpsc=pjoin(path_home,'repo','vpsc-fld-yld')
        elif whereami=='ubuntu@mml':
            path_vpsc=pjoin(path_home,'repo','vpsc-fld-yld')
        elif whereami=='hg@ubuntu':
            path_vpsc=pjoin(path_home,'vpsc')
        else:
            raise IOError('Could not find vpsc repository')
    return path_vpsc

def clues():
    from platform import platform
    if platform()[:6]=='Darwin': return 'Darwin'
    elif platform()[:5]=='Linux': return 'Linux'
    raise IOError

def guessWhereami():
    """
    Determine where am I based on the username
    returned by <whoami>

    Returned locations are all in lowercase.
    if couldn't find, 'unknown' is returned.
    """
    ## add more IDs - locations all in lowercase
    userIDs = dict(younguj='palmetto',yj='mac',youngung='mbp',hwigeon='hg@ubuntu')#,young='desktop-67p6bc3')
    p = os.popen('whoami')
    whoami=p.read().split('\n')[0]
    if whoami in list(userIDs.keys()):
        if whoami=='youngung': ## either my mbp or ubuntu@mml
            path_home = os.environ['HOME']
            if path_home==pjoin(os.sep,'Users','youngung'):
                whereami='mbp'
            elif path_home==pjoin(os.sep,'home','youngung'):
                whereami='ubuntu@mml'
            else:
                raise IOError('Did not expect this case in whichcomp')
            pass
        else:
            whereami=userIDs[whoami]
            pass
        pass
    else:
        whereami ='unknown'
        pass
    return whereami

## more environmental options
def determineEnvironment(whereami=guessWhereami()):
    if whereami=='palmetto':
        submitCommand = 'qsub'
    else:
        submitCommand = None
    from MP.lib import checkX
    if checkX.main()!=0:
        availX = False
    else:
        availX = True
    return submitCommand, availX
