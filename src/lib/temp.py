"""
Functions to generate temporary files and
find a proper location to generate such files
that are expected to be flushed - like /tmp/ folder
in Unix/Linux


<find_tmp> finds and return the path meant for temporary I/O
operations in several computating resources available to me.
e.g., Palmetto or my Mac.

<gen_tempfile> generates a filename suitable for temporary
I/O operation. If <tmp> argument to <gen_tempfile> is not given
it finds the suitable temp folder using <find_tmp>

<gen_tempfolder> generates a folder suitable for temporary
I/O operation similar to <gen_tempfile>


Youngung Jeong
youngung.jeong@gmail.com
"""
import os

def find_writable(*paths):
    """
    Find and return a writable path among the given paths.
    It returns the writable path as soon as it finds it.
    """
    for path in paths:
        if os.access(path,os.W_OK):
            return path

    print('No writable folder found among the given list below')
    print(paths)
    raise IOError

def find_tmp(verbose=False):
    """
    Find the relevant temp folder
    in compliance with the CTCMS cluster policy,
    The rule is if there's /data/
    create files there and run vpsc there.

    Argument
    --------
    verbose = False

    Returns
    -------
    _tmp_
    """
    if os.name=='nt':
        from tempfile import mkdtemp
        _tmp_=mkdtemp()
    else:
        if 'TMPDIR' in os.environ:
            _tmp_=os.environ['TMPDIR']
        else:
            _tmp_='/tmp/'

    date = os.popen('date +%Y%m%d_%H%M%S').read().split('\n')[0]
    _tmp_ = os.path.join(_tmp_,date)

    if not(os.path.isdir(_tmp_)):
        os.mkdir(_tmp_)
    return _tmp_

    # username = os.popen('whoami').read().split('\n')[0]
    # userhome = os.environ['HOME']

    # ## Find local folder that allows fast I/O condition

    # if os.path.isdir(os.path.join(os.sep,'scratch',username)):
    #     _tmp_ = os.path.join(os.sep,'scratch',username)
    #     date=os.popen('date +%Y%m%d_%H%M%S').read().split('\n')[0]
    #     _tmp_ = os.path.join(_tmp_,date)
    # elif os.path.isdir(os.path.join(userhome,'mnt','dummy')):
    #     _tmp_ = os.path.join(userhome,'mnt','dummy')
    # elif os.path.isdir(os.path.join(os.sep,'media','youngung','Maxtor Desktop','youngung_scratch')) and\
    #      username=='youngung':
    #     _tmp_ = os.path.join(os.sep,'media','youngung','Maxtor Desktop','youngung_scratch')
    #     date=os.popen('date +%Y%m%d_%H%M%S').read().split('\n')[0]
    #     _tmp_ = os.path.join(_tmp_,date)
    # else:
    #     if os.path.isdir('/local_scratch/'): ## Palmetto@Clemson
    #         ## check if permission to write is available.
    #         _tmp_ = find_writable(
    #             '/local_scratch','/scratch1/younguj','/scratch2/younguj',
    #             '/scratch3/younguj')

    #     elif os.path.isdir('/data/'): ## CTCMS cluster@NIST
    #         _tmp_='/data/ynj/scratch/'
    #     else: ##
    #         _tmp_='/tmp/'
    #         ## Append user name
    #     _tmp_ = os.path.join(_tmp_,username)
    #     if not(os.path.isdir(_tmp_)):
    #         os.mkdir(_tmp_)
    #     date = os.popen('date +%Y%m%d_%H%M%S').read().split('\n')[0]
    #     _tmp_ = os.path.join(_tmp_,date)
    #     if not(os.path.isdir(_tmp_)):
    #         os.mkdir(_tmp_)

    # if not(os.path.isdir(_tmp_)):
    #     os.mkdir(_tmp_)
    # if verbose:print('_tmp_:%s'%_tmp_)
    # return _tmp_

def gen_tempfolder(prefix='',affix='',tmp=None):
    """
    Create temp folder using tempfile.mkdtemp

    Arguments
    ---------
    prefix
    affix
    tmp    - tmp directory suitable for I/O
             If not given, find one using <find_tmp>

    Returns
    -------
    tempDirectory
    """
    import tempfile
    if type(tmp).__name__=='NoneType':
        tmp = find_tmp(verbose=False)

    return tempfile.mkdtemp(prefix=prefix,suffix=affix,dir=tmp)

def gen_tempfile(prefix='',affix='',ext='txt',i=0,tmp=None):
    """
    Generate temp file in _tmp_ folder.
    Unless <tmp> argument is specified, the _tmp_ folder
    is determined by <def find_tmp> function

    Arguments
    ---------
    prefix = ''
    affix  = ''
    ext    = 'txt'  (extension, defualt: txt)
    i      : an integer to avoid duplicated name
           (may be deprecated since gen_hash_code2 is used...)
    tmp = None

    Return
    ------
    filename
    """
    import os
    from .etc import gen_hash_code2

    if prefix=='':
        prefix = gen_hash_code2(nchar=6)
    if affix=='':
        affix = gen_hash_code2(nchar=6)

    if type(tmp).__name__=='NoneType':
        tmp = find_tmp(verbose=False)
    exitCondition = False
    it = 0
    while not(exitCondition):
        hc = gen_hash_code2(nchar=6)
        tmpLocation = find_tmp(verbose=False)
        if len(affix)>0: filename = '%s-%s-%s'%(prefix,hc,affix)
        else:            filename = '%s-%s'%(prefix,hc)
        if type(ext).__name__=='str':
            filename = '%s.%s'%(filename,ext)

        ## under the temp folder
        filename = os.path.join(tmp,filename)
        exitCondition = not(os.path.isfile(filename))
        it = it + 1
        if it>100: exitCondition=True

    if it>1:
        print(('Warning: Oddly you just had'+\
            ' an overlapped file name'))
    return filename
