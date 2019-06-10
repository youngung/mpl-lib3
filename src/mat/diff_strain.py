"""
Module to process LAT_STR_PH?.OUT file generated by EVPSC calculations

Youngung Jeong, PhD
Changwon National University

yjeong@changwon.ac.kr
"""

def reader_evpsc(fn='/var/folders/4b/vr51t7wn7vb54bdvbmmgfjnw0000gn/T/20170927_173458/vpsc-histsiUpm84/LAT_STR_PH1.OUT'):
    """
    Arguments
    ---------
    fn
    """
    dat=np.loadtxt(fn,skiprows=1).T
