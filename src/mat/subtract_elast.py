## subtract elastic portion of strain from total multi strain.

def bi_axial_iso(eps11,eps22,sig11,sig22,
                 M=2.e9,nu=0.3,):
    """
    Biaxial, isotropic

    M   = 2.e9 (elastic modulus)
    nu  = 0.3  (poisson ratio)

    E_11^el = 1/M (S_11 - nu*S_22)
    E_22^el = 1/M (S_22 - nu*S_11)

    """
    E11 = 1./M * (sig11-nu*sig22)
    E22 = 1./M * (sig22-nu*sig11)
    return E11, E22
