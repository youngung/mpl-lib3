"""
mech.py module


Youngung Jeong, PhD

Department of Materials Science and Engineering
Changwon National University, Korea

yjeong@changwon.ac.kr
"""
import numpy as np
from scipy import integrate
cumtrapz=integrate.cumtrapz

class FlowCurve:
    """
    Flow characteristic in full 3x3 dimensions
    """
    def __init__(self,name=None,description=None):
        """        """
        self.nstp = 0
        self.sigma = np.zeros((3,3,self.nstp))*np.nan
        self.epsilon = np.zeros((3,3,self.nstp))*np.nan
        self.flag_sigma = np.zeros((3,3))
        self.flag_epsilon = np.zeros((3,3))
        self.flag_6s    = np.zeros((6,))
        self.flag_6e    = np.zeros((6,))

        self.is_stress_available = False
        self.is_strain_available = False

    # Voigt vectorial nomenclature (0 - 5)
        self.vo = [[0,0],[1,1],[2,2],[1,2],[0,2],[0,1]]

        self.ivo = np.ones((3,3),dtype='int')
        for i in range(3):
            self.ivo[i,i] = i

        self.ivo[1,2] = 3
        self.ivo[2,1] = 3
        self.ivo[0,2] = 4
        self.ivo[2,0] = 4
        self.ivo[0,1] = 5
        self.ivo[1,0] = 5

        self.name = name
        self.descr = description

    def Decompose_SA(self,a):
        """
        Decompose VG into sym and asym parts

        Argument
        --------
        a (array in 3x3)

        Returns
        -------
        sym
        asym
        """
        sym  = 1./2. * (a+a.T)
        asym = 1./2. * (a-a.T)
        return sym, asym

    def conv6_to_33(self,a):
        """
        Array <a> in 6 D

        Argument
        --------
        a  Array
        """
        a=np.array(a)
        if a.shape[0]!=6: raise IOError('Array should be (6)')
        b=np.zeros((3,3))
        b[0,0]=a[0] # 11
        b[1,1]=a[1] # 22
        b[2,2]=a[2] # 33
        b[1,2]=a[3] # 23
        b[2,1]=a[3]
        b[0,2]=a[4] # 13
        b[2,0]=a[4]
        b[0,1]=a[5] # 12
        b[1,0]=a[5]
        return b

    def conv9_to_33(self,a):
        """
        array <a> in 9

        Argument
        --------
        a  array
        """
        a=np.array(a)
        if a.shape[0]!=9: raise IOError('array should be (9)')
        b=np.zeros((3,3))
        b[0,0] = a[0]
        b[0,1] = a[1]
        b[0,2] = a[2]
        b[1,0] = a[3]
        b[1,1] = a[4]
        b[1,2] = a[5]
        b[2,0] = a[6]
        b[2,1] = a[7]
        b[2,2] = a[8]
        return b

    def get_eqv(self):
        """
        equivalent scholar value that represents
        the full tensorial states
        """
        if self.is_stress_available and \
           self.is_strain_available:
            self.get_energy()
            self.get_vm_stress()
            self.epsilon_vm = self.w/self.sigma_vm

            # print 'VM stress:', self.sigma_vm
            # print 'VM strain:', self.epsilon_vm

        elif self.is_stress_available and \
             not (self.is_strain_available):
            self.get_vm_strain()

    def get_energy(self):
        """
        Integrate \int EijSij dEij
        """
        w = 0
        for i in range(3):
            for j in range(3):
                w = w + self.epsilon[i,j] * self.sigma[i,j]
        self.w = w

    def get_deviatoric_stress(self):
        self.sigma_dev = np.zeros(self.sigma.shape)
        ijx = np.identity(3)
        hydro = 0.
        for i in range(3):
            hydro = hydro + self.sigma[i,i]
        for i in range(3):
            for j in range(3):
                self.sigma_dev[i,j] = self.sigma[i,j]\
                                      - 1./3. * hydro * ijx[i,j]

    def get_deviatoric_strain(self):
        self.epsilon_dev = np.zeros(self.epsilon.shape)
        ijx = np.identity(3)
        vol = 0.
        for i in range(3):
            vol = vol + self.epsilon[i,i]
        for i in range(3):
            for j in range(3):
                self.epsilon_dev[i,j] = self.epsilon[i,j]\
                                        - 1./3. * vol * ijx[i,j]

    def get_vm_stress(self):
        """
        Get Von Mises equivalent stress
        """
        self.get_deviatoric_stress()
        vm = 0.
        for i in range(3):
            for j in range(3):
                vm = vm + self.sigma_dev[i,j]**2
        vm = 3./2. * vm
        self.sigma_vm = np.sqrt(vm)

    def get_vm_strain(self):
        """
        Note that VM strain might be calculated based on
        the principle of the plastic-work equivalence,
        i.e., W = SijEij = S(VM) E(VM)

        However, there are times only
        strain is available whereas stress isn't thus
        not able to calculate the 'work'
        This function is intended to be used in such a case.
        """
        self.get_deviatoric_strain()
        vm = 0.
        for i in range(3):
            for j in range(3):
                vm = vm + self.epsilon_dev[i,j]**2
        vm = 2./3. * vm
        self.epsilon_vm = np.sqrt(vm)
        self.nstp = len(self.epsilon_vm)

    def get_principal(self):
        """
        Calculate principal stresses and principal strains (eignvalues)
        """
        from numpy import linalg as LA
        self.sigma_princ_val=np.zeros((3,self.nstp))
        self.sigma_princ_vec=np.zeros((3,3,self.nstp))
        self.epsilon_princ_val=np.zeros((3,self.nstp))
        self.epsilon_princ_vec=np.zeros((3,3,self.nstp))
        self.edot_princ_val=np.zeros((3,self.nstp))
        self.edot_princ_vec=np.zeros((3,3,self.nstp))
        for istp in range(self.nstp):
            ## not necessarily order results.
            # stress
            w, v = LA.eig(self.sigma[:,:,istp])
            ind = np.argsort(w)[::-1]
            self.sigma_princ_val[:,istp] = w[ind]
            self.sigma_princ_vec[:,:,istp] = v[:,ind]

            # strain
            w, v = LA.eig(self.epsilon[:,:,istp]) # use of 'accumulated strain'
            ind = np.argsort(w)[::-1]
            self.epsilon_princ_val[:,istp] = w[ind]
            self.epsilon_princ_vec[:,:,istp] = v[:,ind]

            # strain rate
            w, v = LA.eig(self.d33[:,:,istp]) # use of 'accumulated strain'
            ind = np.argsort(w)[::-1]
            self.edot_princ_val[:,istp] = w[ind]
            self.edot_princ_vec[:,:,istp] = v[:,ind]

    def get_principal_inplane(self):
        """
        Calculate principal stresses and principal strains using only in-plane components.
        """
        from numpy import linalg as LA
        self.sigma_princ_val_inplane=np.zeros((2,self.nstp))
        self.sigma_princ_vec_inplane=np.zeros((2,2,self.nstp))
        self.epsilon_princ_val_inplane=np.zeros((2,self.nstp))
        self.epsilon_princ_vec_inplane=np.zeros((2,2,self.nstp))
        self.edot_princ_val_inplane=np.zeros((2,self.nstp))
        self.edot_princ_vec_inplane=np.zeros((2,2,self.nstp))
        for istp in range(self.nstp):
            ## not necessarily order results.
            # stress
            w, v = LA.eig(self.sigma[:2,:2,istp])
            ind = np.argsort(w)[::-1]
            self.sigma_princ_val_inplane[:,istp] = w[ind]
            self.sigma_princ_vec_inplane[:,:,istp] = v[:,ind]

            # strain
            w, v = LA.eig(self.epsilon[:2,:2,istp]) # use of 'accumulated strain'
            ind = np.argsort(w)[::-1]
            self.epsilon_princ_val_inplane[:,istp] = w[ind]
            self.epsilon_princ_vec_inplane[:,:,istp] = v[:,ind]

            # strain rate
            w, v = LA.eig(self.d33[:2,:2,istp]) # use of 'accumulated strain'
            ind = np.argsort(w)[::-1]
            self.edot_princ_val_inplane[:,istp] = w[ind]
            self.edot_princ_vec_inplane[:,:,istp] = v[:,ind]

    def get_mohr_coulomb(self,c1,c2):
        """
        Calculate parameters associated with Mohr-Coulomb fracture criterion

        Ref:
        2010 Int. J. Fract (2010) 161:1-20, Yuanli Bai and Tomasz Wierzbicki

        Arguments
        ---------
        c1
        c2
        """
        from numpy import linalg as LA
        calc_det=LA.det

        from scipy import interpolate
        self.get_deviatoric_stress()

        self.sigma_dev ## deviatoric stress
        self.j2=np.zeros(self.sigma_dev.shape[-1])
        self.j3=np.zeros(self.sigma_dev.shape[-1])

        for k in range(self.nstp):
            dum=0.
            for i in range(3):
                for j in range(3):
                    dum=dum+self.sigma_dev[i,j,k]*self.sigma_dev[i,j,k]
            self.j2[k]=0.5*dum
            self.j3[k]=calc_det(self.sigma_dev[:,:,k])

        sigma_mean = self.get_pressure()
        p = sigma_mean*(-1)   # Eq (1)
        q = self.sigma_vm     # Eq (2)
        r = self.get_r()      # Eq (3)
        self.get_principal()
        self.eta = sigma_mean/q    # Eq (5)

        self.lode=np.zeros(self.nstp)
        for k in range(self.nstp):
            dum1=3*np.sqrt(3.)/2.*self.j3[k]/(self.j2[k]**(3./2.))
            dum=1./3.*np.arccos(dum1)
            if np.isnan(dum):
                print('Warning: nan was found.')
                print('dum1:',dum1)
                print('self.j3[k]:',self.j3[k])
                print('self.j2[k]:',self.j2[k])
                print('self.sigma_dev[k]:',self.sigma_dev[:,:,k])

            self.lode[k]=dum

        ksi=(r/q)**3.         # Eq (6)
        th=np.arccos(ksi)/3.  # Eq (6)
        ## Lode angle parameter
        thbar=1.-6*th/np.pi
        # self.lode=thbar

        if False:
            print('p[-1]:', p[-1])
            print('q[-1]:', q[-1])
            print('r[-1]:', r[-1])
            print('eta[-1]:', self.eta[-1])
            print('ksi[-1]:', ksi[-1])
            print('th[-1]:', th[-1])
            print('thbar[-1]:', thbar[-1])

        ## Calc. tau and sn
        tau,sn = self.get_mc_tau_sn(c1) # Eqs (10) and (11)
        self.c2_hist = tau+c1*sn        # Eqs (12) ... and Eq. (15)?

        # print '--------------------------------'
        # print '** minimum self.c2_hist: %4.1f'%min(self.c2_hist)
        # print '** maximum self.c2_hist: %4.1f'%max(self.c2_hist)

        if max(self.c2_hist)<c2:
            print('%f<%f'%(max(self.c2_hist),c2))
            print('%f>%f'%(min(self.c2_hist),c2))
            print('** Warning: C2 exceeds the maximum of calculated c2 values')

        if min(self.c2_hist)>c2:
            print('%f>%f'%(min(self.c2_hist),c2))
            print('** Warning: C2 is lower than the minimum of calculated c2 values')

        if max(self.c2_hist)<c2 or min(self.c2_hist)>c2:
            stress=np.zeros(6)
            strain=np.zeros(6)
            stress[:]=np.nan
            strain[:]=np.nan
            svm_fract, evm_fract = np.nan, np.nan
            p, q, r, eta, ksi, thbar = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            return stress, strain, evm_fract, evm_fract, p, q, r, eta, ksi, thbar

        ## Obtain stress, strain data by interpolating their evolutions to the given c2 value.
        #--- construct interpolant functions.
        #- stress
        f_s11=interpolate.interp1d(self.c2_hist,self.sigma[0,0])
        f_s22=interpolate.interp1d(self.c2_hist,self.sigma[1,1])
        f_s33=interpolate.interp1d(self.c2_hist,self.sigma[2,2])
        f_s23=interpolate.interp1d(self.c2_hist,self.sigma[1,2])
        f_s13=interpolate.interp1d(self.c2_hist,self.sigma[0,2])
        f_s12=interpolate.interp1d(self.c2_hist,self.sigma[0,1])
        f_svm=interpolate.interp1d(self.c2_hist,self.sigma_vm)
        #- strain
        f_e11=interpolate.interp1d(self.c2_hist,self.epsilon[0,0])
        f_e22=interpolate.interp1d(self.c2_hist,self.epsilon[1,1])
        f_e33=interpolate.interp1d(self.c2_hist,self.epsilon[2,2])
        f_e23=interpolate.interp1d(self.c2_hist,self.epsilon[1,2])
        f_e13=interpolate.interp1d(self.c2_hist,self.epsilon[0,2])
        f_e12=interpolate.interp1d(self.c2_hist,self.epsilon[0,1])
        f_evm=interpolate.interp1d(self.c2_hist,self.epsilon_vm)

        #- etc.
        f_p    = interpolate.interp1d(self.c2_hist,p)
        f_q    = interpolate.interp1d(self.c2_hist,q)
        f_r    = interpolate.interp1d(self.c2_hist,r)
        f_eta  = interpolate.interp1d(self.c2_hist,self.eta)
        f_ksi  = interpolate.interp1d(self.c2_hist,ksi)
        f_thbar = interpolate.interp1d(self.c2_hist,thbar)

        s11=f_s11(c2)
        s22=f_s22(c2)
        s33=f_s33(c2)
        s23=f_s23(c2)
        s13=f_s13(c2)
        s12=f_s12(c2)
        svm_fract=f_svm(c2)

        e11=f_e11(c2)
        e22=f_e22(c2)
        e33=f_e33(c2)
        e23=f_e23(c2)
        e13=f_e13(c2)
        e12=f_e12(c2)
        evm_fract=f_evm(c2)

        p = f_p(c2)
        q = f_q(c2)
        r = f_r(c2)
        eta = f_eta(c2)
        ksi = f_ksi(c2)
        thbar = f_thbar(c2)

        return np.array([s11,s22,s33,s23,s13,s12]),\
            np.array([e11,e22,e33,e23,e13,e12]), \
            svm_fract, evm_fract, p, q, r, eta, ksi, thbar

    def get_mc_tau_sn(self,c1):
        v1,v2,v3=self.get_v(c1)

        s1,s2,s3=self.sigma_princ_val[0],self.sigma_princ_val[1],self.sigma_princ_val[2]

        tau=np.sqrt(v1**2*v2**2*(s1-s2)**2+v2**2*v3**2*(s2-s3)**2+v3**2*v1**2*(s3-s1)**2)
        sn =v1**2*s1+v2**2*s2+v3**2*s3
        return tau, sn

    def get_pressure(self):
        pressure=np.zeros(self.nstp)
        for i in range(3):
            pressure=pressure+self.sigma[i,i]
        return pressure/3.

    def get_r(self):
        from numpy import linalg as LA
        r=np.zeros(self.nstp)
        for istp in range(self.nstp):
            det=LA.det(self.sigma_dev[:,:,istp])
            if np.isnan(det):
                print('** Warning: determinant is nan...')
                print('** stress deviator:',self.sigma_dev[:,:,istp])
                r[istp]=np.nan
            elif not(np.isfinite(det)):
                print('** Warning: determinant is not finite...')
                print('** stress deviator:',self.sigma_dev[:,:,istp])
            else:
                r[istp]=(27./2.*det)**(1./3.)

        return r

    def get_v(self,c1):
        """
        Arguments
        ---------
        c1
        """
        v1=(1./(1.+(np.sqrt(1+c1**2)+c1)**2))**0.5
        v2=0.
        v3=(1./(1.+(np.sqrt(1+c1**2)-c1)**2))**0.5
        return v1,v2,v3

    def plot(self,ifig=1):
        import matplotlib.pyplot as plt
        if self.imod=='VPSC':
            fig = plt.figure(ifig)
            ax = fig.add_subplot(111)
            for k in range(6):
                if self.flag_6e[k]==1 and self.flag_6s[k]==1:
                    i,j = self.vo[k]
                    ax.plot(self.epsilon[i,j],self.sigma[i,j],'-x',
                            label='(%i,%i)'%(i+1,j+1))
            ax.legend(loc='best')

        elif self.imod=='EVPSC':
            fig = plt.figure(ifig,figsize=(9,4))
            ax1=fig.add_subplot(121)
            ax2=fig.add_subplot(122)

            ax1.plot(self.epsilon[0,0],self.sigma[0,0],'-x',
                     label='Flow Stress curve')
            ax2.plot(self.epsilon[0,0],self.instR,label='R-value')

    def plot_uni(self,fig=None,**kwargs):
        """
        Arguments
        ---------
        **kwargs - key-worded arguments for plt.plot
        """
        import matplotlib.pyplot as plt
        if type(fig)==type(None):
            fig = plt.figure(figsize=(8,3))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
        else:
            ax1=fig.axes[0]
            ax2=fig.axes[1]

        x=self.epsilon[0,0]
        y=self.sigma[0,0]

        y,x=flip(y,x,mode=0)
        ax1.plot(x,y,**kwargs)
        ax2.plot(x,self.instR,**kwargs)

        ax1.set_xlabel(r'$\bar{\varepsilon}_{11}$')
        ax1.set_ylabel(r'$\bar{\sigma}_{11}$')

        ax2.set_xlabel(r'$\bar{\varepsilon}_{11}$')
        ax2.set_ylabel(r'$R^\mathrm{inst}$')

        plt.tight_layout()

    def get_model(self,fn='STR_STR.OUT'):
        """
        Version 2024 Jan

        Read STR_STR.OUT

        Arguments
        =========
        fn='STR_STR.OUT'

        Read "STR_STR.OUT" that might have 'intermediate headers'
        """
        ncol=None


        ## check the validity of the given file name.
        with open(fn) as f:
            datl = f.read()
            datl = datl.split('\n') ## all the lines.
            if len(datl)<2:
                raise IOError('** Error: too small number of rows')

        ## EVM, SVM, sigma(6), epsilon(6), velgrads(9), tincr, pmac, pwgt, temp, plwork
        alldat=np.loadtxt(fn,skiprows=1).T
        print(f'alldat.shape: {alldat.shape}')

        EVM=alldat[0,:]
        SVM=alldat[1,:]
        self.epsilon=alldat[2:8,:]
        self.sigma=alldat[8:14,:]
        velgrads9=alldat[14:23,:]
        self.tincrs=alldat[23,:]
        self.pmac=alldat[24,:]
        self.pwgt=alldat[25,:]
        self.temp=alldat[26,:]
        self.plwork=alldat[27,:]

        ## post-processing
        self.get_6stress(x=np.array(sigma))
        self.get_6strain(x=np.array(epsilon))
        self.epsilon_vm = EVM[::]
        self.sigma_vm=SVM[::]
        self.w = cumtrapz(y=SVM,x=EVM,initial=0)

        #print(f'self.imod: {self.imod}')

        self.velgrads = np.zeros((3,3,velgrads9.shape[-1]))
        k=0
        for i in range(3):
            for j in range(3):
                self.velgrads[i,j,:] = velgrads9[k,:]
                k=k+1

        ## from vel. gradient, calculate strain rate, spin rate, and inst R value.
        v  = self.velgrads.copy()
        vt = self.velgrads.swapaxes(0,1)
        self.d33 = 0.5 * (v+vt)
        self.w33 = 0.5 * (v-vt)

        ind=~(self.d33[2,2]==0)
        self.instR=np.zeros(len(ind))
        self.instR[~ind]=np.nan
        self.instR[ind] = self.d33[1,1][ind]/self.d33[2,2][ind]


    def get_model_old(self,fn='STR_STR.OUT',iopt=0):
        """
        Version 2015-06

        Read STR_STR.OUT

        Arguments
        =========
        fn='STR_STR.OUT'
        iopt=0 (Full history)
            =1 (Records only at the end of each segment)

        Read "STR_STR.OUT" that might have 'intermediate headers'
        """
        ncol=None
        epsilon=[]
        sigma=[]
        EVM=[]
        SVM=[]
        tincrs=[]
        pmac=[]
        pwgt=[]
        with open(fn) as f:
            datl = f.read()
            datl = datl.split('\n') ## all the lines.
            for i in range(len(datl)):
                l=datl[i]
                if len(l)>2:# or len(l.split())==0: ## eliminate if insufficient datum
                    try:
                        if iopt==0:
                            dat = list(map(float,l.split()))
                        if iopt==1:
                            ## Condition 1
                            # Current line (i) should be strings.
                            dum = l.split()
                            try:
                                float(dum[0])
                            except:
                                pass
                            else:
                                raise IOError

                            ## Condition 2
                            # Previous line i-1 should exist
                            datl[i-1]
                            ## Condition 3
                            ## mappable by float
                            dat = list(map(float,datl[i-1].split()))
                            if len(dat)==0:
                                raise IOError
                    except:
                        pass
                    else:
                        if type(ncol)==type(None):
                            ncol=len(dat)
                            velgrads = []
                            # strain_el=[]
                            # strain_pl=[]
                            strain_tr=[]
                            temps=[]

                        # dat = map(float,l.split())
                        evm,svm=dat[0:2]
                        EVM.append(evm)
                        SVM.append(svm)
                        if ncol in [25,45,17,14]:
                            strain=dat[2:8]
                            stress=dat[8:14]
                        elif ncol==22: ## EPSC4
                            strain=dat[0:6]
                            stress=dat[6:12]

                        epsilon.append(strain)
                        sigma.append(stress)
                        if ncol==25: ## VPSC
                            self.imod='VPSC'
                            tempr = dat[14]
                            v33   = self.conv9_to_33(dat[15:24])
                            velgrads.append(v33)
                            tincrs.append(dat[24])
                            # sr, w = self.Decompose_SA(v33)
                        elif ncol==45: ## EVPSC
                            self.imod='EVPSC'
                            temps.append(dat[14])
                            #eps_el = self.conv6_to_33(dat[15:21])
                            # eps_pl = eps_el.copy() ## not relevant anymore for delta-EVPSC
                            # eps_pl = self.conv6_to_33(dat[21:27])
                            eps_tr = self.conv6_to_33(dat[21:27])
                            #eps_tr = self.conv6_to_33(dat[27:33])
                            v33    = self.conv9_to_33(dat[33:42])
                            # strain_el.append(eps_el)
                            # strain_pl.append(eps_pl)
                            strain_tr.append(eps_tr)
                            velgrads.append(v33)
                            tincrs.append(dat[42])

                            pmac.append(dat[43])
                            pwgt.append(dat[44])

                        elif ncol==22: ## EPSC4
                            self.imod='EPSC4'
                        elif ncol==17: ## EVPSC-HW
                            self.imod='EVPSC-HW'
                        elif ncol==14: ## EVPSCHW-ori
                            self.imod='EVPSC-HW-ORI'
                        else:
                            raise IOError('Unexpected number of columns found in data file')

            # print 'IMOD:', self.imod

            #print(f'ncol:{ncol}')
            #print(f'iopt:{iopt}')

            if iopt==1:
                ibreak=False
                dat=list(map(float,datl[-2].split()))

                evm,svm=dat[0:2]
                EVM.append(evm)
                SVM.append(svm)
                strain=dat[2:8]
                stress=dat[8:14]
                epsilon.append(strain)
                sigma.append(stress)
                if len(dat)==25: # vpsc
                    tempr = dat[14]
                    v33   = self.conv9_to_33(dat[15:24])
                    velgrads.append(v33)
                elif ncol==45: # evpsc
                    tempr = dat[14]
                    eps_el = self.conv6_to_33(dat[15:21])
                    eps_pl = self.conv6_to_33(dat[21:27])
                    eps_tr = self.conv6_to_33(dat[27:33])
                    v33    = self.conv9_to_33(dat[33:42])
                    #strain_el.append(eps_el)
                    #strain_pl.append(eps_pl)
                    strain_tr.append(eps_tr)
                    temps.append(tempr)
                    velgrads.append(v33)
                    tincrs.append(dat[42])
                    pmac.append(dat[43])
                    pwgt.append(dat[44])
                else:
                    print('**ncol:',ncol)
                    print('dat:')
                    print(dat)
                    raise IOError('Unexpected number of columns found in data file')


        self.get_6stress(x=np.array(sigma).T)
        self.get_6strain(x=np.array(epsilon).T)
        self.epsilon_vm = EVM[::]
        self.sigma_vm=SVM[::]
        self.w = cumtrapz(y=SVM,x=EVM,initial=0)

        #print(f'self.imod: {self.imod}')

        if self.imod=='VPSC':
            self.velgrads = np.array(velgrads)
            self.velgrads = self.velgrads.swapaxes(0,2).swapaxes(0,1)
            self.tincrs = np.array(tincrs)
        elif self.imod=='EVPSC':
            self.velgrads = np.array(velgrads)
            self.velgrads = self.velgrads.swapaxes(0,2).swapaxes(0,1)
            #self.strain_el=np.array(strain_el)
            #self.strain_el = self.strain_el.swapaxes(0,2).swapaxes(0,1)
            #self.strain_pl=np.array(strain_pl)
            #self.strain_pl = self.strain_pl.swapaxes(0,2).swapaxes(0,1)
            self.strain_tr=np.array(strain_tr)
            self.strain_tr = self.strain_tr.swapaxes(0,2).swapaxes(0,1)
            self.pmac=np.array(pmac)
            self.pwgt=np.array(pwgt)
            self.tincrs = np.array(tincrs)
            self.temps=np.array(temps)
        elif self.imod=='EVPSC-HW':
            pass
        elif self.imod=='EVPSC-HW-ORI':
            pass

        if self.imod in ['VPSC','EVPSC']:
            v  = self.velgrads.copy()
            vt = self.velgrads.swapaxes(0,1)
            self.d33 = 0.5 * (v+vt)
            self.w33 = 0.5 * (v-vt)

            ind=~(self.d33[2,2]==0)
            self.instR=np.zeros(len(ind))
            self.instR[~ind]=np.nan
            self.instR[ind] = self.d33[1,1][ind]/self.d33[2,2][ind]


    def get_pmodel(self,fn):
        dat    = np.loadtxt(fn,skiprows=1).T
        stress = dat[6:12]
        strain = dat[12:18]
        self.get_6stress(x=stress)
        self.get_6strain(x=strain)

    def get_pmodel_lat(self,fn):
        dat = np.loadtxt(fn,skiprows=1).T
        e_phl = dat[18:24]
        self.get_6strain(x=e_phl)

    def get_stress(self,x,i,j):
        self.is_stress_available = True
        self.flag_sigma[i,j] = 1
        self.flag_6s[self.ivo[i,j]] = 1
        if len(x)>self.nstp:
            self.size(len(x))
        for k in range(len(x)):
            self.sigma[i,j,k] = x[k]

    def get_strain(self,x,i,j):
        self.is_strain_available = True
        self.flag_epsilon[i,j] = 1
        self.flag_6e[self.ivo[i,j]] = 1
        if len(x)>self.nstp:
            self.size(len(x))
        for k in range(len(x)):
            self.epsilon[i,j,k] = x[k]

    def get_6stress(self,x):
        """
        stress dimension: (6,nstp)
        """
        self.is_stress_available = True
        self.flag_sigma[:,:] = 1
        self.flag_6s[:] = 1
        n = x.shape[-1]
        if n>self.nstp:
            self.size(n)
        for k in range(len(self.vo)):
            i,j = self.vo[k]
            self.sigma[i,j,0:n] = x[k,0:n].copy()
            self.sigma[j,i,0:n] = x[k,0:n].copy()

    def get_6strain(self,x):
        """
        strain dimension: (6,nstp)
        """
        self.is_strain_available = True
        self.flag_epsilon[:,:] = 1
        self.flag_6e[:] = 1
        n = x.shape[-1]
        if n>self.nstp:
            self.size(n)
        for k in range(len(self.vo)):
            i,j = self.vo[k]
            self.epsilon[i,j,0:n] = x[k,0:n].copy()
            self.epsilon[j,i,0:n] = x[k,0:n].copy()

    def get_33stress(self,x):
        for i in range(3):
            for j in range(3):
                self.get_stress(x[i,j],i,j)
    def get_33strain(self,x):
        for i in range(3):
            for j in range(3):
                self.get_strain(x[i,j],i,j)

    def set_zero_sigma_ij(self,i,j):
        self.set_zero_sigma_k(k=self.ivo[i,j])

    def set_zero_epsilon_ij(self,i,j):
        self.set_zero_epsilon_k(k=self.ivo[i,j])

    def set_zero_sigma_k(self,k=None):
        i,j = self.vo[k]
        n = self.nstp
        self.sigma[i,j,0:n] = 0
        self.sigma[j,i,0:n] = 0

    def set_zero_epsilon_k(self,k=None):
        i,j = self.vo[k]
        n = self.nstp
        self.epsilon[i,j,0:n] = 0
        self.epsilon[j,i,0:n] = 0

    def set_zero_shear_strain(self):
        for i in range(3):
            for j in range(3):
                if i!=j: self.set_zero_epsilon_ij(i,j)

    def set_zero_shear_stress(self):
        for i in range(3):
            for j in range(3):
                if i!=j: self.set_zero_sigma_ij(i,j)

    def check(self):
        if self.sigma.shape!=self.epsilon.shape:
            raise IOError('Flow data array size is not matched.')

    def set_uni_axial(self):
        self.get_stress([0,100,300,400,500],0,0)
        self.set_zero_sigma_ij(1,1)
        self.set_zero_sigma_ij(2,2)
        self.set_zero_shear_stress()

        self.get_strain([0,0.00001,0.002,0.05,0.015],0,0)
        self.get_strain([0,-0.000003,-0.001,-0.025,-0.0075],1,1)
        self.get_strain([0,-0.000003,-0.001,-0.025,-0.0075],2,2)
        self.set_zero_shear_strain()

    def set_bi_axial(self):
        self.get_stress([0,100,300,400,500],0,0)
        self.get_stress([0,100,300,400,500],1,1)
        self.set_zero_sigma_ij(2,2)
        self.set_zero_shear_stress()

        self.get_strain([0,0.00001,0.002,0.05,0.015],0,0)
        self.get_strain([0,-0.000003,-0.001,-0.025,-0.0075],1,1)
        self.get_strain([0,-0.000003,-0.001,-0.025,-0.0075],2,2)
        self.set_zero_shear_strain()

    def integrate_work(self):
        """
        Cumulative trapzoidal method to
        calculate by integrating multidimensional (3x3)
        stress-strain constitutive data

        Advised to be used for 'experimental' data
        """

        if not(self.is_strain_available) or \
           not(self.is_stress_available):
            raise IOError('Either stress or strain is missing')
        k=0
        for i in range(3):
            for j in range(3):
                if self.flag_epsilon[i,j]==1 and \
                   self.flag_sigma[i,j]==1:
                    sij = self.sigma[i,j]
                    eij = self.epsilon[i,j]
                    if k==0:
                        w = cumtrapz(y=sij,x=eij,initial=0)
                    else:
                        w = w + cumtrapz(y=sij,x=eij,initial=0)
                    k=k+1
                else:
                    ## skip this component
                    pass
        self.work = w

    def size(self,n):
        """
        """
        oldn = self.nstp
        newsigma   = np.zeros((3,3,n))*np.nan
        newepsilon = np.zeros((3,3,n))*np.nan
        self.nstp = n
        if oldn==0:pass
        if oldn>0:
            for i in range(oldn):
                newsigma[:,:,i] = self.sigma[:,:,i].copy()
                newepsilon[:,:,i] = self.epsilon[:,:,i].copy()

        self.sigma=newsigma.copy()
        self.epsilon=newepsilon.copy()

    def fit_hard(self,p0s,p0v,iplot):
        """
        Fit strain-hardening parameters

        Arguments
        ---------
        p0s  - swift parameters
        p0v  - voce  parameters
        iplot
        """
        import mk.materials
        # import mk.materials.func_hard_for

        func_swift=mk.materials.func_hard.func_swift
        func_voce =mk.materials.func_hard.func_voce
        # func_hm   =mk.materials.func_hard.func_hollomon

        dat=[self.epsilon_vm, self.sigma_vm]
        self.f_swift, self.p_swift, pcov_swift \
            = mk.materials.func_hard_char.main(
                exp_dat=dat,f_hard=func_swift,params=p0s)
        self.f_voce,  self.p_voce,  pcov_voce \
            = mk.materials.func_hard_char.main(
                exp_dat=dat,f_hard=func_voce, params=p0v)

        ## Cannot pickle fortran object yet, thus is not considered.
        # a,b0,c,b1 = self.p_voce
        # k,eps_0,n = self.p_swift
        # m,qq = 5e-2, 1e3
        # self.f_voce_for  = mk.materials.func_hard_for.return_voce(
        #     a=a,b0=b0,c=c,b1=b1,m=m,qq=qq)
        # self.f_swift_for = mk.materials.func_hard_for.return_swift(
        #     n=n,m=5e-2,ks=k,e0=eps_0,qq=qq)

        if iplot:
            import matplotlib.pyplot as plt
            fig=plt.figure();ax=fig.add_subplot(111)
            x=np.linspace(self.epsilon_vm[0],self.epsilon_vm[1],100)
            y_voce =self.f_voce(x)
            y_swift=self.f_swift(x)

            ax.plot(self.epsilon_vm,
                    self.sigma_vm,'--',label='von Mises VPSC')
            ax.plot(x,y_voce,':',label='Voce fit')
            ax.plot(x,y_swift,'-.',label='Swift fit')
            ax.legend()


def true2engi(true_e,true_s):
    """
    Convert true stress/strain to engineering strains

    Arguments
    ---------
    true_e
    true_s
    """
    engi_e = __truestrain2e__(true_e[::])
    engi_s = true_s/(1+engi_e)
    return engi_e, engi_s

def engi_e2true_e(engi_e):
    """
    convert engineering strain to true strain.

    Arguments
    ---------
    engi_e
    """
    epsilon_true = np.log(1+engi_e)
    return epsilon_true

def engi2true(engi_e,engi_s):
    """
    Convert engineering stress / strain to true stress and strain.
    """
    if (engi_e>=0).all() and (engi_s>=0).all():
        epsilon_true = np.log(1+engi_e)
        sigma_true   = engi_s * (1+engi_e)
    else:
        raise IOError('Unexpected case')
    return epsilon_true, sigma_true

def __truestrain2e__(e):
    """Convert true strain to that of engineering"""
    return np.exp(e)-1.

def __IsEqFlow__(a,b):
    answer = True
    if not((a.flag_sigma==b.flag_sigma).all):
        print('sigma flag is not matched')
        answer = False
    if not((a.sigma==b.sigma).all):
        print('sigma is not the same')
        answer = False
    if not((a.flag_epsilon==b.flag_epsilon).all):
        print('epsilon flag is not matched')
        answer = False
    if not((a.epsilon==b.epsilon).all):
        print('epsilon is not the same')
        answer = False
    return answer


def average_flow_curve(xs,ys,n=10):
    """
    Return average flow curve

    Arguments
    ---------
    xs
    ys
    n
    """
    ndatset = len(xs)
    ## set proper x spacing
    mx = 0
    for i in range(ndatset):
        m = max(xs[i])
        if m>mx: mx = m

    x_ref = np.linspace(0,mx,n)

    ## interpolate each data files
    Y = []
    for i in range(ndatset):
        xp = xs[i]; fp = ys[i]
        y_i = np.interp(x_ref,xp,fp) # new interpolate y
        Y.append(y_i)
    Y = np.array(Y)
    ## average and standard deviation
    avg = []
    std = []
    for i in range(n):
        f = Y.T[i]
        a=np.average(f)
        b=np.std(f)
        avg.append(a)
        std.append(b)

    return x_ref, avg, std


def find_nhead(fn='STR_STR.OUT'):
    """
    Find the number of 'string' heads on an array file

    =========
    Arguments
    fn = 'STR_STR.OUT'

    =======
    Returns
    The number of heads
    """
    f=open(fn,'r');lines=f.readlines();f.close()
    nhead=0
    success=False
    while not(success):
        try:
            list(map(float,lines[nhead].split()))
        except ValueError:
            nhead=nhead+1
        else:
            success=True
    return nhead

def find_err(fa,fb):
    """
    Compare the two FlowCurve objects and return
    errors at indivial plastic levels.
    """
    from MP.mat.mech import FlowCurve as FC
    fd = FC()
    if fa.nstp!=fb.nstp: raise IOError('Number of data points are not matching')

    fd_sigma = fa.sigma - fb.sigma
    fd.get_33stress(fd_sigma)
    fd.get_vm_stress()

    ##fd.get_33strain(fa.epsilon)
    ## fd.epsilon_vm = fa.epsilon_vm[::]

    return (fd.sigma_vm) / fa.sigma_vm
"""
"""

class WPH:
    """
    May contain its corresponding Flow objective
    with respect to mechanical deformation flow
    """
    def __init__(self,iph=None):
        """
        Argument name: iph
        """
        self.iph=iph
        self.flow = FlowCurve()
        self.nstp = 0
        self.vf = []
        pass
    def get_wph(self,dat,i,j,strain=None,stress=None):
        self.get_vf(dat)
        if strain!=None: self.get_strain(strain,i,j)
        if stress!=None: self.get_stress(stress,i,j)

    def get_vf(self,dat):
        self.vf = dat
        self.nstp = len(self.vf)
    def get_strain(self,dat,i,j):
        self.flow.get_strain(dat,i,j)
    def get_stress(self,dat,i,j):
        self.flow.get_stress(dat,i,j)
    def check(self):
        self.flow.check()
        if self.flow.nstp!=self.nstp:
            raise IOError('Flow curve and WPH array size'\
                ' is not matched')
        self.nstp!=self.flow.nstp

class PCYS:
    def __init__(self):
        """
        How to represent a 'snapshot' of history?
        """
        self.fc=np.nan
        self.tx=np.nan
    def read_pcys(self,fn='PCYS.OUT'):
        s1,s2,d1,d2=np.loadtxt(fn,skiprows=1).T
        self.dat=np.array([s1,s2,d1,d2])
    def read_fc(self,fc):
        self.fc=fc
    # def read_tx_fn(self,fn='TEX_PH1.OUT'):
    #     pass

def dev2pi(s1,s2):
    """
    Project the yield locus (s1,s2) in deviatoric pi-plane
    to that in the plane-stress (s11,s22) with s33=0
    following below:


     s22-s11 = s1*sqrt(2)
    -s22-s11 = s2*sqrt(6)
    -------

    -2 * s11 = s1*sqrt(2) + s2*sqrt(6)
    s11 = -0.5 * (s1*sqrt(2)+s2*sqrt(6))
    s22 = s11 + s1*sqrt(2)


    Arguments
    ---------
    s1
    s2
    """
    sq2 = np.sqrt(2.)
    sq6 = np.sqrt(6.)

    s11 = -0.5 * (s1*sq2 + s2*sq6)
    s22 = s11 + s1*sq2
    return s11, s22

class Texture:
    def __init__(self,fn):
        self.read(fn)
    def read(self,fn='TEX_PH1.OUT'):
        """
        Read texture file from VPSC and store necessary information...

        - based on c_pf.py

        Argument
        --------
        fn='TEX_PH1.OUT'
        """
        conts = open(fn, 'r').read()
        blocks = conts.split('TEXTURE AT STRAIN =')
        blocks = blocks[1::]

        print('number of snapshots:', len(blocks))

        px  = [] ## polycrystal snapshot at each block
        macroStrain = [] ## macro strain of each snapshot.
        for i in range(len(blocks)):
            lines = blocks[i].split('\n')
            e = float(lines[0].split()[-1])
            macroStrain.append(e)
            linesOfGrains = lines[4:-1] # at each block
            grains=[]
            for eachLine in linesOfGrains:
                gr=list(map(float,eachLine.split()))
                grains.append(gr)
            px.append(grains)

        self.px          = np.array(px)
        self.macroStrain = np.array(macroStrain)

    def plot(self,ib=0,csym='cubic',cdim=[1.,1.,1.],**kwargs):
        """
        Plot pole figures for the given snapshot id <ib>

        Arguments
        ---------
        ib
        csym
        cdim
        **kwargs passed to upf.pf_new
        """
        import TX.upf
        mypf=TX.upf.polefigure(grains=self.px[ib],csym=csym,cdim=cdim)
        fig=mypf.pf_new(**kwargs)
        return fig


def flip(l, e, mode=0):
    """
    If any load of the data is found to be negative,
    mirror-flip the flow curve post to that negative point (self.mode==0)
    or translate it to the original point (self.mode=1)

    This will be called several times on a cyclic test result

    Arguments
    ---------
    l  (load array)
    e  (extension array)
    mode (0: mirror-flip or 1:mirror-flip and translate it to origin)
    """
    import numpy as np
    # self.mode= 0 or 1 (0 continuous, 1 translate to the origin)

    load = np.array(l).copy()
    ext  = np.array(e).copy()

    ## Dict list data (deformation segment) ------------- ##
    ext_dict  = dict()
    load_dict = dict()
    rvs_points = list()
    ## -------------------------------------------------- ##
    if len(load)!=len(ext): raise IOError

    iseg = 0 ## segment labeling
    rvs_points.append(0)
    for i in range(len(load)):
        if load[i]<0:
            iseg = iseg + 1
            rvs_points.append(i)
            ## change of the load sign (crossed the border)
            try:
                x0 =  ext[i-1]; x1 =  ext[i]
                y0 = load[i-1]; y1 = load[i]
            ## Probably due to wrong index for ext or load
            except: raise IOError('unexpected index problem')
            else:
                slope = (y1 - y0) / (x1 - x0)
                ## mirror-flip
                if mode==0:
                    point = - y0 / slope + x0
                    ext[i:len(ext)] = point + point - ext[i:len(ext)]
                    load[i:len(load)] = - load[i:len(load)]
                    pass
                ## mirror-flip and translates to (0,0) point
                elif mode==1:
                    ## perform the flip! ------------------------- ##
                    ext[i:] = - (ext[i:len(ext)] - ext[i])
                    load[i:] = - load[i:len(load)]
                    ## ------------------------------------------- ##

                    ## dictionary type data for each segment ----- ##
                    seg_start_ind = rvs_points[len(rvs_points)-2]
                    seg_finish_ind = rvs_points[-1]
                    # print seg_start_ind
                    # print seg_finish_ind
                    # print 'id: %i %i'%(seg_start_ind, seg_finish_ind)

                    dict_name = '%s_seg'%str(iseg).zfill(2)
                    ext_dict[dict_name] = ext[seg_start_ind:seg_finish_ind]
                    load_dict[dict_name] = load[seg_start_ind:seg_finish_ind]
                    ## ------------------------------------------- ##
                    pass
                pass
            pass
        pass
    if mode==0: return load, ext
    elif mode==1: return load_dict, ext_dict
    else: raise IOError
