def BCC_voce(fn='B_ST_2k_tan_bul_rs.sx',
            tau0=80.000,tau1=70.000,thet0=340.000,thet1=20.000,
            fnout=None):
    """
    24 slip systems (110)<111> (112)<111>
    """
    lines=open(fn,'r').readlines()
    L = []
    for i in range(len(lines)):
        L.append(lines[i].split('\n')[0])
    lines = L[:]
    if fnout==None: fnout='dum'
    fout=open(fnout,'w')
    for i in range(50):
        fout.write('%s\n'%lines[i])
    fout.write(' %20.12e  %20.12e  %20.12e  %20.12e   0  0 \n'%(
            tau0,tau1,thet0,thet1))
    fout.write('%s \n'%lines[51])
    fout.write('%s \n'%lines[52])
    fout.write('%s \n'%lines[53])
    fout.write(' %20.12e  %20.12e  %20.12e  %20.12e   0  0 \n'%(
            tau0,tau1,thet0,thet1))
    fout.write('%s \n'%lines[55])
    fout.close()
