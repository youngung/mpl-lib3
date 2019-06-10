## dat analysis library
def trim(x,ref):
    ind=len(ref)-1
    for i in range(len(ref)):
        if ref[i]!=0:
            ind0 = i
            break
    reft=ref[::-1]
    for i in range(len(reft)):
        if reft[i]!=0:
            ind1=i
            break
    ind1=len(reft)-ind1
    return x[ind0:ind1],ref[ind0:ind1]
