"""
Shell sort from wikibooks
-------------------------

http://en.wikibooks.org/wiki/Algorithm_Implementation/Sorting/Shell_sort
"""
import numpy as np
def shellSort(a):
    """Shell sort using Shell's (original)
    gap sequence: n/2, n/4, ..., 1."""
    array = list(a)
    array = np.array(array)
    ind = np.arange(len(array))
    #i0 = -1000
    gap = len(array) // 2
    # loop over the gaps
    while gap > 0:
        # do the insertion sort
        for i in range(gap, len(array)):
            val = array[i]
            i0 = ind[i]
            j = i
            while j >= gap and array[j - gap] > val:
                array[j] = array[j - gap]
                ind[j] = ind[j-gap]
                j -= gap
            array[j] = val
            ind[j] = i0
        gap //= 2
    return array, ind

def sh(array=None,*args):
    """
    sort the given array and the sort
    the *args arrays according to
    the sorting arangement taken on the array.
    """
    new_arrays = []
    a = list(array)
    a = np.array(a)
    newa, ind = shellSort(a)
    new_arrays.append(newa)
    for a in args:
        rst = ind_swap(a,ind)
        new_arrays.append(rst)

    return new_arrays

def ind_swap(a,ind):
    """ Rearrange the order of array a in
    accordance with the given index

    Arguments
    =========
    a   : array
    ind : ind
    """
    import numpy as np
    a = np.array(a); ind = np.array(ind)
    if len(a)!=len(ind): raise IOError('Wrong index...')
    old = list(a)
    old = np.array(old)
    new = []
    for i in range(len(ind)):
        new.append(old[ind[i]])
    return np.array(new)
