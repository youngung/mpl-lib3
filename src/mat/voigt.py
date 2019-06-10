import numpy as np
ijv = np.zeros((2,6),dtype='int')
ijv[0,0],ijv[1,0] = 0,0
ijv[0,1],ijv[1,1] = 1,1
ijv[0,2],ijv[1,2] = 2,2
ijv[0,3],ijv[1,3] = 1,2
ijv[0,4],ijv[1,4] = 0,2
ijv[0,5],ijv[1,5] = 0,1

vij = np.zeros((3,3),dtype='int')
vij[0,0] = 0
vij[1,1] = 1
vij[2,2] = 2
vij[1,2] = 3
vij[0,2] = 4
vij[0,1] = 5 

# lower triangle
vij[2,1] = 3
vij[2,0] = 4
vij[1,0] = 5 
