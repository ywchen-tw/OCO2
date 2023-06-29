import h5py
import sys
import numpy as np

def getdatah5a(nopr, FileName, VarName, 
              ii,jj,ncount,iwcm1,iwcm2,jbroad,
              iout=True):
    """
    iout: True to save details to the exist log.txt
    """
    if iout:
        original_stdout = sys.stdout # Save a reference to the original standard output
        f = open("log.txt", "a")
        sys.stdout = f

    # ****************
    # Work with a hyperslab

    # ****************
    # dataset id
    h5data = h5py.File(FileName, 'r')

    # Details (will stop) with nopr2=1
    nopr2=0
    if nopr2 == 1:
        print('  ')
        print(f'  getdatah5: filename {FileName}')

    # **
    # Specify the hyperslab
    # hdfview has absco values as 71 x 17 x 4 x 90001
    # refer to in  reverse  order

    # Read in ncount elements
    # Temperature index is ii    (17)
    # Pressure    index is jj    (71)
    # Will start with wavenumber at position iwcm1

    # Read in the data
    #print(h5data[VarName][...].shape, file=sys.stderr)
    #print([np.int(iwcm1), np.int(iwcm1+ncount), jbroad, ii, jj], file=sys.stderr)
    #absco = h5data[VarName][...][np.int(iwcm1):np.int(iwcm1+ncount), jbroad, ii, jj]
    absco = h5data[VarName][...][jj, ii, jbroad, np.int(iwcm1):np.int(iwcm1+ncount)]
    Dims = absco.shape
    if nopr2 == 1:
        print('  getdatah5: min and max absco ', np.min(absco), np.max(absco))

    # **
    # This action pg 1026 of IDl ref guide N-Z, returns array deleting
    # all leading dimensions of size 1. Not necessary here.
    # absco = reform(absco)
    """
    reform_shape = []
    for i in range(len(absco.shape)):
        dim_len = absco.shape[i]
        reform_shape.append(dim_len) if dim_len != 1 else None
    if tuple(reform_shape) != absco.shape:
        absco = absco.reshape(reform_shape)
    """

    # *****
    # Close the file
    h5data.close()

    # ******************************
    if nopr == 1:
        print('  getdatah5a: FileName ', FileName)
        print('  getdatah5a: VarName ', VarName)
        print('  getdatah5a: Dims ',Dims)
        print('  getdatah5a: ncount ', ncount)
        print('  getdatah5a: min and max absco ')
        print('  ', np.min(absco), np.max(absco))

    if nopr2 == 1:
        sys.exit()

    # ******************************
    if iout:
        f.close()
    
    return absco
