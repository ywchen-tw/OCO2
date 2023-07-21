import h5py
import sys
import numpy as np

def getdatah5(nopr, FileName, VarName, iout=True):
    """
    iout: True to save details to the exist log.txt
    """
    # ****************
    # See get_aura.pro
    h5data = h5py.File(FileName, 'r')

    if nopr == 1:
        print('  ', )
        print(f'  getdatah5: filename {FileName}')
        print(f'  getdatah5: VarName {VarName}')

    data = h5data[VarName][...].copy()
    print(data)
    Dims = data.shape

    if nopr == 1:
        print('  getdatah5: Dims ',Dims)

    h5data.close()


    if nopr == 1:
        print('  getdatah5: min and max data ', np.min(data), np.max(data))

    #####
    # This action pg 1026 of IDl ref guide N-Z, returns array deleting
    # all leading dimensions of size 1.
    reform_shape = []
    for i in range(len(data.shape)):
        dim_len = data.shape[i]
        reform_shape.append(dim_len) if dim_len != 1 else None
    if tuple(reform_shape) != data.shape:
        data = data.reshape(reform_shape)

    # ******************************
    if nopr == 1:
        print('  getdatah5: FileName ', FileName)
        print('  getdatah5: VarName ', VarName)
        print('  getdatah5: Dims ',Dims)
        print('  getdatah5: min and max data ')
        print('  ',np.min(data),np.max(data))

    # ******************************

    
    return Dims, data

