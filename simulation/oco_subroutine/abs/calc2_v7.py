
import sys
import numpy as np
import h5py
import time

def calc2(dzf,pprf,tprf,
          denf,
          iz,zen,muzen,
          nwav,wcmdat,wavedat,
          absco, trns,
          iout=True):
    """
    iout: True to save details to the exist log.txt
    """
    if iout:
        f = open("log.txt", "a")
        sys.stdout = f

    # *********
    # The absco coefficients are cm2 per molecule
    # denf are number molecules per cm3 for one species
    # To yield extinction [1/km], multiply
    #   1.0e5 * absco (cm2/molecule) o2den (molecule/cm3) 
    # *********

    conv = 1.0e5 * denf[iz]
    # ----- original -----
    """
    ext = np.zeros(nwav)
    for i in range(nwav):
        ext[i]=conv*absco[i]
        """
    # -----------

    # faster, but the values are very close but slightly different
    ext = (conv*absco).flatten() 
 
    # *********
    #if iout:
    #    f.close()
    """print('ext shape:', file=sys.stderr)
    print(ext.shape, file=sys.stderr)
    
    print('denf:', denf, file=sys.stderr)

    print('absco shape:', absco.shape, file=sys.stderr)
    print('absco:', absco, file=sys.stderr)
    print('absco min max:', absco.min(), absco.max(), file=sys.stderr)"""
    return ext
