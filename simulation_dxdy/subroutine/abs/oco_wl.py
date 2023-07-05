import sys
import numpy as np
import h5py

def oco_wv(iband, sat, footprint=1):
    """
    Sub-program calculates OCO wavelengths by L1b dispersion coefficients

    # output
    wl: wavelength in mircon
    """
    if iband < 0 or iband > 2:
        sys.exit('oco_wl: wrong band #')

    # Get dispersion coefficients from l1b data
    with h5py.File(sat.fnames['oco_l1b'][0], 'r') as f:
        dis = f["InstrumentHeader/dispersion_coef_samp"][...]
        
    lam = np.zeros([8, 1016, 3])
    wli = np.arange(1,1017,dtype=float)
    for i in range(8): 
        for j in range(3):
            for k in range(5):
                lam[i,:,j]=lam[i,:,j] + dis[j,i,k]*wli**k  

    return lam[footprint, :, iband]