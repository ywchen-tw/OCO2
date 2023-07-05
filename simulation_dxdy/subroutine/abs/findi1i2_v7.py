
import sys
import numpy as np
from bisect import bisect_right as bs
import time

def findi1i2(wcm1, wcm2, wcm, iout=True):
    """
    wcm1, wcm2: wavenumber boundary for the band in cm-1
    wcm: wavenumber series from the absco file
    iout: True to save details to the exist log.txt
    """
    if iout:
        original_stdout = sys.stdout # Save a reference to the original standard output
        f = open("log.txt", "a")
        sys.stdout = f
    
    iwcm1 = bs(wcm, wcm1)-1
    iwcm2 = bs(wcm, wcm2) #bs(wcm, wcm2)-1 

    #iwcm1 = np.argmin(np.abs(wcm-wcm1))
    #iwcm2 = np.argmin(np.abs(wcm-wcm2))

    """
    print(wcm1, wcm2, file=sys.stderr)
    print(wcm[iwcm1], wcm[iwcm2], file=sys.stderr)
    print(wcm[iwcm1-1:iwcm1+2], file=sys.stderr)
    print(wcm[iwcm2-1:iwcm2+2], file=sys.stderr)
    sys.exit()
    """

    nwav = iwcm2-iwcm1+1
    wcmdat = wcm[iwcm1:iwcm2+1]
    wavedat = 1e4/wcmdat
    
    # *********
    if iout:
        print('  ')
        print(f'  findi1i2: wavel1={1e4/wcm1:.5f} um, wavel2={1e4/wcm2:.5f} um')
        print(f'  findi1i2: wcm1={wcm1} cm-1, wcm2={wcm2} cm-1')
        print('  findi1i2: iwcm1,iwcm2 ',iwcm1,iwcm2  )
        print('  findi1i2: wcm(iwcm1),wcm(iwcm2) ',wcm[iwcm1],wcm[iwcm2])
        print('  findi1ii2: nwav ',nwav)
        print('  findi1ii2: min and max wcmdat ',np.min(wcmdat),np.max(wcmdat))
        print('  findi1ii2: min and max wavedat ',np.min(wavedat),np.max(wavedat))
    # *********

    if iout:
        f.close()

    return iwcm1, iwcm2, nwav, wcmdat, wavedat

