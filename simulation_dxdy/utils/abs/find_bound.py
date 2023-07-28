
import sys
import numpy as np
from bisect import bisect_right as bs
import time

def find_boundary(wcm1, wcm2, wcm):
    """
    wcm1, wcm2: wavenumber boundary for the band in cm-1
    wcm: wavenumber series from the absco file
    """

    iwcm1 = bs(wcm, wcm1)-1
    iwcm2 = bs(wcm, wcm2) #bs(wcm, wcm2)-1 

    nwav = iwcm2-iwcm1+1
    wcmdat = wcm[iwcm1:iwcm2+1]
    wavedat = 1e4/wcmdat

    return iwcm1, iwcm2, nwav, wcmdat, wavedat

