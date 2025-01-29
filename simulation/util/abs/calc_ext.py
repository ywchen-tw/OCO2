
import sys

def calc_ext(denf, iz, absco):

    # *********
    # The absco coefficients are cm2 per molecule
    # denf are number molecules per cm3 for one species
    # To yield extinction [1/m], multiply
    #   1.0e2 * absco (cm2/molecule) * den (molecule/cm3) 
    # *********

    conv = 1.0e2 * denf[iz]
    ext = (conv*absco).flatten() 
 
    return ext
