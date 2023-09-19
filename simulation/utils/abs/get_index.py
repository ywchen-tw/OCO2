import sys
import numpy as np
import bisect as bs

def get_PT_index(tkobs, pobs, tk, hpa, trilinear=True):
    """
    tkobs: temperatrue of the layer in K
    pobs: pressure of the layer in hPa

    # Output
    T_ind:  temperature index
    P_ind:  pressure index

    """

    # *********
    # hpa values range from small to large
    P_ind = bs.bisect_left(hpa, pobs)

    if trilinear:
        P_ind -= 1
    
    # *********
    # temperature values range from small to large
    # note that tk is stored tk(temp index, pressure index)
    T_ind = bs.bisect_left(tk[P_ind, :], tkobs)
            

    if trilinear:
        # get the left index for trilinear interpolation 
        T_ind -= 1
    
    if pobs >= hpa[P_ind] and pobs < hpa[P_ind+1]:
        None
    else:
        print('[Warning!!!]', pobs, [hpa[P_ind], hpa[P_ind+1]], file=sys.stderr)

    if tkobs >= tk[P_ind, T_ind] and tkobs < tk[P_ind, T_ind+1]:
        None
    else:
        print('[Warning!!!]', tkobs, [tk[P_ind, T_ind], tk[P_ind, T_ind+1], tk[P_ind, T_ind+2]], file=sys.stderr)
        print('tk', tk[P_ind, :], file=sys.stderr)

    return T_ind, P_ind

