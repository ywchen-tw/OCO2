import sys
import numpy as np
import bisect as bs

def getiijj(tkobs, pobs, tk, hpa, trilinear=True, iout=True):
    """
    tkobs: temperatrue of the layer in K
    pobs: pressure of the layer in hPa

    iout: True to save details to the exist log.txt

    # Output
    ii:  temperature index
    jj:  pressure index

    """

    # *********
    # hpa values range from small to large
    jj = bs.bisect_left(hpa, pobs)

    if trilinear:
        jj -= 1
    
    # *********
    # temperature values range from small to large
    # note that tk is stored tk(temp index, pressure index)
    ii = bs.bisect_left(tk[jj, :], tkobs)
            

    if trilinear:
        # get the left index for trilinear interpolation 
        ii -= 1
    
    if pobs >= hpa[jj] and pobs < hpa[jj+1]:
        None
    else:
        print('[Warning!!!]', pobs, [hpa[jj], hpa[jj+1]], file=sys.stderr)

    if tkobs >= tk[jj, ii] and tkobs < tk[jj, ii+1]:
        None
    else:
        print('[Warning!!!]', tkobs, [tk[jj, ii], tk[jj, ii+1], tk[jj, ii+2]], file=sys.stderr)
        print('tk', tk[jj, :], file=sys.stderr)



    # *********
    # Write out results
    if iout:
        print('  ')
        print('  getiijj: tkobs,pobs')
        print('  ',tkobs,pobs)
        print('  getiijj: temperature ii, pressure jj ', ii, jj)
        print('  getiijj: hpa(jj),tk(ii,jj) ')
        print('  ', hpa[jj], tk[jj, ii])


    # *********
    # Stop if there is a problem
    """
    if ((iokt == 0)  or (iokp == 0)):
        print()
        sys.exit(f'  getiijj: oops  iokt: {iokt}, iokp: {iokp}')
    """
    # *********
    # Close the log.txt file
    #if iout:
    #    f.close()
    return ii, jj

