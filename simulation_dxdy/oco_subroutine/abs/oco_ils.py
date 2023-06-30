import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

def oco_ils(iband, sat):
    """
    Written by Steve Massie in IDL
    Edited by Sebastian Schmidt 3/2014 (program can still be optimized to only read one band at a time)
    Converted to Python by Yu-Wen Chen 1/2023
    ********
    See march 28 2013 oco2 science team presentation
    OCO2 and GOSAT: A tale of two instruments
    page 9 graph of ils
    """


    print('Results come out in nm, need to convert to microns')


    """
    # *****
    # The output arrays
    ndat = 101
    nbands = 3
    xils = np.empty((ndat,nbands))
    yils = np.empty((ndat,nbands))
    bandstr = ['O2 A band', 'Weak CO2', 'Strong CO2']
    xilstr = np.chararray(1)
    yilstr = np.chararray(1)
    xilstr = ['xils in nm']
    yilstr = ['yils normalized ILS, see Randy Pollock March 2013 OCO2 meeting']

    # ******
    # Loop over the three bands
    for k in range(nbands):
        # *****
        # o2 a band
        if k == 0:
            npts = 15
            scale = 1.0/29.5
            
            # in mn
            xval = np.array([-0.3,-0.2,-0.1,-0.08,-0.06,-0.04,-0.02,0.0,
                             0.02,0.04,0.06,0.08,0.1,0.2,0.3])
            
            # normalized response
            yval = np.empty(npts)
            yvaln = np.empty(npts)
            # in mm
            yvalmm = np.array([3.3,13.5,9.5,17.5,0.0,21.5,22.0,0.0,
                               22.0,21.5,0.0,18.0,14.0,14.0,3.0])
            # add this to yval
            yvalmm0 = np.array([0.0001,0.0001,0.001,0.001,0.01,0.01,0.1,1.0,
                                0.1,0.01,0.01,0.001,0.001,0.0001,0.0001])

            # The output range
            x1 = -0.30
            x2 = 0.30
            dx = (x2-x1)/(1.00*(ndat-1))

        # *****
        # weak co2
        if k == 1:
            npts=15
            scale=1.0/29.5

            # in mn
            xval = np.array([-1.0,-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.0,
                             0.05,0.1,0.2,0.3,0.4,0.5,1.0])
            # normalized response
            yval = np.empty(npts)
            yvaln = np.empty(npts)

            # in mm
            yvalmm = np.array([0.0,13.0,18.0,23.0,2.0,15.0,13.0,0.0,
                    0.0,15.0,1.0,23.5,15.5,11.0,26.5])
            # add this to yval
            yvalmm0 = np.array([0.0001,0.0001,0.0001,0.0001,0.0010,0.0010,0.10,1.0,
                    0.10,0.0010,0.0010,0.0001,0.0001,0.0001,0.00001])

            # The output range
            x1 = -1.0
            x2 = 1.0
            dx = (x2-x1)/(1.00*(ndat-1))

        # *****
        # strong co2
        if k == 2:
            npts=17
            scale=1.0/29.5

            # in mn
            xval = np.array([-1.5,-1.0,-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.0,
                             0.05,0.1,0.2,0.3,0.4,0.5,1.0,1.5])
            # normalized response
            yval = np.empty(npts)
            yvaln = np.empty(npts)

            # in mm
            yvalmm = np.array([0.0,8.5,21.0,27.0,0.5,11.0,9.0,17.0,0.0,
                               17.0,5.0,13.0,29.0,26.0,25.0,8.0,0.0])
            # add this to yval
            yvalmm0 = np.array([0.0001,0.0001,0.0001,0.0001,0.0010,0.0010,0.010,0.10,1.0,
                                0.10,0.010,0.0010,0.0001,0.0001,0.0001,0.0001,0.0001])

            # The output range
            x1 = -1.50
            x2 = 1.50
            dx = (x2-x1)/(1.00*(ndat-1))

        # *****
        for n in range(npts):
            val = yvalmm[n]*scale
            yval[n] = yvalmm0[n]*(10.0**val)
            yvaln[n] = np.log(yval[n])

        # *****
        # Obtain the ils for a single band
        nlast = npts-1
        for i in range(ndat):
            # The nm value of the ils
            x = x1+(i*dx)
            xils[i,k] = x

            # Do interpolation to find the ils
            for n in range(npts):
                if x >= xval[n]:
                    # Use adjacent points
                    n1 = n
                    n2 = n+1

                    # For the last point
                    if n == nlast:
                        ycalc = yval[n]
                        break

                    # use logs of the yval values
                    dxval = x-xval[n]
                    deriv = (yvaln[n2]-yvaln[n1])/(xval[n2]-xval[n1])
                    a1 = dxval*deriv
                    ycalc = np.exp(yvaln[n1]+a1)

            # Loop over table values
            mn = 0
            # The output
            yils[i, k]=ycalc
    # Loop over ils points is done

    # *****
    # Write out the ils values

    
    xx = xils[:, iband]
    yy = yils[:, iband]

    # ********
    #"""

    """
    # Get real ILS from one of the l1b data
    # neet to change to read l1b data eventually
    ils_l1b_output = pd.read_csv('l1b_ils_20190621_mean.csv')
    gas_dict = {0: 'o2', 1:'wco2', 2:'sco2'}
    xx = ils_l1b_output[f'{gas_dict[iband]}_del_lambda']*1000
    yy = ils_l1b_output[f'{gas_dict[iband]}_rel_response']/(ils_l1b_output[f'{gas_dict[iband]}_rel_response']).max()
    # """

    #
    # Get real ILS from l1b sounding data
    with h5py.File(sat.fnames['oco_l1b'][0], 'r') as f:
        del_lambda = f['InstrumentHeader/ils_delta_lambda'][...]
        rel_lresponse = f['InstrumentHeader/ils_relative_response'][...]

        del_lambda_mean = del_lambda[iband, :, :, :].mean(axis=(0, 1))
        rel_lresponse_mean = rel_lresponse[iband, :, :, :].mean(axis=(0, 1))

    xx = del_lambda_mean*1000
    yy = rel_lresponse_mean/(rel_lresponse_mean).max()
    
    




    print('xx shape', xx.shape, file=sys.stderr)
    return np.array(xx), np.array(yy)
