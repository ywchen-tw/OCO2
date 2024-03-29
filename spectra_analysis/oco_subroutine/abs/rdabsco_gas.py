import sys
import numpy as np
import h5py
import bisect as bs
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt

def rdabsco_species(filnm, 
                    wcmo2,p_species, tk_species,broad_species,
                    hpa_species, wavelo2,
                    tkobs,pobs,jbroad,
                    ii,jj, h2o_vmr,
                    nwav,wcmdat,wavedat,
                    wavel1,wavel2,wcm1,wcm2,iwcm1,iwcm2,
                    nunits, units_species,
                    species, 
                    VarName,
                    mode="trilinear", # single, linear, trilinear 
                    iout=True):
    """
    mode: single    -> use the closest indices for P, T, H2O mixing ratio
          linear    -> use the closest indices for P, T and linear interpolation for H2O mixing ratio
          trilinear -> trilinear interpolation for P, T, H2O mixing ratio
    iout: True to save details to the exist log.txt

    ii:  temperature index
    jj:  pressure index

    """
    if iout:
        original_stdout = sys.stdout # Save a reference to the original standard output
        f = open("log.txt", "a")
        sys.stdout = f

    # ********
    # Open the hdf5 file
    h5data = h5py.File(filnm, 'r')

    if iout:
        print('  ')
        print(f'  rdabsco {species} FileName: {filnm}')


    # ******
    # Will read in every absco coefficient for the temperature index ii
    # and pressure index jj for a range of wavenumber
    # hdfview says Gas_07_Absorption is 64       x 17     x 3   x very large
    #                                   pressure  temp   broad  wcm

    # Obtain the absco coefficients by grabbing a hyperslab
    ncount = nwav


    if mode == 'single':
        absco = h5data[VarName][...][jj, ii, jbroad, iwcm1:iwcm2+1]

    elif mode == 'linear':
        # H2O mixing ratio linear interpoation
        hh = bs.bisect_left(broad_species, h2o_vmr)-1
        if hh == 2:
            print('hh out ouf range!', file=sys.stderr)
            hh = 1
        absco_h0 = h5data[VarName][...][jj,    ii,     hh,      iwcm1:iwcm2+1]
        absco_h1 = h5data[VarName][...][jj,    ii,     hh+1,    iwcm1:iwcm2+1]
        dH2O_vmr = (h2o_vmr-broad_species[hh])/(broad_species[hh+1]-broad_species[hh])
        absco = absco_h0+dH2O_vmr*absco_h1

    elif mode == 'trilinear':
        # P, T, H2O mixing ratio trilinear interpolation
        print(f'p={pobs}, [{p_species[jj]/100}, {p_species[jj+1]/100}]', file=sys.stderr)
        print(f'T={tkobs}, [{tk_species[jj, ii]}, {tk_species[jj, ii+1]}]', file=sys.stderr)

        hh = bs.bisect_left(broad_species, h2o_vmr)-1
        if hh == 2:
            print('H2O mixing ratio out ouf range, using extrapolation!', file=sys.stderr)
            hh = 1
        print(f'H2O vmr={h2o_vmr}, [{broad_species[hh]}, {broad_species[hh+1]}]', file=sys.stderr)

        dp = (pobs-hpa_species[jj])/(hpa_species[jj+1]-hpa_species[jj])
        dT = (tkobs-tk_species[jj, ii])/(tk_species[jj, ii+1]-tk_species[jj, ii])
        dH2O_vmr = (h2o_vmr-broad_species[hh])/(broad_species[hh+1]-broad_species[hh])
        print('dp:', dp, file=sys.stderr)
        print('dT:', dT, file=sys.stderr)
        print('dH2O_vmr:', dH2O_vmr, file=sys.stderr)
        matrix = np.array([[ 1,  0,  0,  0,  0,  0,  0,  0],
                        [-1,  0,  0,  0,  1,  0,  0,  0],
                        [-1,  0,  1,  0,  0,  0,  0,  0],
                        [-1,  1,  0,  0,  0,  0,  0,  0],
                        [ 1,  0, -1,  0, -1,  0,  1,  0],
                        [ 1, -1, -1,  1,  0,  0,  0,  0],
                        [ 1, -1,  0,  0, -1,  1,  0,  0],
                        [-1,  1,  1, -1,  1, -1, -1,  1]])
        absco_000 = h5data[VarName][...][jj,    ii,     hh,     iwcm1:iwcm2+1]
        absco_100 = h5data[VarName][...][jj+1,  ii,     hh,     iwcm1:iwcm2+1]
        absco_010 = h5data[VarName][...][jj,    ii+1,   hh,     iwcm1:iwcm2+1]
        absco_001 = h5data[VarName][...][jj,    ii,     hh+1,   iwcm1:iwcm2+1]
        absco_110 = h5data[VarName][...][jj+1,  ii+1,   hh,     iwcm1:iwcm2+1]
        absco_101 = h5data[VarName][...][jj+1,  ii,     hh+1,   iwcm1:iwcm2+1]
        absco_011 = h5data[VarName][...][jj,    ii+1,   hh+1,   iwcm1:iwcm2+1]
        absco_111 = h5data[VarName][...][jj+1,  ii+1,   hh+1,   iwcm1:iwcm2+1]

        coeff = np.dot(matrix, np.array([absco_000, absco_001, absco_010, absco_011,
                                         absco_100, absco_101, absco_110, absco_111]))
        Q_vec = np.array([1.0, dp, dT, dH2O_vmr, dp*dT, dT*dH2O_vmr, dH2O_vmr*dp, dp*dT*dH2O_vmr]).reshape(8, 1)
        absco = np.dot(Q_vec.T, coeff).flatten()


        """# use LinearNDInterpolator to interpolate the value at the desired point
        points = np.array([[hpa_species[jj], tk_species[jj, ii], broad_species[hh]],
                           [hpa_species[jj+1], tk_species[jj, ii], broad_species[hh]],
                           [hpa_species[jj], tk_species[jj, ii+1], broad_species[hh]],
                           [hpa_species[jj], tk_species[jj, ii], broad_species[hh+1]],
                           [hpa_species[jj+1], tk_species[jj, ii+1], broad_species[hh]],
                           [hpa_species[jj+1], tk_species[jj, ii], broad_species[hh+1]],
                           [hpa_species[jj], tk_species[jj, ii+1], broad_species[hh+1]],
                           [hpa_species[jj+1], tk_species[jj, ii+1], broad_species[hh+1]],])
        print('points shape:', points.shape, file=sys.stderr)
        values_at_points = np.array([absco_000, absco_001, absco_010, absco_011,
                                         absco_100, absco_101, absco_110, absco_111])
        print('values_at_points shape:', values_at_points.shape, file=sys.stderr)
        f = LinearNDInterpolator(points, values_at_points)
        result = f(pobs, tkobs, h2o_vmr)

        print(h5data[VarName][...].shape, file=sys.stderr)
        print('trilinear absco mean:', absco.mean(), file=sys.stderr)
        print('scipy absco mean:', result.mean(), file=sys.stderr)
        print('diff absco mean:', ((result-absco)/absco).mean(), file=sys.stderr)
        #points_all = np.array([])"""


        #sys.exit()

    # *********
    if iout:
        print('\n')
        print(f'  rdabsco {species}: filnm ',filnm)
        print(f'  rdabsco {species}: i,units_{species}(i)')
        for i in range(nunits):
            print('  ', i, ' ', units_species[i])
        print('\n')
        print(f'  rdabsco {species}: tkobs,pobs,jbroad')
        print('  ',tkobs,pobs,jbroad)
        print(f'  rdabsco {species}: temperature ii, pressure jj ',ii,jj)
        print(f'  rdabsco {species}: hpa_{species}(jj),tk_{species}(ii,jj) ')
        print('  ',hpa_species[jj], tk_species[jj, ii])
        iskip=nwav//30
        print('\n')
        print(f'  rdabsco {species}: i,wcmdat(i),wavedat(i),absco,')
        for i in range(0, nwav, iskip):
            print('  ', i, wcmdat[i], wavedat[i], absco[i])

    # *****
    # Close the file
    h5data.close()

    return absco

