import numpy as np
import h5py
import sys

def rdabs_species(filnm, species, iout=True):
    """
    iout: True to save details to the exist log.txt
    """

    #########
    # Open the hdf5 file
    # (pg 347 of IDL Scientific Data Formats)
    h5data = h5py.File(filnm, 'r')

    if iout == 1:
        print('  ')
        print(f'  rdabs {species} FileName: {filnm}')


    #########
    # Read in the Pressure values
    # note that p_species has 71 elements
    p_species = h5data['Pressure'][...]
    np_species = len(p_species)

    # Obtain hpa pressure
    hpa_species = p_species/100.0

    # ******
    # Read in the Temperature values
    # Note that tk_species is 71 x 17 (by hdfview)
    #  and that the data is stored tk_species(17,71)
    # 71 pressure values
    # 17 temperature values (they are  not  the same at each pressure level)
    tk_species = h5data['Temperature'][...]
    ntk_species = len(tk_species)

    

    n17 = 17
    vec17 = np.empty(n17)

    # ******
    # Read in the Wavenumber grid
    wcm_species = h5data['Wavenumber'][...]
    nwcm_species = len(wcm_species)

    # ******
    # Read in the Broadener VMRs
    broad_species = h5data['Broadener_01_VMR'][...]
    nbroad_species = len(broad_species)

    # ******    
    # Close the hdf file
    h5data.close()


    Gas_ID = {'o2': '07',
              'wco2': '02',
              'sco2': 'sco2',
              'h2o': 'h2o',
              'co2': '02',
              'ch4': '06'}

    # ********
    # Specify the units of the data
    nunits_species = 5
    units_species = np.chararray(nunits_species)
    units_species[0] = 'Pressure (Pascal)'
    units_species[1] = 'Temperature (K)'
    units_species[2] = 'Wavenumber (cm-1)'
    # units_species[3] = 'Gas_07_Absorption (m-2/mol)'
    # See oc_species atbd
    
    units_species[3] = f'Gas_{Gas_ID[species]}_Absorption (cm2/mol)'
    units_species[4] = 'Broadner_01_VMR   (volume mix ratio)'

    # *********
    # Obtain the wavelengths in micrometer (um)
    wavel_species = np.empty(nwcm_species)
    for i in range(nwcm_species):
        wavel_species[i] = 1.0e4/wcm_species[i] 

    # *********
    if iout:
        print('  ')
        print(f'  rdabs {species}: filnm ',filnm)
        print(f'  rdabs {species}: np_{species},ntk_{species},nbroad_{species},nwcm_{species}')
        print(f'  ', np_species, ntk_species, nbroad_species,nwcm_species)
        print(f'  rdabs {species}: i,units_{species}(i)')
        for i in range(nunits_species):#
            print('  ', i, ' ', units_species[i])
        print(f'  rdabs {species}: p_species ', p_species)
        print(f'  rdabs {species}: broad_species ',broad_species)
        print(f'  rdabs {species}: min and max wcm_{species} ',np.min(wcm_species),np.max(wcm_species))
        print(f'  rdabs {species}: min and max wavel_{species} ',np.min(wavel_species),np.max(wavel_species))

    # *
    # note that tk_species is stored tk_species(temp index, pressure index)
    iwrsp1=1
    if iwrsp1 == 1:
        print('\n')
        print(f'  rdabs {species}: i,p{species}(i),hpa_{species}(i),tk_{species}(i,j) for j=0,16')
        for i in range(np_species):
            for j in range(n17):
                vec17[j] = tk_species[i, j]
                print('  ',i, p_species[i], hpa_species[i],'  ',vec17)

    # *
    iwrsp2=1
    if iwrsp2 == 1:
        iskip=1000
        print('  ')
        print(f'  rdabs {species}: i,wcm_{species}(i),wavel_{species}(i)')
        for i in range(0, nwcm_species, iskip):
            print('  ',i,wcm_species[i],wavel_species[i])
    
    return np_species, ntk_species, nbroad_species, nwcm_species,\
           wcm_species, p_species, tk_species, broad_species,\
           hpa_species,wavel_species, nunits_species, units_species

