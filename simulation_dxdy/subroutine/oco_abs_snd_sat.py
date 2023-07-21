import numpy as np
import platform
import os, sys
import pickle, h5py
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm, colors

plt.rcParams["font.family"] = "Arial"

#+
# NAME:
#   abs_mcar
#
# PURPOSE:
#   makes atmosphere for OCO, including absorption coefficient profiles
#    
# CATEGORY:
#   mcarats / input
#   
# DEPENDENCIES:
#   - solar.py     # Reads in solar spectrum as used for OCO-2 at high spectral resolution
#
# EXAMPLE:
#   > abs(iband=0, nx=5, reextract=True, plot=True)
#   
# MODIFICATION HISTORY:
#  - written : Sebastian Schmidt, January 25, 2014
#  -      v3 : Partition in Transmittance rather than optical thickness
#         v4 : Use actual OCO wavelengths & ILS
#         v5 : Minor bug fix# port to cluster
#         v6 : Calculate *all* OCO-2 channels (setting "all_r")
#         v7 : Include water vapor absorption (only implemented in O2A so far)
#         v8 : Converted to Python-based, change the ILS function
#
#  Done:
#  - generalize to SCO2 & WCO2 for water vapor/methane inclusion & consolidate sub-routines
#  - get accurate wavelengths from recent cal & accurate OCO line shape (but just from csv file for now)
#  - Remove getdatah5, getdatah5a by just use h5py module
#  - implement H2O broadening
#  - bi/trilinear interpolation in {T,p,(h)} - can re-use Odele's code
#---------------------------------------------------------------------------
from subroutine.abs.read_atm import read_oco_zpt
# from oco_subroutine.abs.rho_air import rho_air  # density of air
from subroutine.abs.rdabs_gas import rdabs_species
from subroutine.abs.findi1i2_v7 import findi1i2 # get wavenumber indices.initialize()import abs/getiijj_v7.pro  # find levels in absco files that are closest to the atmosphere
from subroutine.abs.getiijj_v7 import getiijj  # find levels in absco files that are closest to the atmosphere.initialize()import abs/rdabscoo2.pro   # read absorption coefficients O2
from subroutine.abs.rdabsco_gas import rdabsco_species   # read absorption coefficients O2.initialize()import abs/rdabscoco2.pro  # read absorption coefficients CO2
from subroutine.abs.calc2_v8 import calc_ext   # calculates extinction profiles & layer transmittance from oco_subroutine.absorption coefficients & density profiles (CO2, O2)
from subroutine.abs.oco_wl import oco_wv      # reads OCO wavelengths
from subroutine.abs.oco_ils import oco_ils    # reads OCO line shape ("slit function")    
from subroutine.abs.solar import solar   # read solar file
from subroutine.abs.oco_convolve import oco_conv
from subroutine.abs.oco_abs_g_mode import oco_wv_select
from subroutine.oco_utils import timing
from subroutine.oco_cfg import grab_cfg



@timing
def oco_abs(cfg, sat, zpt_file, 
            iband=0, nx=None, Trn_min=0, 
            pathout=None, 
            reextract=True, plot=False):
    # *********
    # Specify which band to work with
    # iband=0,1,2 for o2, weak co2, strong co2

    if not iband in [0, 1, 2]:
        print('iband should be either 0 (O2), 1 (weak CO2), or 2 (Strong CO2), set to the default 0.')
        iband   = 0

    pathinp = './subroutine/abs/'
    if pathout == None:
        pathout = './'

    pl_ils  = 1       # plot ils, else result of convolution
    all_r   = 0       # 0: use T-sampling >0: full range & percentage of transmittance
    if nx < 0:
        nx = 5       # # of representative wavelengths @ native resolution of Vivienne's data base
    pdmax       = 100   # max difference between requested & available atmospheric pressure level
    tdmax       = 10    # max difference between requested & available atmospheric temp value
    sol         = pathinp+'sol/solar.txt' # high spectral resolution solar file at 1 AU

    original_stdout = sys.stdout
    sys.stdout = original_stdout
    # *************************************************************
        # important atmospheric settings (some inherited from 9/4/2015)
        # Note that co2mix & other mixing
        # ratios are set with code that
        # processes
        # reanalysis data - so values only
        # need to be included when reading
        # sonde instead
    # *************************************************************
  
    o2mix     = 0.20935       # See oco2-atbd pg 17 (pg 10 of doc, Version 2.0 Rev 3)
    #gmao      = '/Users/schmidt/rtm/ocosim/dat/160829/oco2_L2MetND_11496a_160829_B8000r_170710205752.h5_profile.h5'
    atminp     = pathinp+'../atm.h5'

    # *************************************************************
    # important spectroscopy settings
    # *************************************************************
    # wr = sub-range
    wr_dict = {0: [0.758, 0.773],   #o2
               1: [1.593, 1.622],   #wco2
               2: [2.043, 2.083],}  #sco2
    # xr = range of entire band
    xr_dict = {0: [0.757, 0.773],   #o2
               1: [1.590, 1.623],   #wco2
               2: [2.042, 2.085],}  #sco2
    
    lb_dict = {0: 'o2a', 1: 'wco2', 2: 'sco2',}
               
    wr, xr, lb = wr_dict[iband], xr_dict[iband], lb_dict[iband]     
    output = pathout+'/atm_abs_'+lb+'_'+str(nx+1)+'.h5' 

    print('***********************************************************')
    print(f'Make MCARATS atmosphere/gas absorption input for {lb}')
    print('***********************************************************')


    # Files that contain the ABSCO coefficients
    files = ['o2_v51.hdf',         # O2
             'co2_v51.hdf',   # CO2 weak
             'co2_v51.hdf',   # CO2 strong
             'h2o_v51.hdf',               # H2O
            ]
    files = [pathinp + i for i in files]
    h2o = files[3]

    # Get OCO sounding info
    cfg_info = grab_cfg(cfg)
    abs_inter = cfg_info['abs_interpolation']
    fp = int(cfg_info['footprint'])-1
    if fp < 0 or fp > 7:
        print('footprint should be between 1 and 8, set to the default 1.')
        fp = 0
    
    # ils0: lowest level within ILS that will be included in calculations
    # standard: 0.002 --> 0.2% cutoff for lines 
    # (not worth calculating them)
    ils0 = float(cfg_info['ils_min'])
    if ils0 < 0 or ils0 > 1:
        print('ils_min should be between 0 and 1, set to the default 0.05.')
        ils0 = 0.05
    wlsol0, fsol0 = solar(sol)

    # ******************************************************************************************************
    # Extract information from oco_subroutine.absCOF files (native resolution).
    # Check whether this was previously extracted.
    savpkl = f'{pathout}/{lb_dict[iband]}_abs.pkl'
    if not os.path.isfile(savpkl) or reextract==True:
        # *********
        # Read in z,p,t profile from radiosonde, calculate layer & interface properties

        intf, pintf, pprf, tprf, o2den, co2den,\
              h2oden, h2o_vmr, dzf, lay, alb_o2a,\
              alb_wco2, alb_sco2, solzen, obszen = read_oco_zpt(zpt_file=zpt_file)
        # p,t,number concentrations for layers (not interfaces)
        # solar zenith angle (solzen) & satellite view angle (obszen) are in degree
     
        nlay=len(lay)
        
        # absco hdf5 file for target band
        filnm=files[iband] 

        # *********
        # Specify the wavelength (wavenumber) range to work with
        #   refl is the surface reflectance (ignore)
        #   jbroado2 is the broadening index that is used
        wavel1, wavel2 = xr[0], xr[1]
        if (iband == 0) :
            refl = alb_o2a
            jbroado2=1
            
        # Weak CO2 band
        if (iband == 1) :
            refl = alb_wco2
            jbroadco2=1
            
        # Strong CO2 band
        if (iband == 2):
            refl = alb_sco2
            jbroadco2=1

        # no H2O broadening for single interpolation option
        jbroadh2o=0
        
        # *********
        # Calculate the wavenumbers for the two wavelengths
        wcm2 = 1.0e4/wavel1
        wcm1 = 1.0e4/wavel2

        # *********
        # Calculate the 1/cos factors for solzen & obszen geometry
        convr = np.pi/180.0
        musolzen = 1.0/(np.cos(solzen*convr))
        muobszen = 1.0/(np.cos(obszen*convr))

        # *********
        # Read in O2 absco atmosphere information
        if iband == 0 :

            # Find out the pressure,temperature,boadening,wavenumber(wavelength) grid within
            # the O2 absco file (do not read absorption yet)
            npo2, ntko2, nbroado2, nwcmo2, wcmo2, po2, tko2, \
            broado2, hpao2, wavelo2, nunits, unitso2 = rdabs_species(filnm=filnm, species='o2', iout=True)
                    
            # *********
            # Obtain the wavenumber indices to work with for O2
            # define subset of wavenumber (wcmdat) & wavelength grid (wavedat)
            # from O2 ABSCO file, based on wavelength interval as specified by the user
            iwcm1, iwcm2, nwav, wcmdat, wavedat = findi1i2(wcm1, wcm2, wcmo2, iout=True)
        
        # *********
        # Read in CO2 atmosphere information
        if ((iband == 1) | (iband == 2)) :
             
            # Find out the pressure,temperature,broadening,wavenumber info
            # species names for wco2 and sco2 are same
            npco2,ntkco2,nbroadco2,nwcmco2, wcmco2,pco2,tkco2, \
            broadco2, hpaco2,wavelco2, nunits,unitsco2 = rdabs_species(filnm=filnm, species='co2', iout=True)
                    
            # *********
            # Obtain the wavenumber indices to work with target gas
            # define subset of wavenumber (wcmdat) & wavelength grid (wavedat)
            # from ABSCO file, based on wavelength interval as specified by the user
            iwcm1, iwcm2, nwav, wcmdat, wavedat = findi1i2(wcm1, wcm2, wcmco2, iout=True)

        # *********
        # Read in H2O atmosphere information

        # Find out the pressure,temperature,boadening,wavenumber(wavelength) grid within
        # the H2O absco file (do not read absorption yet)
        nph2o, ntkh2o, nbroadh2o, nwcmh2o, wcmh2o, ph2o, tkh2o, \
        broadh2o, hpah2o, wavelh2o, nunitsh2o, unitsh2o = rdabs_species(filnm=h2o, species='h2o', iout=True)
        
        # *********
        # Obtain the wavenumber indices to work with for H2O
        # define subset of wavenumber (wcmdat) & wavelength grid (wavedat)
        # from H2O ABSCO file, based on wavelength interval as specified by the user
        iwcm1h2o,iwcm2h2o, nwavh2o,wcmdath2o,wavedath2o = findi1i2(wcm1, wcm2, wcmh2o, iout=True)
            
        if nwav != nwavh2o: 
            print('[Warning] Wavenumber gridding of O2 & H2O ABSCO files do not match.')                

        # *********
        # Now that the {p,T,vH2O,wl} grid is set up, based on ABSCO data base
        # & on user specifications, read out that actual absorption
        # information from the data base

        # *********
        # Initialize
        # Optical depths on interfaces & absorption coefficients on layers
        ext = np.empty((nwav,nlay))
        tau_in = np.zeros(nwav)
        tau_out = np.zeros(nwav)
        trns = np.zeros(nwav) 

        # *********
        # Start at top of atmosphere & go to the surface
        for iz in range(nlay)[::-1]: 
        # ADD FUNCTIONALITY    print,'  iz ',iz
            print(f'  iz  {iz}')
            tkobs, pobs = tprf[iz], pprf[iz]

            ext0  = np.zeros(nwav) # absorption coef for O2 or CO2
            ext1  = np.zeros(nwav) # absorption coef for H2O
            ext0[...] = np.nan
            ext1[...] = np.nan

            # *********
            # First find the indices ii & jj (ii pressure, jj temperature)
            # that are closest to tkobs & pobs (with pobs ij hPa)

            trilinear_opt = abs_inter=='trilinear'
            # For o2
            if (iband == 0) :
                
                # (1) get {p,T} indices within O2 ABSCO grid
                ii, jj = getiijj(tkobs, pobs, tko2, hpao2, trilinear=trilinear_opt, iout=True)

                if np.abs(pobs-hpao2[jj]) > pdmax: 
                    print('[Warning] O2 pressure grid too coarse - interpolate?')
                if np.abs(tkobs-tko2[jj, ii]) > tdmax: 
                    print('[Warning] O2 Temperature grid too coarse - interpolate?')

                # (2) get {p,T} indices within H2O ABSCO grid
                iih2o,jjh2o = getiijj(tkobs, pobs, tkh2o, hpah2o, trilinear=trilinear_opt, iout=True)

                if np.abs(pobs-hpah2o[jjh2o]) > pdmax: 
                    print('[Warning] H2O pressure grid too coarse - interpolate?')
                if np.abs(tkobs-tkh2o[jjh2o, iih2o]) > tdmax: 
                    print('[Warning] H2O temperature grid too coarse - interpolate?')
                print(f'p (O2, H2O) : {pobs:.2f} hPa, {hpao2[jj]:.2f} hPa, {hpah2o[jjh2o]:.2f} hPa')
                print(f'T (O2, H2O) : {tkobs:.2f} K, {tko2[jj, ii]:.2f} K, {tkh2o[jjh2o, iih2o]:.2f} K')
                
            # For co2
            if ((iband == 1) | (iband == 2)) :
                # (1) get {p,T} indices within WCO2 or SCO2 ABSCO grid
                ii, jj = getiijj(tkobs, pobs, tkco2, hpaco2, trilinear=trilinear_opt, iout=True)

                # (2) get {p,T} indices within H2O ABSCO grid
                iih2o,jjh2o = getiijj(tkobs, pobs, tkh2o, hpah2o, trilinear=trilinear_opt, iout=True)

                if np.abs(pobs-hpah2o[jjh2o]) > pdmax: 
                    print('[Warning] H2O pressure grid too coarse - interpolate?')
                if np.abs(tkobs-tkh2o[jjh2o, iih2o]) > tdmax: 
                    print('[Warning] H2O temperature grid too coarse - interpolate?')
                print(f'p (CO2, H2O) : {pobs:.2f} hPa, {hpaco2[jj]:.2f} hPa, {hpah2o[jjh2o]:.2f} hPa',file=sys.stderr)
                print(f'T (CO2, H2O) : {tkobs:.2f} K, {tkco2[jj, ii]:.2f} K, {tkh2o[jjh2o, iih2o]:.2f} K')                    
            
            # *********
            # Specify the absorption coefficients - here is where we actually read
            #                                       them!!!
            # (An accurate calculation would use Lagrange interpolation
            # amongst three temperature & three pressure sets of absco vectors)

            # For o2
            if (iband == 0):
                absco = rdabsco_species(filnm,
                                        wcmo2, po2, tko2,broado2,
                                        hpao2, wavelo2,
                                        tkobs,pobs,jbroado2,
                                        ii,jj,h2o_vmr[iz],
                                        nwav,wcmdat,wavedat,
                                        wavel1,wavel2,wcm1,wcm2,iwcm1,iwcm2,
                                        nunits,unitso2,
                                        species='o2', 
                                        VarName='Gas_07_Absorption',
                                        mode=abs_inter,
                                        iout=True)
            # For CO2
            elif ((iband == 1) | (iband == 2)):
                absco = rdabsco_species(filnm,
                                        wcmco2,pco2,tkco2,broadco2,
                                        hpaco2, wavelco2,
                                        tkobs,pobs,jbroadco2,
                                        ii,jj,h2o_vmr[iz],
                                        nwav,wcmdat,wavedat,
                                        wavel1,wavel2,wcm1,wcm2,iwcm1,iwcm2,
                                        nunits,unitsco2,
                                        species='co2', 
                                        VarName='Gas_02_Absorption',
                                        mode=abs_inter,
                                        iout=True)
            # For H2O  
            abscoh2o = rdabsco_species(h2o,
                                        wcmh2o,ph2o,tkh2o,broadh2o,
                                        hpah2o,wavelh2o,
                                        tkobs,pobs,jbroadh2o,
                                        iih2o,jjh2o,h2o_vmr[iz],
                                        nwavh2o,wcmdath2o,wavedath2o,
                                        wavel1,wavel2,wcm1,wcm2,iwcm1h2o,iwcm2h2o,
                                        nunitsh2o,unitsh2o,
                                        species='h2o', 
                                        VarName='Gas_01_Absorption',
                                        mode=abs_inter,
                                        iout=True)
            
            # *********
            # Use the absco coefficients
            # Note that absco coefficients are cm-2 per molecule

            if iband == 0:
                # For O2
                ext0 = calc_ext(o2den,
                             iz,solzen,musolzen,
                             nwav,wcmdat,wavedat,
                             absco, trns)
            elif ((iband == 1) | (iband == 2)) :
                # For CO2
                ext0 = calc_ext(co2den,
                             iz,solzen,musolzen,
                             nwav,wcmdat,wavedat,
                             absco, trns)
            # For H2O
            ext1 = calc_ext(h2oden,
                            iz,solzen,musolzen,
                            nwavh2o,wcmdath2o,wavedath2o,
                            abscoh2o, trns)   

            # Store the results in ext
            ext[:,iz] = ext0 + ext1 # ext0: O2/CO2, ext1: H2O
            tau_in += ext[:,iz] * dzf[iz] * musolzen
            tau_out += ext[:,iz] * dzf[iz] * muobszen

            # End of loop down to the surface     
        trns = np.exp(-(tau_in+tau_out))#*refl
        # *********
        # End of loop over all layers

 
        wloco = oco_wv(iband, sat, footprint=fp) # (micron)
        xx, yy = oco_ils(iband, sat, footprint=fp)
        trnsc, trnsc0, indlr, ils = oco_conv(iband, sat, ils0, wavedat, nwav, trns, fp=fp)

        # (4) plots
        if plot and pl_ils:
            fig, ax = plt.subplots(1, 1, figsize=(8, 3.5), sharex=False)
            fig.tight_layout(pad=5.0)
            title_size = 18
            label_size = 16
            legend_size = 16
            tick_size = 14

            x = xx.mean(axis=0)*1000
            ax.plot(x, yy.mean(axis=0), color='k')
            ax.vlines([x[ils[0]], x[ils[-1]]], ils0, 1, 'r')

            ax.tick_params(axis='both', labelsize=tick_size)

            ax.set_xlabel('$\mathrm{\Delta}$ Wavelength (nm)', fontsize=label_size)
            ax.set_ylabel('response', fontsize=label_size)
            ax.set_title('# ILS terms', fontsize=title_size)
            fig.savefig(f'{pathout}/band{iband}_4_ILS_mean.png', dpi=150, bbox_inches='tight')

        # read solar file
        wlsol0, fsol0 = solar(sol) # obtain irradiance [W m-2 nm-1]
        fsol = np.interp(wavedat, wlsol0, fsol0)

        # *** Save file contains abs data for pre-selected wavelength range & specified atmosphere
        # *** along with
        # *** ---1) ILS/ILS_index(ILS resolution),
        # *** ---2) wloco (OCO-2 band resolution),
        # *** ---3) solar file (interpolated to absorption data base resolution)
        # *** When changing atmosphere | wavelength range, need to re-read (run with flag /r)
        #savpkl = filnm+'.pkl'
        print('Save absorption data to pickle save file: ' + savpkl)
        with open(savpkl, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([wcmdat, tprf, pprf, trnsc, ext, tau_in, tau_out, wloco, 
                         indlr, xx, yy, fsol, lay, nlay, pintf, dzf, intf, refl], f)

    else: # if no previous hdf5 save file exists of this band
        print('Restore absorption data from IDL save file: '+savpkl)
        with open(savpkl, 'rb') as f:
            (wcmdat, tprf, pprf, trnsc, ext, tau_in, tau_out, wloco, 
             indlr, xx, yy, fsol, lay, nlay, pintf, dzf, intf, refl) = pickle.load(f)
    # End: Extract information from oco_subroutine.absCOF file (native resolution)


    # *********
    # Sub-sampling, output & display

    wl = 10000./np.float64(wcmdat) 
    wlc = wloco.copy()

    # ** define spectral sub-range within band
    flt = np.logical_and(wl  >= wr[0], wl  < wr[1]).nonzero()[0]
    nf = len(flt)
    flc = np.logical_and(wlc >= wr[0], wlc < wr[1]).nonzero()[0]
    nfc = len(flc)
    wlf = wlc[flc]

    trnsx = trnsc[flc]   # extract spectral sub-range transmittance
    # sort column-integrated tau & two altitudes using transmittance (v2 of code: using OD)
    sx = np.argsort(trnsx)     # sort the transmittance
    trnsx = trnsx[sx] 
    wls = wlf[sx]   # these are the wavelengths within the sub-range

    if all_r > 0 :
        flr = (trnsx < np.max(trnsx)*all_r/100.).nonzero()[0]
        nx = len(flr)
        wlf = wlf[flr]
        trnsx = trnsx[flr]
        print(f'Now using {nx} wavelengths.')
        nx = nx-1

    g_mode = cfg_info['g_mode']
    if g_mode == 'TRUE':
        g_mode = True
    elif g_mode == 'FALSE':
        g_mode = False
    else:
        sys.exit('"g_mode" in the config file should be either "TRUE" or "FALSE".')
    g = 16
    wli, abs_g_final, prob_g_final, \
        weight_g_final, sol_g_final = oco_wv_select(trnsx, Trn_min, refl, nlay, nx, all_r, 
                                                    wlc, wlf, wls, wl, indlr, xx, yy, ext, fsol, iband,
                                                    g_mode, g=g)

    # ** Now extract all the absorption data for the wavelengths that correspond
    # ** to the OD increments (wavelengths are defined above)
    wx = np.empty(nx+1) # wavelengths
    lx = np.empty(nx+1, dtype=int) # wavelength indices in original gridding
    tx = np.empty(nx+1) # transmittance in original gridding
    for i in range(nx+1):
        if all_r > 0 :
            l0 = np.argmin(np.abs(wlc-wlf[i]))
        else:
            l0 = np.argmin(np.abs(wlc-wls[np.int(wli[i])]))
        wx[i] = wlc[l0]
        lx[i] = l0
        tx[i] = trnsc[l0]*refl

    if plot:
        title_size = 18
        label_size = 16
        legend_size = 16
        tick_size = 14

        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.5), sharex=False)
        fig.tight_layout(pad=5.0)
        ax.plot(wlc, trnsc*refl, color='k')
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.set_xlabel('Wavelength ($\mathrm{\mu m}$)', fontsize=label_size)
        ax.set_ylabel('Transmittance', fontsize=label_size)
        ax.set_title(f'Sub-band of {lb}', fontsize=title_size)
        ax.plot([wr[0],wr[0]],[0, 0.5], color='orange', linestyle='--', linewidth=3)
        ax.plot([wr[1],wr[1]],[0, 0.5], color='orange', linestyle='--', linewidth=3)
        for i in range(nx+1):
            ax.scatter(wlc[lx[i]], tx[i], facecolors='none', edgecolor='orange', marker='D')
        fig.savefig(f'{pathout}/band{iband}_1-transmittance_sat.png', dpi=150, bbox_inches='tight')

        plt.clf()
        norm = colors.Normalize(vmin=0.0, vmax=255.0, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.rainbow)
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.5), sharex=False)
        fig.tight_layout(pad=5.0)
        ax.scatter(np.arange(nfc), trnsx[sx]*refl, color='k', s=3)
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.set_xlabel('Wavelength index', fontsize=label_size)
        ax.set_ylabel('Transmittance', fontsize=label_size)
        ax.set_title(f'Sub-band of {lb}', fontsize=title_size)
        for i in range(nx+1):
            wli0 = wli[i]
            transmittance = trnsx[wli0]*refl
            ax.plot([0,nfc],[transmittance, transmittance],color='orange',linestyle='dotted')
            ax.plot([wli0,nfc],[trnsx[wli0]*refl,trnsx[wli0]*refl],linestyle='dashed',color='orange')
            cl = 30*(i+1)
            ax.plot([wli0, wli0], [0, transmittance], linestyle='dashed', color=mapper.to_rgba(cl), linewidth=2)        
        fig.savefig(f'{pathout}/band{iband}_2-wavelength_selection.png', dpi=150, bbox_inches='tight')

    # ** assign values / profiles for atm/abs file (final output)
    nlg       = np.max(indlr[:,2])  # max # of individual absco wl for each oco wl (within ILS)
    # todo: # z needs to be nz, not nz-1
    absgl     = np.empty((nlay,nx+1,nlg)) # absorption coefficient profile for nx+1 wavelengths & nlg individual absorption lines
    absgn     = np.empty(nx+1, dtype=int)          # # of absorption lines that have to be considered for each wl, given the ILS
    absgx     = np.empty((nx+1,nlg))      # ILS-lamdas for each of the selected OCO-2 wavelengths at ABSCO resolution
    absgy     = np.empty((nx+1,nlg))      # ILS-shape  for each of the selected OCO-2 wavelengths at ABSCO resolution
    atm_dz    = dzf*1000.             # layer thickness in m
    atm_zgrd  = intf*1000.            # altitudes [m]
    unit_z    = 'm'
    lamx      = wx                    # these are the sub-selected wavelengths from OCO-2
    unit_l    = 'nm'
    atm_p     = pprf                  # pressure profile for layers
    unit_p    = 'hPa'
    atm_temp  = tprf                  # temperature profile for layers
    unit_T    = 'K'
    solx      = np.empty((nx+1,nlg))      # solar irradiance for each of the selected OCO-2 wavelengths at full ABSCO resolution
    unit_abs  = '/km'
    # reminder: wl = ABSCO resolution !

    extcheck = np.empty((nx+1,nlay)) # check the extinction profile
    
    # nan initialization
    absgl[...] = np.nan
    absgx[...] = np.nan
    absgy[...] = np.nan
    solx[...] = np.nan
    
    for l in range(0, nx+1):
        absgx[l, 0:indlr[lx[l],2]]= wl[indlr[lx[l],1]:indlr[lx[l],0]+1] # ILS - xx (lamda)
        ilg0 = np.interp(wl[indlr[lx[l],1]:indlr[lx[l],0]+1], xx[lx[l], :]+wlc[lx[l]], yy[lx[l], :]) # partial slit function within valid range
        absgy[l, 0:indlr[lx[l], 2]] = ilg0 # ILS - yy ("weight")
        absgn[l]                   = indlr[lx[l],2] # additional stuff (# of k's)
        solx [l, 0:indlr[lx[l], 2]]= fsol[indlr[lx[l],1]:indlr[lx[l],0]+1]
        for z in range(0, nlay):
            absgl[z,l,0:indlr[lx[l],2]]=ext[indlr[lx[l],1]:indlr[lx[l],0]+1,z]
            if z == 0 and plot:
                plt.clf()
                fig, ax = plt.subplots(1, 1, figsize=(8, 3.5), sharex=False)
                fig.tight_layout(pad=5.0)
                ax.plot(absgx[l,0:absgn[l]-1], absgy[l,0:absgn[l]-1], color='k', label='ILS')
                ax.plot(absgx[l,0:absgn[l]-1], solx[l,0:absgn[l]-1], color='orange', label='solar')
                norm = np.max(absgl[z,l,0:absgn[l]-1])
                ax.plot(absgx[l,0:absgn[l]-1], absgl[z,l,0:absgn[l]-1]/norm, color='red', label='rel. abs coeff.')
                ax.tick_params(axis='both', labelsize=tick_size)
                ax.legend(fontsize=legend_size)
                ax.set_xlabel('Wavelength ($\mathrm{\mu m}$)', fontsize=label_size)
                ax.set_ylabel('normalized # photons/nm', fontsize=label_size)
                ax.set_title(f'# ILS terms {absgn[l]}', fontsize=title_size)
                fig.savefig(f'{pathout}/band{iband}_3-individual_line_at_wvl_{wx[l]:.5f}nm.png', dpi=150, bbox_inches='tight')
                    
            extcheck[l,z] = np.sum(absgl[z,l,0:absgn[l]-1]*absgy[l,0:absgn[l]-1])/np.sum(absgy[l,0:absgn[l]-1])

    # save output file
    if os.path.isfile(output): 
        print('[Warning] Output file exists - overwriting!')
    print('Saving to file '+output)
    wl_oco = wlc
    trns_oco = trnsc
    atm_pi = pintf
    if not g_mode:
        with h5py.File(output, 'w') as h5_output:
            h5_output.update({'atm_zgrd': atm_zgrd, 
                            'lay': lay, 
                            'atm_p': atm_p,
                            'atm_pi': atm_pi, 
                            'atm_temp': atm_temp, 
                            'atm_dz': atm_dz,
                            'lamx': lamx, 
                            'tx': tx, 
                            'absgl': absgl,
                            'absgn': absgn, 
                            'absgx': absgx, 
                            'absgy': absgy, 
                            'solx': solx, 
                            'ils0': ils0, 
                            'unit_z': unit_z, 
                            'unit_l': unit_l, 
                            'unit_p': unit_p, 
                            'unit_T': unit_T, 
                            'unit_abs': unit_abs, 
                            'wl_oco': wl_oco,
                            'trns_oco': trns_oco})
    else:
        with h5py.File(output, 'w') as h5_output:
            h5_output.update({'atm_zgrd': atm_zgrd, 
                            'lay': lay, 
                            'atm_p': atm_p,
                            'atm_pi': atm_pi, 
                            'atm_temp': atm_temp, 
                            'atm_dz': atm_dz,
                            'lamx': lamx, 
                            'tx': tx, 
                            'absgl': abs_g_final,
                            'absgn': np.array([g,]*(nx+1)), 
                            'absgx': np.ones((nx+1, g)), 
                            'absgy': weight_g_final, 
                            'solx': sol_g_final, 
                            'ils0': ils0, 
                            'unit_z': unit_z, 
                            'unit_l': unit_l, 
                            'unit_p': unit_p, 
                            'unit_T': unit_T, 
                            'unit_abs': unit_abs, 
                            'wl_oco': wl_oco,
                            'trns_oco': trns_oco})
    return None
    #todo: check if vertical resolution changes tau and/or ext

if __name__ == '__main__':
    None