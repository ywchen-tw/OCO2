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
from utils.abs.read_atm import read_oco_zpt
from utils.abs.rdabs_gas import rdabs_species
from utils.abs.find_bound import find_boundary # get wavenumber indices.initialize()import abs/get_index  # find levels in absco files that are closest to the atmosphere
from utils.abs.get_index import get_PT_index  # find levels in absco files that are closest to the atmosphere.initialize()import abs/rdabscoo2.pro   # read absorption coefficients O2
from utils.abs.rdabsco_gas import rdabsco_species   # read absorption coefficients O2.initialize()import abs/rdabscoco2.pro  # read absorption coefficients CO2
from utils.abs.calc_ext import calc_ext   # calculates extinction profiles & layer transmittance from oco_utils.absorption coefficients & density profiles (CO2, O2)
from utils.abs.oco_wl import oco_wv      # reads OCO wavelengths
from utils.abs.oco_ils import oco_ils    # reads OCO line shape ("slit function")    
from utils.abs.solar import solar   # read solar file
from utils.abs.oco_convolve import oco_conv
from utils.abs.oco_abs_g_mode import oco_wv_select
from utils.oco_util import timing
from utils.oco_cfg import grab_cfg


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

    pathinp = './utils/abs/'
    if pathout == None:
        pathout = './'

    all_r   = 0       # 0: use T-sampling >0: full range & percentage of transmittance
    if nx < 0:
        nx = 5       # of representative wavelengths @ native resolution of Vivienne's data base
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
    files = ['o2_v51.hdf',    # O2
             'co2_v51.hdf',   # CO2 weak
             'co2_v51.hdf',   # CO2 strong
             'h2o_v51.hdf',   # H2O
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
    # Extract information from oco_utils.absCOF files (native resolution).
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
        
        
        filnm = files[iband]  # absco hdf5 file for target band

        # *********
        # Specify the wavelength (wavenumber) range to work with
        wavel1, wavel2 = xr # (micron)
        wcm2, wcm1 = 1.0e4/wavel1, 1.0e4/wavel2 # (cm-1)
        refl = [alb_o2a, alb_wco2, alb_sco2][iband] #s urface reflectance

        # *********
        # Calculate the 1/cos factors for solzen & obszen geometry
        convr = np.pi/180.0
        musolzen = 1.0/(np.cos(solzen*convr))
        muobszen = 1.0/(np.cos(obszen*convr))

        # *********
        if iband == 0:
            rdabs_gas = 'o2'
            rdabs_gas_den = o2den
        elif ((iband == 1) | (iband == 2)):
            rdabs_gas = 'co2'
            rdabs_gas_den = co2den

        # Read target gas absco atmosphere information
        # Find out the P, T, boadening, wavenumber(wavelength) grid within
        # the target gas's absco file 
        wcm_gas, p_gas, tk_gas, broad_gas, hpa_gas, units_gas = rdabs_species(filnm=filnm, species=rdabs_gas)
        
        # *********
        # Obtain the wavenumber indices to work with target gas
        # define subset of wavenumber (wcmdat) & wavelength grid (wavedat)
        # from ABSCO file, based on wavelength interval as specified by the user
        iwcm1, iwcm2, nwav, wcmdat, wavedat = find_boundary(wcm1, wcm2, wcm_gas)

        # *********
        # Read in H2O atmosphere information
        # Find out the pressure,temperature,boadening,wavenumber(wavelength) grid within
        # the H2O absco file 
        wcmh2o, ph2o, tkh2o, broadh2o, hpah2o, unitsh2o = rdabs_species(filnm=h2o, species='h2o')
        
        # *********
        # Obtain the wavenumber indices to work with for H2O
        # define subset of wavenumber (wcmdat) & wavelength grid (wavedat)
        # from H2O ABSCO file, based on wavelength interval as specified by the user
        iwcm1h2o, iwcm2h2o, nwavh2o, wcmdath2o, wavedath2o = find_boundary(wcm1, wcm2, wcmh2o)
            
        if nwav != nwavh2o: 
            print('[Warning] Wavenumber gridding of O2 & H2O ABSCO files do not match.')                

        # *********
        # Read out that actual absorption information from the data base

        # *********
        # Initialize
        # Optical depths on interfaces & absorption coefficients on layers
        ext = np.empty((nwav,nlay))
        tau_in = np.zeros(nwav)
        tau_out = np.zeros(nwav)

        # *********
        # Start at top of atmosphere & go to the surface
        for iz in range(nlay)[::-1]: 
        # ADD FUNCTIONALITY    print,'  iz ',iz
            tkobs, pobs = tprf[iz], pprf[iz]
            ext0  = np.zeros(nwav) # absorption coef for O2 or CO2
            ext1  = np.zeros(nwav) # absorption coef for H2O
            ext0[...] = np.nan
            ext1[...] = np.nan

            # *********
            # First find the indices T_ind & P_ind (P_ind pressure, T_ind temperature)
            # that are closest to tkobs & pobs (with pobs ij hPa)

            trilinear_opt = abs_inter=='trilinear'

            # (1) get {p,T} indices within target gas ABSCO grid
            T_ind, P_ind = get_PT_index(tkobs, pobs, tk_gas, hpa_gas, trilinear_opt)

            if np.abs(pobs-hpa_gas[P_ind]) > pdmax: 
                print(f'[Warning] {rdabs_gas.upper()} pressure grid too coarse - interpolate?')
            if np.abs(tkobs-tk_gas[P_ind, T_ind]) > tdmax: 
                print(f'[Warning] {rdabs_gas.upper()} Temperature grid too coarse - interpolate?')

            # (2) get {p,T} indices within H2O ABSCO grid
            T_ind_h2o,P_ind_h2o = get_PT_index(tkobs, pobs, tkh2o, hpah2o, trilinear_opt)

            if np.abs(pobs-hpah2o[P_ind_h2o]) > pdmax: 
                print('[Warning] H2O pressure grid too coarse - interpolate?')
            if np.abs(tkobs-tkh2o[P_ind_h2o, T_ind_h2o]) > tdmax: 
                print('[Warning] H2O temperature grid too coarse - interpolate?')
            # ----- for check P, T fields only -----
            # print(f'p ({rdabs_gas.upper()}, H2O) : {pobs:.2f} hPa, {hpa_gas[P_ind]:.2f} hPa, {hpah2o[P_indh2o]:.2f} hPa')
            # print(f'T ({rdabs_gas.upper()}, H2O) : {tkobs:.2f} K, {tk_gas[P_ind, T_ind]:.2f} K, {tkh2o[P_ind_h2o, T_ind_h2o]:.2f} K')
            # -------------------------------------- 

            # Specify the absorption coefficients - here is where we actually read them!!!
            # (An accurate calculation would use Lagrange interpolation
            # amongst three temperature & three pressure sets of absco vectors)

            # For O2 or CO2
            absco = rdabsco_species(filnm, p_gas, tk_gas, broad_gas,
                                    hpa_gas, tkobs, pobs, 
                                    T_ind, P_ind, h2o_vmr[iz], iwcm1, iwcm2,
                                    species=rdabs_gas, mode=abs_inter,)

            # For H2O  
            abscoh2o = rdabsco_species(h2o, ph2o, tkh2o, broadh2o,
                                       hpah2o, tkobs, pobs,
                                       T_ind_h2o, P_ind_h2o, h2o_vmr[iz], iwcm1h2o, iwcm2h2o,
                                       species='h2o', mode=abs_inter,)
            
            # *********
            # Use the absco coefficients
            # Note that absco coefficients are cm-2 per molecule
            # and that the density profiles are in molecules per cm3
            # so that the extinction coefficients are in cm-1 but converted to km-1

            ext0 = calc_ext(rdabs_gas_den, iz, absco) # For O2 or CO2
            ext1 = calc_ext(h2oden, iz, abscoh2o)     # For H2O

            # Store the results in ext
            ext[:,iz] = ext0 + ext1 # ext0: O2/CO2, ext1: H2O
            tau_in  += ext[:,iz] * dzf[iz] * musolzen
            tau_out += ext[:,iz] * dzf[iz] * muobszen

            # End of loop down to the surface     
        trns = np.exp(-(tau_in+tau_out))
        # *********
        # End of loop over all layers

        wloco = oco_wv(iband, sat, footprint=fp) # (micron)
        xx, yy = oco_ils(iband, sat, footprint=fp)
        trnsc, trnsc0, indlr, ils = oco_conv(iband, sat, ils0, wavedat, nwav, trns, fp=fp)

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
                         indlr, ils, xx, yy, fsol, lay, nlay, pintf, dzf, intf, refl], f)

    else: # if no previous hdf5 save file exists of this band
        print('Restore absorption data from IDL save file: '+savpkl)
        with open(savpkl, 'rb') as f:
            (wcmdat, tprf, pprf, trnsc, ext, tau_in, tau_out, wloco, 
             indlr, ils, xx, yy, fsol, lay, nlay, pintf, dzf, intf, refl) = pickle.load(f)
    # End: Extract information from oco_utils.absCOF file (native resolution)


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
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
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
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
        fig.tight_layout(pad=5.0)
        ax.scatter(np.arange(nfc), trnsx*refl, color='k', s=3)
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

        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
        fig.tight_layout(pad=5.0)
        x = xx.mean(axis=0)*1000
        ax.plot(x, yy.mean(axis=0), color='k')
        ax.vlines([x[ils[0]], x[ils[-1]]], ils0, 1, 'r')
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.set_xlabel('$\mathrm{\Delta}$ Wavelength (nm)', fontsize=label_size)
        ax.set_ylabel('response', fontsize=label_size)
        ax.set_title('# ILS terms', fontsize=title_size)
        fig.savefig(f'{pathout}/band{iband}_3_ILS_mean.png', dpi=150, bbox_inches='tight')

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
                fig.savefig(f'{pathout}/band{iband}_4-individual_line_at_wvl_{wx[l]:.5f}nm.png', dpi=150, bbox_inches='tight')
                    
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