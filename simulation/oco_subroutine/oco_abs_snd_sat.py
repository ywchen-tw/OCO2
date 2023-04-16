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
# TODOs:
#  
#  
#  Done:
#  - generalize to SCO2 & WCO2 for water vapor/methane inclusion & consolidate sub-routines
#  - get accurate wavelengths from recent cal & accurate OCO line shape (but just from csv file for now)
#  - Remove getdatah5, getdatah5a by just use h5py module
#  - implement H2O broadening
#  - bi/trilinear interpolation in {T,p,(h)} - can re-use Odele's code
#---------------------------------------------------------------------------
from oco_subroutine.abs.read_atm import read_oco_zpt
from oco_subroutine.abs.rho_air import rho_air  # density of air
from oco_subroutine.abs.rdabs_gas import rdabs_species
from oco_subroutine.abs.findi1i2_v7 import findi1i2 # get wavenumber indices.initialize()import abs/getiijj_v7.pro  # find levels in absco files that are closest to the atmosphere
from oco_subroutine.abs.getiijj_v7 import getiijj  # find levels in absco files that are closest to the atmosphere.initialize()import abs/rdabscoo2.pro   # read absorption coefficients O2
from oco_subroutine.abs.rdabsco_gas import rdabsco_species   # read absorption coefficients O2.initialize()import abs/rdabscoco2.pro  # read absorption coefficients CO2
from oco_subroutine.abs.calc2_v8 import calc2   # calculates extinction profiles & layer transmittance from oco_subroutine.absorption coefficients & density profiles (CO2, O2)
from oco_subroutine.abs.oco_wl import oco_wv      # reads OCO wavelengths
from oco_subroutine.abs.oco_ils import oco_ils    # reads OCO line shape ("slit function")    
from oco_subroutine.abs.solar import solar   # read solar file
from oco_subroutine.abs.oco_snd import *

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def oco_abs(cfg, zpt_file, iband=0, nx=None, Trn_min=0, pathout=None, reextract=True, plot=False):
    # *********
    # Specify which band to work with
    # iband=0,1,2 for o2, weak co2, strong co2

    if not iband in [0, 1, 2]:
        print('iband should be either 0 (O2), 1 (weak CO2), or 2 (Strong CO2), set to the default 0.')
        iband   = 0

    pathinp = './oco_subroutine/abs/'
    if pathout == None:
        pathout = './'

    save    = 1       # save final data? 
    pl_ils  = 1       # plot ils, else result of convolution
    all_r   = 0       # 0: use T-sampling >0: full range & percentage of transmittance
    if nx < 0:
        nx = 5       # # of representative wavelengths @ native resolution of Vivienne's data base
    ils0        = 0.05  # lowest level within ILS that will be included in calculations
                # standard: 0.002 --> 0.2% cutoff for lines 
                # (not worth calculating them)
    pdmax       = 100   # max difference between requested & available atmospheric pressure level
    tdmax       = 10    # max difference between requested & available atmospheric temp value
    sol         = pathinp+'sol/solar.txt' # high spectral resolution solar file at 1 AU

    f = open("log.txt", "w")
    f.close()
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
    
    wr_dict = {0: [0.758, 0.773],   #o2
               1: [1.593, 1.622],   #wco2
               2: [2.043, 2.083],}  #sco2

    xr_dict = {0: [0.757, 0.773],   #o2
               1: [1.590, 1.623],   #wco2
               2: [2.042, 2.085],}  #sco2
    
    lb_dict = {0: 'o2a', 1: 'wco2', 2: 'sco2',}
               
    wr = wr_dict[iband]     # wr = sub-range
    xr = xr_dict[iband]     # xr = range of entire band
    lb = lb_dict[iband]
    output = pathout+'/atm_abs_'+lb+'_'+str(nx+1)+'.h5' 


    print('***********************************************************', file=sys.stderr)
    print('Make MCARATS atmosphere/gas absorption input for '+lb, file=sys.stderr)
    print('***********************************************************', file=sys.stderr)


    # Files that contain the ABSCO coefficients
    files = ['o2_v51.hdf',         # O2
             'co2_v51.hdf',   # CO2 weak
             'co2_v51.hdf',   # CO2 strong
             'h2o_v51.hdf',               # H2O
            ]
    nfiles = len(files)
    files = [pathinp + i for i in files]
    h2o = files[3]

    # Get OCO sounding info
    cfg_info = grab_cfg(cfg)
    abs_inter = cfg_info['abs_interpolation']
    sat = get_sat(cfg)

    wlsol0, fsol0 = solar(sol)

    # User specifications are done

    # ******************************************************************************************************
    # *********
    # Extract information from oco_subroutine.absCOF files (native resolution).
    # Check whether this was previously extracted.
    savpkl = f'{pathout}/{files[iband]}.pkl'
    if not os.path.isfile(savpkl) or reextract==True:

        # *********
        # Identify which file is for co2, & which is for o2
        # These numbers are file specific ! Careful !
        io2   = 0
        ico2w = 1
        ico2s = 2

        

        # *********
        # Read in z,p,t profile from radiosonde, calculate layer & interface properties

        intf, pintf, pprf, tprf, o2den, co2den, h2oden, h2o_vmr, dzf, lay, alb_o2a, alb_wco2, alb_sco2, sza, vza = read_oco_zpt(zpt_file=zpt_file)
        # p,t,number concentrations for layers (not interfaces)

        nlay=len(lay)


        # ********
        # Have nadir geometry (zolzen=0.0, don't change this value)
        # Specify the solar zenith angle & satellite view angle
        solzen = sza        # in degree

        # Ignore obszen
        obszen = vza        # in degree        

        # *********
        # Specify the wavelength (wavenumber) range to work with
        #   refl is the surface reflectance (ignore)
        #   jbroado2 is the broadening index that is used
        if (iband == 0) :
            wavel1=xr[0]
            wavel2=xr[1]
            jbroado2=1
            refl = alb_o2a
            

        # Weak CO2 band
        if (iband == 1) :
            wavel1=xr[0]
            wavel2=xr[1]
            refl = alb_wco2
            jbroadco2=1
            

        # Strong CO2 band
        if (iband == 2) :
            wavel1=xr[0]
            wavel2=xr[1]
            refl = alb_sco2
            jbroadco2=1
            

        # todo: what is the broadening for the H2O self-broadening (if there
        # is such a thing?
        jbroadh2o=0
        
        # *********
        # Calculate the wavenumbers for the two wavelengths
        wcm2 = 1.0e4/wavel1
        wcm1 = 1.0e4/wavel2

        # *********
        # Calculate the 1/cos factors for solzen & obszen geometry
        pi = np.pi
        convr = pi/180.0
        a1 = solzen*convr
        musolzen = 1.0/(np.cos(a1))
        a2 = obszen*convr
        muobszen = 1.0/(np.cos(a2))

        # *********
        # Read in O2 absco atmosphere information
        if iband == 0 :
            print(iband, file=sys.stderr)
            # absco hdf5 file for o2
            filnm = files[io2]

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
            if (iband == 1) :       # absco hdf5 file for weak co2 band
                filnm=files[ico2w]  
            elif (iband == 2) :       # absco hdf5 file for strong co2 band
                filnm=files[ico2s]

            # Find out the pressure,temperature,broadening,wavenumber info
            # species names for wco2 and sco2 are same
            npco2,ntkco2,nbroadco2,nwcmco2, wcmco2,pco2,tkco2, \
            broadco2, hpaco2,wavelco2, nunits,unitsco2 = rdabs_species(filnm=filnm, species='co2', iout=True)
                    
            # *********
            # Obtain the wavenumber indices to work with
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
            print('[Warning] Wavenumber gridding of O2 & H2O ABSCO files do not match.', file=sys.stderr)                

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
            print(f'  iz  {iz}', file=sys.stderr)
            tkobs = tprf[iz]
            pobs  = pprf[iz]

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
                    print('[Warning] O2 pressure grid too coarse - interpolate?', file=sys.stderr)
                if np.abs(tkobs-tko2[jj, ii]) > tdmax: 
                    print('[Warning] O2 Temperature grid too coarse - interpolate?', file=sys.stderr)

                # (2) get {p,T} indices within H2O ABSCO grid
                iih2o,jjh2o = getiijj(tkobs, pobs, tkh2o, hpah2o, trilinear=trilinear_opt, iout=True)

                if np.abs(pobs-hpah2o[jjh2o]) > pdmax: 
                    print('[Warning] H2O pressure grid too coarse - interpolate?', file=sys.stderr)
                if np.abs(tkobs-tkh2o[jjh2o, iih2o]) > tdmax: 
                    print('[Warning] H2O temperature grid too coarse - interpolate?', file=sys.stderr)
                print('p :',pobs ,hpao2[jj],hpah2o[jjh2o], file=sys.stderr)
                print('T :',tkobs,tko2[jj, ii],tkh2o[jjh2o, iih2o], file=sys.stderr)
                
            # For co2
            if ((iband == 1) | (iband == 2)) :
                # (1) get {p,T} indices within WCO2 ABSCO grid
                ii, jj = getiijj(tkobs, pobs, tkco2, hpaco2, trilinear=trilinear_opt, iout=True)

                # (2) get {p,T} indices within H2O ABSCO grid
                iih2o,jjh2o = getiijj(tkobs, pobs, tkh2o, hpah2o, trilinear=trilinear_opt, iout=True)

                if np.abs(pobs-hpah2o[jjh2o]) > pdmax: 
                    print('[Warning] H2O pressure grid too coarse - interpolate?', file=sys.stderr)
                if np.abs(tkobs-tkh2o[jjh2o, iih2o]) > tdmax: 
                    print('[Warning] H2O temperature grid too coarse - interpolate?', file=sys.stderr)
                print('p (CO2, H2O) :',pobs ,hpaco2[jj],hpah2o[jjh2o], file=sys.stderr)
                print('T (CO2, H2O) :',tkobs,tkco2[jj, ii],tkh2o[jjh2o, iih2o], file=sys.stderr)                    
            
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

            # For O2
             # For O2
            if iband == 0:
                ext0 = calc2(dzf,pprf,tprf,
                             o2den,
                             iz,solzen,musolzen,
                             nwav,wcmdat,wavedat,
                             absco, trns,
                             iout=True)

            # For CO2
            if ((iband == 1) | (iband == 2)) :
                ext0 = calc2(dzf,pprf,tprf,
                             co2den,
                             iz,solzen,musolzen,
                             nwav,wcmdat,wavedat,
                             absco, trns,
                             iout=True)

            # For H2O
            ext1 = calc2(dzf,pprf,tprf,
                            h2oden,
                            iz,solzen,musolzen,
                            nwavh2o,wcmdath2o,wavedath2o,
                            abscoh2o, trns,
                            iout=True)   

            # Store the results in ext
            ext[:,iz] = ext0 + ext1 # ext0: O2/CO2, ext1: H2O
            # tau did not change in calc2 function for now!!!
            tau_in += ext[:,iz] *dzf[iz]*musolzen
            tau_out += ext[:,iz] *dzf[iz]*musolzen


            # End of loop down to the surface     
        trns = np.exp(-(tau_in+tau_out))#*refl
        #trns = np.exp(-(tau_out))
        
        # *********
        # Get OCO wavelengths & slit function
        # ---------------------------------------------
        #       sampling interval (nm)      FWHM (nm)
        # O2A:      0.015                       0.04
        # WCO2:     0.031                       0.08
        # SCO2:     0.04                        0.10
        # ---------------------------------------------
        # (1) read wavelengths
        wloco = oco_wv(iband) # (micron)
        nlo = len(wloco)
        # (2) read instrument line shape
        xx, yy = oco_ils(iband, sat) # xx: relative wl shift (nm)# yy: normalized ILS
        xx  = xx*0.001 # convert xx into micron
        nils= len(xx)
        # (3) convolute tau & trns across entire wavelength range -- & how about kval
        ils = yy/np.max(yy) >= ils0	
        nils0 = ils.nonzero()[0]
        trnsc = np.empty(nlo)
        tauc  = np.empty(nlo)
        trnsc0 = np.empty(nlo) 
        indlr = np.empty((nlo,3), dtype=int) # left & right index for cropped ILS (in ABSCO gridding) + total #

        for l in range(nlo):
            # get wl range within absco that falls within the ILS (total range) & within pre-set threshold
            abswlL = wloco[l] + np.min(xx)      # left  full range
            abswlR = wloco[l] + np.max(xx)      # right full range
            abswlL0 = wloco[l] + np.min(xx[ils]) # left  "valid" range (ILS above threshold ils0)
            abswlR0 = wloco[l] + np.max(xx[ils]) # right "valid" range (ILS above threshold ils0)
            il = np.argmin(np.abs(wavedat-abswlL))
            ir = np.argmin(np.abs(wavedat-abswlR))
            il0 = np.argmin(np.abs(wavedat-abswlL0))
            ir0 = np.argmin(np.abs(wavedat-abswlR0))
            indlr[l,0] = il0        # left index
            indlr[l,1] = ir0        # right index
            indlr[l,2] = il0-ir0+1  # how many
            if ir0 == 0: 
                print('[Warning] Range exceeded (R)')
            if il0 == nwav-1: 
                print('[Warning] Range exceeded (L)')
            if ir  >= il : 
                print('[Warning] Something wrong with range/ILS')
            if ir0 >= il0: 
                print('[Warning] Something wrong with range/ILS0')
            # actual convolution ---
            ilg = np.interp(wavedat[ir:il], xx+wloco[l], yy)                # full slit function in absco gridding
            ilg0 = np.interp(wavedat[ir0:il0], xx[ils]+wloco[l], yy[ils])   # partial slit function within valid range
            trnsc[l] = np.sum(trns[ir:il]*ilg)/np.sum(ilg)                  # ir:il because it is descending in wl
            trnsc0[l] = np.sum(trns[ir0:il0]*ilg0)/np.sum(ilg0)
            #tauc[l] = np.sum(tau[ir:il]*ilg)/np.sum(ilg)



        # (4) plots
        if plot and pl_ils:
            fig, ax = plt.subplots(1, 1, figsize=(8, 3.5), sharex=False)
            fig.tight_layout(pad=5.0)
            title_size = 18
            label_size = 16
            legend_size = 16
            tick_size = 14

            x = xx*1000
            #y = trnsx[sx]
            ax.plot(x, yy, color='k')
            ax.vlines([x[ils[0]], x[ils[-1]]], ils0, 1, 'r')
            #ax.plot(1000*np.array([wlc[l], wlc[l]]),[ils0,1], linestyle='--')

            #norm = np.max(absgl[z,l,0:absgn[l]-1])
            #ax.plot(1000*absgx[l,0:absgn[l]-1], absgl[z,l,0:absgn[l]-1]/norm, color='red')

            #ax.set_xticks(range(0, 160, 20))
            ax.tick_params(axis='both', labelsize=tick_size)

            ymin, ymax = ax.get_ylim()
            xmin, xmax = ax.get_xlim()
            #ymin, ymax = -1., 1.
            #xmin, xmax = 0., 10.
            #ax.set_ylim(ymin, ymax)
            #ax.set_xlim(xmin, xmax)
            ax.set_xlabel('Wavelength (nm)', fontsize=label_size)
            ax.set_ylabel('response', fontsize=label_size)
            ax.set_title('# ILS terms', fontsize=title_size)
            fig.savefig(f'{pathout}/band{iband}_4-test.png', dpi=150, bbox_inches='tight')

        # read solar file
        wlsol0, fsol0 = solar(sol) # obtain irradiance [W m-2 nm-1]
        fsol = np.interp(wavedat, wlsol0, fsol0)

        # *** Save file contains abs data for pre-selected wavelength range & specified atmosphere
        # *** along with
        # *** ---1) ILS/ILS_index(ILS resolution),
        # *** ---2) wloco (OCO-2 band resulution),
        # *** ---3) solar file (interpolated to absorption data base resolution)
        # *** When changing atmosphere | wavelength range, need to re-read (run with flag /r)
        savpkl = filnm+'.pkl'
        print('Save absorption data to pickle save file: ' + savpkl)
        with open(savpkl, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([wcmdat,tprf,pprf,trnsc,tauc,ext,tau_in, tau_out,wloco,indlr,xx,yy,fsol,lay,nlay,pintf,dzf,intf], f)

    else: # if no previous IDL save file exists of this band
        print('Restore absorption data from IDL save file: '+savpkl)
        with open(savpkl, 'rb') as f:
            wcmdat,tprf,pprf,trnsc,tauc,ext,tau_in, tau_out,wloco,indlr,xx,yy,fsol,lay,nlay,pintf,dzf,intf = pickle.load(f)
    # End: Extract information from oco_subroutine.absCOF file (native resolution)


    # *********
    # Sub-sampling, output & display
    if (save == 1 | plot == 1):
        wl = 10000./np.float64(wcmdat) 
        nl = len(wl)
        wlc = wloco.copy()
        tauex = tauc.copy() # column-integrated absorption optical thickness
        tauex0 = tauex.copy()
        # ** define spectral sub-range within band
        # ** & extract k values at two altitudes (surface & higher) 
        flt = np.logical_and(wl  >= wr[0], wl  < wr[1]).nonzero()[0]
        nf = len(flt)
        flc = np.logical_and(wlc >= wr[0], wlc < wr[1]).nonzero()[0]
        nfc = len(flc)
        wlf = wlc[flc]
        ###k1=kval[flt,ai1]   # at altitude
        ###k0=kval[flt,ai0]   # surface 
        tauex = tauex[flc]   # extract spectral sub-range OD
        trnsx = trnsc[flc]   # extract spectral sub-range transmittance
        tlf = trnsx
        sx = np.argsort(trnsx)     # can also use tauex, but now trying trnsx (for linearity# range)
        # *********

        if plot == 1 :
            plt.clf()
            fig, ax = plt.subplots(1, 1, figsize=(8, 3.5), sharex=False)
            fig.tight_layout(pad=5.0)
            title_size = 18
            label_size = 16
            legend_size = 16
            tick_size = 14
            x = np.arange(nfc)
            y = trnsx[sx]*refl
            ax.scatter(x, y, color='k', s=3)
            
            #ax.set_xticks(range(0, 160, 20))
            ax.tick_params(axis='both', labelsize=tick_size)

            ymin, ymax = ax.get_ylim()
            xmin, xmax = ax.get_xlim()
            #ymin, ymax = -1., 1.
            #xmin, xmax = 0., 10.
            #ax.set_ylim(ymin, ymax)
            #ax.set_xlim(xmin, xmax)
            ax.set_xlabel('Wavelength index', fontsize=label_size)
            ax.set_ylabel('Transmittance', fontsize=label_size)
            ax.set_title('Sub-band of '+lb, fontsize=title_size)

        # sort column-integrated tau & two altitudes using transmittance (v2 of code: using OD)
        tauex = tauex[sx]
        trnsx = trnsx[sx] 
        ###k1   =k1[sx]
        ###k0   =k0[sx]
        wls = wlf[sx]   # these are the wavelengths within the sub-range

        # ** do spectral sub-sample using equi-distant transmittance values
        mx = np.max(trnsx)            # maximum transmittance (T) within spectral sub-range
        mn = np.max([Trn_min*np.max(trnsx), np.min(trnsx)])    # minimum transmittance with spectral sub-range
        m0 = (mx-mn)/np.float64(nx)  # T increments
        ods = np.empty(nx+1) # T sorted (nx sub-samples)
        wli = np.empty(nx+1) # wl index
        # plot setting
        norm = colors.Normalize(vmin=0.0, vmax=255.0, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.rainbow)
        for i in range(0, nx+1):
            ods[i] = (m0*np.float64(i)+mn)*refl
            wli0 = np.argmin(np.abs(ods[i]-trnsx*refl))
            print(i, wli0, ods[i], trnsx[wli0]*refl)
            wli[i] = wli0
            if plot == 1 :
                ax.plot([0,nfc],[ods[i],ods[i]],color='orange',linestyle='dotted')
                ax.plot([wli0,nfc],[trnsx[wli0]*refl,trnsx[wli0]*refl],linestyle='dashed',color='orange')
                cl = 30*(i+1)
                if cl == 0: 
                    cl=255
                ax.plot([wli0,wli0], [0,ods[i]], linestyle='dashed', color=mapper.to_rgba(cl), linewidth=2)
                    
        # ods[nx] = mn*refl
        # wli0 = np.argmin(np.abs(ods[nx]-trnsx*refl))
        # wli[nx]=wli0
        # print(nx, wli0, ods[nx], trnsx[wli0]*refl)
        # cl = 30*nx
        # ax.plot([wli0,wli0],[0,ods[nx]],linestyle='dashed',color=mapper.to_rgba(cl), linewidth=2)
        # ax.plot([wli0,nfc],[trnsx[wli0]*refl,trnsx[wli0]*refl],linestyle='dashed',color='orange')
        fig.savefig(f'{pathout}/band{iband}_2-wavelength_selection.png', dpi=150, bbox_inches='tight')
        

        if plot == 1 :
            plt.clf()
            fig, ax = plt.subplots(1, 1, figsize=(8, 3.5), sharex=False)
            fig.tight_layout(pad=5.0)
            title_size = 18
            label_size = 16
            legend_size = 16
            tick_size = 14
            x = wlc
            y = trnsc*refl

            ax.plot(x, y, color='k')
            ax.tick_params(axis='both', labelsize=tick_size)

            #ax.set_ylim(ymin, ymax)
            #ax.set_xlim(xr[0],xr[1])
            #ax.legend(loc='center left', bbox_to_anchor=(0.65, 0.15), fontsize=legend_size)
            ax.set_xlabel('Wavelength [$\mathrm{\mu m}$]', fontsize=label_size)
            ax.set_ylabel('Transmittance', fontsize=label_size)
            #ax.set_title('Sub-band of '+lb, fontsize=title_size)
            
            #!P.multi=[0,1,2]
            #plot,wlc,tauex0,xr=xr,xtit='Wavelength [micron]',ytit='Absorption Optical Thickness',chars=1.5 
            #mxk=max(tauex0)
            #oplot,[wr[0],wr[0]],[0,mxk],color=120,linesty=2,thick=3
            #oplot,[wr[1],wr[1]],[0,mxk],color=120,linesty=2,thick=3
            #for i=0,nx do begin
            #  oplot,[wls[wli[i]],wls[wli[i]]],[0,tauex[wli[i]]],color=120,thick=2
            #  plots,[wls[wli[i]],tauex[wli[i]]],color=120,psym=4,syms=2
            #
            ax.plot([wr[0],wr[0]],[0,0.5], color='orange', linestyle='--', linewidth=3)
            ax.plot([wr[1],wr[1]],[0,0.5], color='orange', linestyle='--', linewidth=3)
            #!P.multi=0
            
        if all_r > 0 :
            flr = (tlf < np.max(tlf)*all_r/100.).nonzero()[0]
            nx = len(flr)
            wlf = wlf[flr]
            tlf = tlf[flr]
            print(f'Now using {nx} wavelengths.')
            nx = nx-1
            
        # ** Now extract all the absorption data for the wavelengths that correspond
        # ** to the OD increments (wavelengths are defined above)
        wx = np.empty(nx+1) # wavelengths
        lx = np.empty(nx+1, dtype=np.int64) # wavelength indices in original gridding
        tx = np.empty(nx+1) # transmittance in original gridding
        for i in range(0, nx+1):
            if all_r > 0 :
                l0 = np.argmin(np.abs(wlc-wlf[i]))
            else:
                l0 = np.argmin(np.abs(wlc-wls[np.int(wli[i])]))
            wx[i] = wlc[l0]
            lx[i] = l0
            tx[i] = trnsc[l0]*refl
            if plot == 1: 
                ax.scatter(wlc[lx[i]], tx[i], facecolors='none', edgecolor='orange', marker='D')
            
        if plot == 1 :
            fig.savefig(f'{pathout}/band{iband}_1-transmittance_sat.png', dpi=150, bbox_inches='tight')
            #plt.show()
            
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
        solx[...] = np.nan
        unit_abs  = '/km'
        # reminder: wl = ABSCO resolution !

        extcheck = np.empty((nx+1,nlay)) # check the extinction profile
        for l in range(0, nx+1):
            absgx[l, 0:indlr[lx[l],2]]= wl[indlr[lx[l],1]:indlr[lx[l],0]+1] # ILS - xx (lamda)
            ilg0 = np.interp(wl[indlr[lx[l],1]:indlr[lx[l],0]+1], xx+wlc[lx[l]], yy) # partial slit function within valid range
            absgy[l, 0:indlr[lx[l], 2]] = ilg0 # ILS - yy ("weight")
            absgn[l]                   = indlr[lx[l],2] # additional stuff (# of k's)
            solx [l, 0:indlr[lx[l], 2]]= fsol[indlr[lx[l],1]:indlr[lx[l],0]+1]
            print('-'*15, l,  file=sys.stderr)
            print(fsol[indlr[lx[l],1]:indlr[lx[l],0]+1],  file=sys.stderr)
            print(fsol[indlr[lx[l],1]:indlr[lx[l],0]+1].min(), fsol[indlr[lx[l],1]:indlr[lx[l],0]+1].max(), file=sys.stderr)
            for z in range(0, nlay):
                absgl[z,l,0:indlr[lx[l],2]]=ext[indlr[lx[l],1]:indlr[lx[l],0]+1,z]
                if z == 0 and plot:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5), sharex=False)
                    fig.tight_layout(pad=5.0)
                    title_size = 18
                    label_size = 16
                    legend_size = 16
                    tick_size = 14

                    x = np.arange(nfc)
                    y = trnsx[sx]
                    ax.plot(1000*absgx[l,0:absgn[l]-1], absgy[l,0:absgn[l]-1], color='k')
                    #ax.plot(1000*np.array([wlc[l], wlc[l]]),[ils0,1], linestyle='--')
                    ax.plot(1000*absgx[l,0:absgn[l]-1], solx[l,0:absgn[l]-1], color='orange')

                    norm = np.max(absgl[z,l,0:absgn[l]-1])
                    ax.plot(1000*absgx[l,0:absgn[l]-1], absgl[z,l,0:absgn[l]-1]/norm, color='red')

                    #ax.set_xticks(range(0, 160, 20))
                    ax.tick_params(axis='both', labelsize=tick_size)

                    ymin, ymax = ax.get_ylim()
                    xmin, xmax = ax.get_xlim()
                    #ymin, ymax = -1., 1.
                    #xmin, xmax = 0., 10.
                    #ax.set_ylim(ymin, ymax)
                    #ax.set_xlim(xmin, xmax)
                    ax.set_xlabel('Wavelength (nm)', fontsize=label_size)
                    ax.set_ylabel('normalized # photons/nm', fontsize=label_size)
                    ax.set_title(f'# ILS terms {absgn[l]}', fontsize=title_size)
                    fig.savefig(f'{pathout}/band{iband}_3-test.png', dpi=150, bbox_inches='tight')

                        
                extcheck[l,z]=np.sum(absgl[z,l,0:absgn[l]-1]*absgy[l,0:absgn[l]-1])/np.sum(absgy[l,0:absgn[l]-1])
            

        if save == 1 :
            if os.path.isfile(output): 
                print('[Warning] Output file exists - overwriting!')
            print('Saving to file '+output)
            wl_oco = wlc
            trns_oco = trnsc
            atm_pi = pintf

            h5_output = h5py.File(output, 'w')
            h5_output.create_dataset('atm_zgrd',    data=atm_zgrd)
            h5_output.create_dataset('lay',         data=lay)
            h5_output.create_dataset('atm_p',       data=atm_p)
            h5_output.create_dataset('atm_pi',      data=atm_pi)
            h5_output.create_dataset('atm_temp',    data=atm_temp)
            h5_output.create_dataset('atm_dz',      data=atm_dz)
            h5_output.create_dataset('lamx',        data=lamx)
            h5_output.create_dataset('tx',          data=tx)
            h5_output.create_dataset('absgl',       data=absgl)
            h5_output.create_dataset('absgn',       data=absgn)
            h5_output.create_dataset('absgx',       data=absgx)
            h5_output.create_dataset('absgy',       data=absgy)
            h5_output.create_dataset('solx',        data=solx)
            h5_output.create_dataset('ils0',        data=ils0)
            h5_output.create_dataset('unit_z',       data=unit_z)
            h5_output.create_dataset('unit_l',        data=unit_l)
            h5_output.create_dataset('unit_p',        data=unit_p)
            h5_output.create_dataset('unit_T',       data=unit_T)
            h5_output.create_dataset('unit_abs',        data=unit_abs)
            h5_output.create_dataset('wl_oco',        data=wl_oco)
            h5_output.create_dataset('trns_oco',       data=trns_oco)


    
        if 1 & plot :
            None
            """
            window,5
            lp=1
            plot,extcheck[lp,*],lay,psym=-4,xtit='absorption coef [1/km]',ytit='Layer Altitude [km]',chars=2,tit=string(dzf[0])
            window,6
            #plot,wloco,trnsc
            plot,wloco,tauc
            """
                
        return None
        #todo: check if vertical resolution changes tau and/or ext


if __name__ == '__main__':
    cfg = '/Users/yuch8913/programming/oco/simulation/cfg/20181018_central_asia_2_470cloud_test.csv'
    oco_abs(cfg, iband=1, nx=10, reextract=True, plot=True)