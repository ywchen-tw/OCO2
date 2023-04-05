from genericpath import isfile
import os
import sys
import glob
import pickle
import h5py
import numpy as np
import pandas as pd
import datetime
import time
from scipy.io import readsav
from scipy import interpolate
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy import stats as st
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams, ticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

import er3t
from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_16g, abs_oco_idl, abs_oco_h5
from er3t.pre.cld import cld_sat, cld_les
from er3t.pre.sfc import sfc_sat
from er3t.pre.pha import pha_mie_wc as pha_mie # newly added for phase function
from er3t.util.modis import modis_l1b, modis_l2, modis_03, modis_04, modis_09a1, modis_43a3, download_modis_https, get_sinusoidal_grid_tag, get_filename_tag, download_modis_rgb
from er3t.util.oco2 import oco2_std, download_oco2_https
from er3t.util import cal_r_twostream, grid_by_extent, grid_by_lonlat, cal_ext

from er3t.rtm.mca import mca_atm_1d, mca_atm_3d, mca_sfc_2d
from er3t.rtm.mca import mcarats_ng
from er3t.rtm.mca import mca_out_ng
from er3t.rtm.mca import mca_sca # newly added for phase function

from oco_subroutine.oco_satellite import satellite_download

import timeit
import argparse
import matplotlib.image as mpl_img

class sat_tmp:

    def __init__(self, data):

        self.data = data

def cal_mca_rad_650(sat, zpt_file, wavelength, photons=1e7, fdir='tmp-data', solver='3D', case_name_tag='default', overwrite=False):

    """
    Simulate MODIS radiance
    """

    # atm object
    # =================================================================================
    with h5py.File(zpt_file, 'r') as oco_zpt:
        levels = oco_zpt['h_edge'][...]
    fname_atm = '%s/atm.pk' % fdir
    atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    # =================================================================================


    # abs object
    # =================================================================================
    fname_abs = '%s/abs.pk' % fdir
    abs0      = abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)
    # =================================================================================


    # sfc object
    # =================================================================================
    data = {}
    with h5py.File(f'{sat.fdir_out}/pre-data.h5', 'r') as f:
        data['alb_2d'] = dict(data=f['mod/sfc/alb_43_650'][...], name='Surface albedo', units='N/A')
        data['lon_2d'] = dict(data=f['mod/sfc/lon'][...], name='Longitude', units='degrees')
        data['lat_2d'] = dict(data=f['mod/sfc/lat'][...], name='Latitude' , units='degrees')

    fname_sfc = '%s/sfc.pk' % fdir
    mod09 = sat_tmp(data)
    sfc0      = sfc_sat(sat_obj=mod09, fname=fname_sfc, extent=sat.extent, verbose=True, overwrite=overwrite)
    sfc_2d    = mca_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/mca_sfc_2d.bin' % fdir, overwrite=overwrite)
    # =================================================================================


    # cld object
    # =================================================================================
    data = {}
    with h5py.File(f'{sat.fdir_out}/pre-data.h5', 'r') as f:
        data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=f['lon'][...])
        data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=f['lat'][...])
        data['rad_2d'] = dict(name='Gridded radiance'                , units='km'         , data=f[f'mod/rad/rad_650'][...])
        if solver.lower() == 'ipa':
            data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=f['mod/cld/cot_ipa0'][...])
            data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=f['mod/cld/cer_ipa0'][...])
            data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=f['mod/cld/cth_ipa0'][...])
        elif solver.lower() == '3d':
            data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=f['mod/cld/cot_ipa'][...])
            data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=f['mod/cld/cer_ipa'][...])
            data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=f['mod/cld/cth_ipa'][...])

    modl1b    =  sat_tmp(data)
    fname_cld = '%s/cld.pk' % fdir

    cth0 = modl1b.data['cth_2d']['data']
    cth0[cth0>10.0] = 10.0
    cgt0 = np.zeros_like(cth0)
    cgt0[cth0>0.0] = 1.0                  # all clouds have geometrical thickness of 1 km
    cgt0[cth0>4.0] = cth0[cth0>4.0]-3.0   # high clouds (cth>4km) has cloud base at 3 km
    cld0      = cld_sat(sat_obj=modl1b, fname=fname_cld, cth=cth0, cgt=cgt0, dz=0.5,#np.unique(atm0.lay['thickness']['data'])[0],
                        overwrite=overwrite)
    # =================================================================================


    # mca_sca object
    # =================================================================================
    pha0 = er3t.pre.pha.pha_mie_wc(wavelength=wavelength, overwrite=overwrite)
    sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)
    # =================================================================================


    # mca_cld object
    # =================================================================================
    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir)
    #atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)

    # homogeneous 1d mcarats "atmosphere"
    # =================================================================================
    atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)

    # add homogeneous 1d mcarats "atmosphere", aerosol layer
    f = h5py.File(f'{sat.fdir_out}/pre-data.h5', 'r')
    AOD_550_land_mean = f['mod/aod/AOD_550_land_mean'][...]
    Angstrom_Exponent_land_mean = f['mod/aod/Angstrom_Exponent_land_mean'][...]
    SSA_land_mean = f['mod/aod/SSA_660_land_mean'][...]

    aod = AOD_550_land_mean*((wavelength/550)**(Angstrom_Exponent_land_mean*-1))
    ssa = SSA_land_mean
    cth_mode = st.mode(cth0[np.logical_and(cth0>0, cth0<4)])
    print('Angstrom Exponent:', Angstrom_Exponent_land_mean)
    print('aod 550nm mean:', AOD_550_land_mean)
    print('aod 650nm mean:', aod)
    print('ssa 650nm mean:', ssa)
    f.close()
    print('cth mode:', cth_mode.mode[0])
    #hist_result = plt.hist(cth0[cth0>0].flatten(), bins=25)
    #print(hist_result)
    #plt.vlines(cth_mode, 0, 100, 'r')
    #plt.show()
    #aod    = 0.4 # aerosol optical depth
    #ssa    = 0.9 # aerosol single scattering albedo
    asy    = 0.6 # aerosol asymmetry parameter
    z_bot  = 0 # altitude of layer bottom in km
    z_top  = cth_mode.mode[0]#8.0 # altitude of layer top in km
    aer_ext = aod / (atm0.lay['thickness']['data'].sum()*1000.0)

    #atm1d0.add_mca_1d_atm(ext1d=aer_ext, omg1d=ssa, apf1d=asy, z_bottom=z_bot, z_top=z_top)
    # data can be accessed at
    #     atm1d0.nml[ig]['Atm_zgrd0']['data']
    #     atm1d0.nml[ig]['Atm_wkd0']['data']
    #     atm1d0.nml[ig]['Atm_mtprof']['data']
    #     atm1d0.nml[ig]['Atm_tmp1d']['data']
    #     atm1d0.nml[ig]['Atm_nkd']['data']
    #     atm1d0.nml[ig]['Atm_nz']['data']
    #     atm1d0.nml[ig]['Atm_ext1d']['data']
    #     atm1d0.nml[ig]['Atm_abs1d']['data']
    #     atm1d0.nml[ig]['Atm_omg1d']['data']
    #     atm1d0.nml[ig]['Atm_apf1d']['data']
    # =================================================================================

    atm_1ds = [atm1d0]
    atm_3ds = [atm3d0]
    # =================================================================================

    
    # solar zenith/azimuth angles and sensor zenith/azimuth angles
    # =================================================================================
    f = h5py.File(f'{sat.fdir_out}/pre-data.h5', 'r')
    sza = f['mod/geo/sza'][...].mean()
    saa = f['mod/geo/saa'][...].mean()
    vza = f['mod/geo/vza'][...].mean()
    vaa = f['mod/geo/vaa'][...].mean()
    print('vza', vza, '; vaa', vaa)
    f.close()
    # =================================================================================

    # run mcarats
    # =================================================================================
    mca0 = mcarats_ng(
            date=sat.date,
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            sfc_2d=sfc_2d,
            sca=sca,
            Ng=abs0.Ng,
            target='radiance',
            solar_zenith_angle   = sza,
            solar_azimuth_angle  = saa,
            sensor_zenith_angle  = vza,
            sensor_azimuth_angle = vaa,
            fdir='%s/%.4fnm/rad_%s' % (fdir, wavelength, solver.lower()),
            Nrun=3,
            weights=abs0.coef['weight']['data'],
            photons=photons,
            solver=solver,
            Ncpu=8,
            mp_mode='py',
            overwrite=overwrite
            )

    # mcarats output
    out0 = mca_out_ng(fname='%s/mca-out-rad-modis-%s_%.4fnm.h5' % (fdir, solver.lower(), wavelength), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)
    # =================================================================================

def modis_650_simulation_plot(extent_list, case_name_tag='default', fdir='tmp', solver='3D', wvl=650, ref_threshold=0.1, plot=False):

    # create data directory (for storing data) if the directory does not exist
    # ==================================================================================================
    #name_tag = __file__.replace('.py', '')
    fname_sat = f'data/{case_name_tag}/sat.pk'
    sat = satellite_download(fname=fname_sat, overwrite=False)
    mod_img = mpl_img.imread(sat.fnames['mod_rgb'][0])
    mod_img_wesn = sat.extent
    
    fdir_data = os.path.abspath('data/%s' % case_name_tag)
    # ==================================================================================================

    # read in MODIS measured radiance
    # ==================================================================================================
    f = h5py.File('data/%s/pre-data.h5' % case_name_tag, 'r')
    extent = f['extent'][...]
    lon_mod = f['lon'][...]
    lat_mod = f['lat'][...]
    rad_mod = f['mod/rad/rad_650'][...]
    cth_mod = f['mod/cld/cth_l2'][...]
    if solver.lower() == 'ipa':
        cth_mod = f['mod/cld/cth_ipa0'][...]
    elif solver.lower() == '3d':
        cth_mod = f['mod/cld/cth_ipa'][...]
    f.close()
    # ==================================================================================================


    # read in EaR3T simulations (3D)
    # ==================================================================================================
    fname = '%s/mca-out-rad-modis-%s_%.4fnm.h5' % (fdir, solver.lower(), 650)
    f = h5py.File(fname, 'r')
    rad_rtm_3d     = f['mean/rad'][...]
    rad_rtm_3d_std = f['mean/rad_std'][...]
    toa = f['mean/toa'][...]
    f.close()
    # ==================================================================================================


    # save data
    # ==================================================================================================
    f = h5py.File('data/%s/post-data.h5' % case_name_tag, 'w')
    f['wvl'] = wvl
    f['lon'] = lon_mod
    f['lat'] = lat_mod
    f['extent']         = extent
    f['rad_obs']        = rad_mod
    f['rad_sim_3d']     = rad_rtm_3d
    f['rad_sim_3d_std'] = rad_rtm_3d_std
    f.close()
    # ==================================================================================================

    if plot:

        # ==================================================================================================
        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(234)
        ax1.imshow(rad_mod.T, cmap='Greys_r', extent=extent, origin='lower', vmin=0.0, vmax=0.5)
        ax1.set_xlabel('Longititude [$^\circ$]')
        ax1.set_ylabel('Latitude [$^\circ$]')
        ax1.set_xlim((extent_list[:2]))
        ax1.set_ylim((extent_list[2:]))
        ax1.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 181.0, 0.5)))
        ax1.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 91.0, 0.5)))
        ax1.set_title('MODIS Measured Radiance')

        logic = (lon_mod>=extent_list[0]) & (lon_mod<=extent_list[1]) & (lat_mod>=extent_list[2]) & (lat_mod<=extent_list[3])

        xedges = np.arange(-0.01, 0.61, 0.005)
        yedges = np.arange(-0.01, 0.61, 0.005)
        heatmap, xedges, yedges = np.histogram2d(rad_mod[logic], rad_rtm_3d[logic], bins=(xedges, yedges))
        YY, XX = np.meshgrid((yedges[:-1]+yedges[1:])/2.0, (xedges[:-1]+xedges[1:])/2.0)

        levels = np.concatenate((np.arange(1.0, 10.0, 1.0),
                                 np.arange(10.0, 200.0, 10.0),
                                 np.arange(200.0, 1000.0, 100.0),
                                 np.arange(1000.0, 10001.0, 5000.0)))
        
        ax3 = fig.add_subplot(232)
        ax3.imshow(rad_rtm_3d.T, cmap='Greys_r', extent=extent, origin='lower', vmin=0.0, vmax=0.5)
        ax3.set_xlabel('Longititude [$^\circ$]')
        ax3.set_ylabel('Latitude [$^\circ$]')
        ax3.set_xlim((extent_list[:2]))
        ax3.set_ylim((extent_list[2:]))
        ax3.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 181.0, 0.5)))
        ax3.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 91.0, 0.5)))
        ax3.set_title(f'EaR$^3$T Simulated {solver} Radiance')
        
        ax13 = fig.add_subplot(233)
        diff = ax13.imshow((rad_rtm_3d-rad_mod).T, cmap='bwr', extent=extent, origin='lower', vmin=-0.15, vmax=0.15)
        ax13.set_xlabel('Longititude [$^\circ$]')
        ax13.set_ylabel('Latitude [$^\circ$]')
        ax13.set_xlim((extent_list[:2]))
        ax13.set_ylim((extent_list[2:]))
        ax13.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 181.0, 0.5)))
        ax13.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 91.0, 0.5)))
        cbar_13 = fig.colorbar(diff, ax=ax13)
        cbar_13.set_label('3D Simulated - MODIS Radiance')
        ax13.set_title(f'{solver} Simulated - MODIS Radiance')
        
        ax2 = fig.add_subplot(231)
        cs = ax2.contourf(XX, YY, heatmap, levels, extend='both', locator=ticker.LogLocator(), cmap='jet')
        ax2.plot([0.0, 1.0], [0.0, 1.0], lw=1.0, ls='--', color='gray', zorder=3)
        ax2.set_xlim(0.0, 0.6)
        ax2.set_ylim(0.0, 0.6)
        ax2.set_xlabel('MODIS Measured Radiance')
        ax2.set_ylabel(f'Simulated {solver} Radiance')

        ax22 = fig.add_subplot(235)
        ax22.imshow(mod_img, extent=mod_img_wesn)
        cth_mask = ~np.isnan(cth_mod)
        ax22.scatter(lon_mod, lat_mod, cth_mod, c='r')
        ax22.set_xlabel('Longititude [$^\circ$]')
        ax22.set_ylabel('Latitude [$^\circ$]')
        ax22.set_xlim((extent_list[:2]))
        ax22.set_ylim((extent_list[2:]))
        ax22.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 181.0, 0.5)))
        ax22.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 91.0, 0.5)))
        ax22.set_title(f'EaR$^3$T Cloud mask\n(ref threshold: {ref_threshold})')

        ax4 = fig.add_subplot(236) 
        cth_img = ax4.imshow(cth_mod.T, cmap='jet', extent=extent, origin='lower', vmin=0.0, vmax=10)
        fig.colorbar(cth_img, ax=ax4)
        ax4.set_xlabel('Longititude [$^\circ$]')
        ax4.set_ylabel('Latitude [$^\circ$]')
        ax4.set_xlim((extent_list[:2]))
        ax4.set_ylim((extent_list[2:]))
        ax4.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 181.0, 0.5)))
        ax4.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 91.0, 0.5)))
        ax4.set_title('EaR$^3$T CTH')

        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        if not os.path.exists('outut_img/'):
            os.makedirs('outut_img/')
        plt.savefig(f'data/modis_650_{case_name_tag}_{solver}.png', bbox_inches='tight')
        plt.show()
        # ==================================================================================================
