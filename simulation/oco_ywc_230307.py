

from genericpath import isfile
import os
import sys
import glob
import pickle
import h5py
from pyhdf.SD import SD, SDC
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
from matplotlib import rcParams
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

from oco_subroutine.oco_subroutine import *
from oco_subroutine.oco_create_atm import create_oco_atm
from oco_subroutine.oco_satellite import satellite_download
from oco_subroutine.oco_cloud import func_cot_vs_rad, para_corr, wind_corr
from oco_subroutine.oco_cfg import grab_cfg, save_h5_info, check_h5_info, save_subdomain_info
from oco_subroutine.oco_abs_snd import oco_abs
from oco_subroutine.oco_modis_time import cal_sat_delta_t
from oco_subroutine.oco_raw_collect import cdata_sat_raw
from oco_subroutine.oco_cloud import cdata_cld_ipa

import timeit
import argparse
import matplotlib.image as mpl_img




def create_sfc_alb_2d(x_ref, y_ref, data_ref, x_bkg_2d, y_bkg_2d, data_bkg_2d, scale=True, replace=True):

    def func(x, a):
        return a*x

    points = np.transpose(np.vstack((x_bkg_2d.ravel(), y_bkg_2d.ravel())))
    data_bkg = interpolate.griddata(points, data_bkg_2d.ravel(), (x_ref, y_ref), method='nearest')

    if scale:
        popt, pcov = curve_fit(func, data_bkg, data_ref)
        slope = popt[0]
    else:
        slope = 1.0

    data_2d = data_bkg_2d*slope

    dx = x_bkg_2d[1, 0] - x_bkg_2d[0, 0]
    dy = y_bkg_2d[0, 1] - y_bkg_2d[0, 0]

    if replace:
        indices_x = np.int_(np.round((x_ref-x_bkg_2d[0, 0])/dx, decimals=0))
        indices_y = np.int_(np.round((y_ref-y_bkg_2d[0, 0])/dy, decimals=0))
        data_2d[indices_x, indices_y] = data_ref

    return data_2d

def pre_cld(sat, tag, cth=None, cot_source='2stream', ref_threshold=0.1, scale_factor=1.0, solver='3D', fdir_cot='tmp-data'):

    # retrieve 1. cloud top height; 2. sensor zenith; 3. sensor azimuth for MODIS L1B (250nm) data from MODIS L2 (5km resolution)
    
    # Extract
    #   1. cloud top height (cth, 1km resolution);
    #   2. solar zenith and azimuth angles (sza and saa, 1km resolution);
    #   3. sensor zenith and azimuth angles (vza and vaa, 1km resolution);
    #   4. surface height (sfc, 1km resolution)
    # ===================================================================================
 
    # ===================================================================================


    

    # Process MODIS radiance and reflectance at 650 nm (250m resolution)
    # ===================================================================================
    
    # ===================================================================================

    # Process mcd04 AOD data
    # =========================================================================

    # =========================================================================

    # Process MODIS radiance and reflectance at 650/860 nm (250m resolution)
    # ===================================================================================


    # ===================================================================================

    #----------------------------------------------

    #----------------------------------------------


    # special note: threshold of 1.06 is hard-coded for this particular case;
    #               improvement will be made by Yu-Wen Chen and/or Katey Dong
    #logic_rgb_nan0 = (mod_r<=(np.median(mod_r)*1.06)) |\
    #                 (mod_g<=(np.median(mod_g)*1.06)) |\
    #                 (mod_b<=(np.median(mod_b)*1.06))
    #logic_rgb_nan0 = ref_2d-mod09_sfc_alb_inter < ref_threshold
    #logic_rgb_nan0 = (ref_2d*ref_2d_860)>=0.02
    #logic_rgb_nan0 = mod_v_over_s < 4
    #logic_rgb_nan0 = np.logical_or(mod_v<=0.62, mod_s>0.19)
    """
    logic_rgb_nan = np.flipud(logic_rgb_nan0).T

    x0_rgb = lon_rgb[0]
    y0_rgb = lat_rgb[0]
    dx_rgb = lon_rgb[1] - x0_rgb
    dy_rgb = lat_rgb[1] - y0_rgb

    indices_x = np.int_(np.round((lon_2d-x0_rgb)/dx_rgb, decimals=0))
    indices_y = np.int_(np.round((lat_2d-y0_rgb)/dy_rgb, decimals=0))
    indices_x = np.int_(np.round((lon_2d-x0_rgb)/dx_rgb, decimals=0))
    indices_y = np.int_(np.round((lat_2d-y0_rgb)/dy_rgb, decimals=0))

    logic_ref_nan = logic_rgb_nan[indices_x, indices_y]
    #"""

    
    # parallax correction
    # ====================================================================================================
    if cth is None:
        # Upscale CTH from 1km (L2) to 250m resolution
        # ===================================================================================
        now = time.time()
        points     = np.transpose(np.vstack((lon0[logic_cth], lat0[logic_cth])))
        #cth        = interpolate.griddata(points, cth0[logic_cth], (lon, lat), method='nearest')
        points_1km     = np.transpose(np.vstack((lon_1km[logic_cth_1km], lat_1km[logic_cth_1km])))
        cth        = interpolate.griddata(points_1km, cth_1km[logic_cth_1km], (lon, lat), method='nearest')

        cth_2d_l2 = np.zeros_like(lon_2d)
        cth_2d_l2[indices_x, indices_y] = cth
        print('cth_1km (un-filterd):', np.nanmax(cth_1km), np.nanmin(cth_1km))
        print('cth_1km (filtered):', np.nanmax(cth_1km[logic_cth_1km]), np.nanmin(cth_1km[logic_cth_1km]))
        print('cth (ungridded):', np.nanmax(cth), np.nanmin(cth))
        print(f'CTH interpolation time: {(time.time()-now)/60} min')
        # Upscale cloud effective radius from 1km (L2) to 250m resolution
        # =============================================================
        now = time.time()
        points_cer = np.transpose(np.vstack((lon_cer, lat_cer)))
        #cer        = interpolate.griddata(points_cer, cer0, (lon, lat), method='nearest')
        _, _, cer_2d_l2 = grid_by_lonlat(lon_cer, lat_cer, cer0, lon_1d=lon_2d[:, 0], lat_1d=lat_2d[0, :], method='linear')
        cer_2d_l2[cer_2d_l2<=1] = 1         # make sure the minimum cer is 1
        print(f'CER griding time: {(time.time()-now)/60} min')

        # =============================================================
        
        # Upscale cloud optical thickness from 1km (L2) to 250m resolution
        # =============================================================
        now = time.time()
        cot        = interpolate.griddata(points_cer, cot0, (lon, lat), method='nearest')
        _, _, cot_2d_l2 = grid_by_lonlat(lon_cer, lat_cer, cot0, lon_1d=lon_2d[:, 0], lat_1d=lat_2d[0, :], method='nearest')

        cot_2d = np.zeros_like(lon_2d)
        cot_2d[indices_x, indices_y] = cot
        print(f'COT griding time: {(time.time()-now)/60} min')
        # =============================================================

    # Parallax correction (for the cloudy pixels detected previously)
    # ====================================================================================================
    now = time.time()
    points     = np.transpose(np.vstack((lon1[logic_sfh], lat1[logic_sfh])))
    sfh        = interpolate.griddata(points, sfh1[logic_sfh], (lon, lat), method='linear')

    points     = np.transpose(np.vstack((lon1, lat1)))
    vza        = interpolate.griddata(points, vza1, (lon, lat), method='linear')
    vaa        = interpolate.griddata(points, vaa1, (lon, lat), method='linear')
    #vza[...] = np.nanmean(vza)
    #vaa[...] = np.nanmean(vaa)
    if solver == '3D':
        lon_corr_p, lat_corr_p  = para_corr(lon, lat, vza, vaa, cth*1000.0, sfh*1000.0)
    elif solver == 'IPA':
        lon_corr_p, lat_corr_p  = para_corr(lon, lat, vza, vaa, sfh, sfh)
    print(f'Parallax correction time: {(time.time()-now)/60} min')
    # ====================================================================================================


    # wind correction
    # ====================================================================================================
    now = time.time()
    f = h5py.File(sat.fnames['oco_l1b'][0], 'r')
    lon_oco_l1b = f['SoundingGeometry/sounding_longitude'][...]
    lat_oco_l1b = f['SoundingGeometry/sounding_latitude'][...]
    logic = (lon_oco_l1b>=sat.extent[0]) & (lon_oco_l1b<=sat.extent[1]) & (lat_oco_l1b>=sat.extent[2]) & (lat_oco_l1b<=sat.extent[3])
    utc_oco_byte = f['SoundingGeometry/sounding_time_string'][...][logic]
    f.close()
    utc_oco = np.zeros(utc_oco_byte.size, dtype=np.float64)
    for i, utc_oco_byte0 in enumerate(utc_oco_byte):
        utc_oco_str0 = utc_oco_byte0.decode('utf-8').split('.')[0]
        utc_oco[i] = (datetime.datetime.strptime(utc_oco_str0, '%Y-%m-%dT%H:%M:%S')-datetime.datetime(1993, 1, 1)).total_seconds()

    f = SD(sat.fnames['mod_03'][0], SDC.READ)
    lon_mod = f.select('Longitude')[:][::10, :]
    lat_mod = f.select('Latitude')[:][::10, :]
    utc_mod = f.select('SD start time')[:]
    f.end()
    logic = (lon_mod>=sat.extent[0]) & (lon_mod<=sat.extent[1]) & (lat_mod>=sat.extent[2]) & (lat_mod<=sat.extent[3])
    logic = (np.sum(logic, axis=1)>0)
    utc_mod = utc_mod[logic]

    f = h5py.File(sat.fnames['oco_met'][0], 'r')
    lon_oco_met = f['SoundingGeometry/sounding_longitude'][...]
    lat_oco_met = f['SoundingGeometry/sounding_latitude'][...]
    logic = (lon_oco_met>=sat.extent[0]) & (lon_oco_met<=sat.extent[1]) & (lat_oco_met>=sat.extent[2]) & (lat_oco_met<=sat.extent[3])
    u_oco = f['Meteorology/wind_u_profile_met'][...][logic]
    v_oco = f['Meteorology/wind_v_profile_met'][...][logic]
    p_oco = f['Meteorology/vector_pressure_levels_met'][...][logic]
    f.close()

    lon_corr, lat_corr  = wind_corr(lon_corr_p, lat_corr_p, np.median(u_oco), np.median(v_oco), utc_oco.mean()-utc_mod.mean())
    print(f'Wind correction time: {(time.time()-now)/60} min')
    # ====================================================================================================

    # Cloud optical property
    #  1) cloud optical thickness: MODIS 650 reflectance -> two-stream approximation -> cloud optical thickness
    #  2) cloud effective radius: from MODIS L2 cloud product (upscaled to 250m resolution from raw 1km resolution)
    #  3) cloud top height: from MODIS L2 cloud product
    #
    #   special note: for this particular case, saturation was found on MODIS 860 nm reflectance
    # ===================================================================================
    # two-stream

    # get surface albedo by dividing the modis 43 reflectance with cos(solar zenith angle)
    sfc_alb_inter /= np.cos(sza_inter/180*np.pi)
    ref_2d /= np.cos(sza_inter/180*np.pi)

    a0         = np.nanmedian(sfc_alb_inter)
    mu0        = np.cos(np.deg2rad(sza1.mean()))

    xx_2stream = np.linspace(0.0, 400.0, 10000)
    yy_2stream = cal_r_twostream(xx_2stream, a=a0, mu=mu0)  #g0 still constant for now


    

    # lon/lat shift due to parallax and wind correction
    lon_1d = lon_2d[:, 0]
    indices_x_new = np.int_(np.round((lon_corr-lon_1d[0])/(((lon_1d[1:]-lon_1d[:-1])).mean()), decimals=0))
    lat_1d = lat_2d[0, :]
    indices_y_new = np.int_(np.round((lat_corr-lat_1d[0])/(((lat_1d[1:]-lat_1d[:-1])).mean()), decimals=0))



    Nx, Ny = ref_2d.shape
    cot_2d_l1b = np.zeros_like(ref_2d)
    cot_2d_l1b_b4_correction = np.zeros_like(ref_2d)
    cot_2d_mcar_b4_correction = np.zeros_like(ref_2d)
    cer_2d_l1b = np.ones_like(ref_2d)
    
    cer_2d_l1b[...] = 12 # minimum cer is set to 12
    cth_2d_l1b = np.zeros_like(ref_2d)
    now = time.time()
    for i in range(indices_x.size):
        cot_2d_l1b_b4_correction[indices_x[i], indices_y[i]] = xx_2stream[np.argmin(np.abs(yy_2stream-ref_2d[indices_x[i], indices_y[i]]))]
        if 0<=indices_x_new[i]<Nx and 0<=indices_y_new[i]<Ny:
            cer_2d_l1b[indices_x_new[i], indices_y_new[i]] = cer_2d_l2[indices_x[i], indices_y[i]]
            cth_2d_l1b[indices_x_new[i], indices_y_new[i]] = cth_2d_l2[indices_x[i], indices_y[i]]
            if cot_source=='2stream':
                # COT from two-stream
                cot_2d_l1b[indices_x_new[i], indices_y_new[i]] = xx_2stream[np.argmin(np.abs(yy_2stream-ref_2d[indices_x[i], indices_y[i]]))]
            #elif cot_source=='mcarats':
            #    # COT from two-stream
            #    cot_2d_l1b[indices_x_new[i], indices_y_new[i]] = f_mca.interp_from_rad(rad_2d[indices_x[i], indices_y[i]])
            elif cot_source=='MODIS':
                # COT from closest COT from MODIS L2 cloud product
                cot_2d_l1b[indices_x_new[i], indices_y_new[i]] = cot_2d_l2[indices_x[i], indices_y[i]]
            else:
                sys.exit('Error  [pre_cld_modis]: cot_select must be either "2stream" or "MODIS" ')
    print(f'New Cloud position correction time: {(time.time()-now)/60} min')
    modl1b.data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=lon_2d)
    modl1b.data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=lat_2d)
    modl1b.data['rad_2d'] = dict(name='Gridded radiance'                , units='W/m^2/nm/sr', data=rad_2d)
    modl1b.data['ref_2d'] = dict(name='Gridded reflectance (650nm)'     , units='N/A'        , data=ref_2d)
    modl1b.data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=cot_2d_l1b)#scale_factor)
    modl1b.data['cot_2d_1km'] = dict(name='Gridded cloud optical thickness' , units='N/A'    , data=cot_2d_l2)
    modl1b.data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=cth_2d_l1b)
    modl1b.data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=cer_2d_l1b)
    modl1b.data['AOD_550_land_mean'] = dict(name='Domain averaged 550 nm AOD (Land)' , units='N/A'        , data=AOD_550_land_mean)
    modl1b.data['Angstrom_Exponent_land_mean'] = dict(name='Domain averaged 550 nm AOD Angstrom Exponent (Land)' , units='N/A'        , data=Angstrom_Exponent_land_mean)
    modl1b.data['SSA_land_mean'] = dict(name='Domain averaged 550 nm AOD (Land)' , units='N/A'        , data=SSA_land_mean)    
    #modl1b.data['wvl']    = dict(name='Wavelength'                      , units='nm'         , data=wvl)
    modl1b.data['sza']    = dict(name='Solar Zenith Angle'              , units='degree'     , data=np.nanmean(sza1))
    modl1b.data['saa']    = dict(name='Solar Azimuth Angle'             , units='degree'     , data=np.nanmean(saa1))
    modl1b.data['vza']    = dict(name='Sensor Zenith Angle'             , units='degree'     , data=np.nanmean(vza1))
    modl1b.data['vaa']    = dict(name='Sensor Azimuth Angle'            , units='degree'     , data=np.nanmean(vaa1))

    run_cot = not os.path.isfile('%s/mca-out-rad-3d_cot-%.2f.h5' % (fdir_cot, 400))

    f_mca =  func_cot_vs_rad(sat, modl1b, fdir_cot, 650, sfc_albedo=np.nanmean(mcd_bsa_2d_650), 
                             cth=cth_1km, ctt=ctt_1km, run=run_cot)  # use 650 nm to calculate cot
    
    for i in range(indices_x.size):
        cot_2d_mcar_b4_correction[indices_x[i], indices_y[i]] = f_mca.interp_from_rad(rad_2d[indices_x[i], indices_y[i]])
        if 0<=indices_x_new[i]<Nx and 0<=indices_y_new[i]<Ny:
            # COT from tmcarats
            cot_2d_l1b[indices_x_new[i], indices_y_new[i]] = f_mca.interp_from_rad(rad_2d[indices_x[i], indices_y[i]])

    """plt.scatter(cot_2d_l1b_b4_correction.flatten(), rad_2d.flatten(), label='2-stream', alpha=0.35)
    plt.scatter(cot_2d_mcar_b4_correction.flatten(), rad_2d.flatten(), label='MCARaTS', alpha=0.35)
    plt.plot(xx_2stream, f_mca.interp_from_cot(xx_2stream), 'r', label='MCARaTS relationship')
    plt.xlabel('COT')
    plt.ylabel('Radiance')
    plt.legend()
    plt.show()
    sys.exit()"""
    return modl1b


def preprocessing(tag, sat, cth=None, scale_factor=1.0, fdir='tmp-data', fdir_cot='tmp-data', zpt_file='zpt.h5',
                  sfc_scale=True, sfc_replace=True, solver='3D', ref_threshold=0.1, overwrite=True):
    if not os.path.isfile(f'{sat.fdir_out}/pre-data.h5') :
        cdata_sat_raw(tag, sat0=sat, overwrite=True, plot=True)

    cdata_cld_ipa(tag, sat, fdir, fdir_cot, zpt_file, ref_threshold=ref_threshold, photons=1e6, plot=True)

    return None


class sat_tmp:

    def __init__(self, data):

        self.data = data

def cal_mca_rad_oco2(date, tag, sat, wavelength, fname_idl=None, cth=None, photons=1e6, scale_factor=1.0, 
                     sfc_scale=True, sfc_replace=True, fdir='tmp-data', solver='3D', ref_threshold=0.1, overwrite=True):

    """
    Calculate OCO2 radiance using cloud (MODIS level 1b) and surface properties (MOD09A1) from MODIS
    """

    # atm object
    # =================================================================================
    levels    = np.array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. , 5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. , 11. , 12. , 13. , 14. , 20. , 25. , 30. , 35. , 40. ])
    levels = np.array([1.15731616, 1.54158454, 1.92585293, 2.3101213, 2.69438969, 3.07865808, 3.46292646, 3.84719485, 4.23146323, 4.61573162, 5., 5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. , 11. , 12. , 13. , 14. , 20. , 25. , 30. , 35. , 40. ])
    fname_atm = '%s/atm.pk' % fdir
    atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    # =================================================================================

    # abs object, in the future, we will implement OCO2 MET file for this
    # =================================================================================
    fname_abs = '%s/abs.pk' % fdir
    #abs0      = abs_oco_idl(wavelength=wavelength, fname=fname_abs, fname_idl=fname_idl, atm_obj=atm0, overwrite=overwrite)
    #print('idl shape:', abs0.abs.coef['abso_coef']['data'].shape)
    abs0      = abs_oco_h5(wavelength=wavelength, fname=fname_abs, fname_h5=fname_idl, atm_obj=atm0, overwrite=overwrite)
    # =================================================================================

    # mca_sfc object
    # =================================================================================
    #"""
    data = {}
    f = h5py.File(f'{sat.fdir_out}/pre-data.h5', 'r')
    data['alb_2d'] = dict(data=f[f'mod/sfc/alb_43_{tag}'][...], name='Surface albedo', units='N/A')
    data['lon_2d'] = dict(data=f['mod/sfc/lon'][...], name='Longitude', units='degrees')
    data['lat_2d'] = dict(data=f['mod/sfc/lat'][...], name='Latitude' , units='degrees')
    f.close()
    #"""
    print('Average sfc albedo: ', np.nanmean(data['alb_2d']['data'][...]))
    #data['alb_2d']['data'][...] = np.nanmean(data['alb_2d']['data'][...]-0.1)
    data['alb_2d']['data'][...] = 0.05

    #mod09     = pre_sfc(sat, tag, scale=sfc_scale, replace=sfc_replace)
    mod09     = sat_tmp(data)
    fname_sfc = '%s/sfc.pk' % fdir
    sfc0      = sfc_sat(sat_obj=mod09, fname=fname_sfc, extent=sat.extent, verbose=True, overwrite=overwrite)
    sfc_2d    = mca_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/mca_sfc_2d.bin' % fdir, overwrite=overwrite)
    # =================================================================================


    # mca_sca object (newly added for phase function)
    # =================================================================================
    pha0 = er3t.pre.pha.pha_mie_wc(wavelength=wavelength, overwrite=overwrite)
    sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)
    # =================================================================================


    # cld object
    # =================================================================================
    data = {}
    f = h5py.File(f'{sat.fdir_out}/pre-data.h5', 'r')
    data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=f['lon'][...])
    data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=f['lat'][...])
    data['rad_2d'] = dict(name='Gridded radiance'                , units='km'         , data=f[f'mod/rad/rad_%4.4d' % 650][...])
    if solver.lower() == 'ipa':
        data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=f['mod/cld/cot_ipa0'][...])
        data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=f['mod/cld/cer_ipa0'][...])
        data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=f['mod/cld/cth_ipa0'][...])
    elif solver.lower() == '3d':
        data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=f['mod/cld/cot_ipa'][...])
        data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=f['mod/cld/cer_ipa'][...])
        data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=f['mod/cld/cth_ipa'][...])
    f.close()

    # =================================================================================

    # aod object
    # =================================================================================
    f = h5py.File(f'{sat.fdir_out}/pre-data.h5', 'r')
    AOD_550_land_mean = f['mod/aod/AOD_550_land_mean'][...]
    Angstrom_Exponent_land_mean = f['mod/aod/Angstrom_Exponent_land_mean'][...]
    SSA_land_mean = f['mod/aod/SSA_660_land_mean'][...]
    f.close()
    # =================================================================================

    #modl1b    = pre_cld(sat, tag, cth=cth, scale_factor=scale_factor, solver=solver, ref_threshold=ref_threshold)
    modl1b    =  sat_tmp(data)
    fname_cld = '%s/cld.pk' % fdir

    cth0 = modl1b.data['cth_2d']['data']
    cgt0 = np.zeros_like(cth0)
    cgt0[cth0>0.0] = 1.0                  # all clouds have geometrical thickness of 1 km
    cgt0[cth0>4.0] = cth0[cth0>4.0]-3.0   # high clouds (cth>4km) has cloud base at 3 km
    cld0      = cld_sat(sat_obj=modl1b, fname=fname_cld, cth=cth0, cgt=cgt0, dz=np.unique(atm0.lay['thickness']['data'])[0], overwrite=overwrite)
    # =================================================================================


    # mca_cld object
    # =================================================================================
    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir) # newly modified for phase function
    atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)

    aod = AOD_550_land_mean*((wavelength/550)**(Angstrom_Exponent_land_mean*-1)) 
    ssa = SSA_land_mean
    cth_mode = st.mode(cth0[np.logical_and(cth0>0, cth0<4)])
    print('aod 650nm mean:', aod)
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


    atm_1ds = [atm1d0]
    atm_3ds = [atm3d0]
    # =================================================================================


    f = h5py.File(sat.fnames['oco_l1b'][0], 'r')
    lon_oco_l1b = f['SoundingGeometry/sounding_longitude'][...]
    lat_oco_l1b = f['SoundingGeometry/sounding_latitude'][...]
    logic = (lon_oco_l1b>=sat.extent[0]) & (lon_oco_l1b<=sat.extent[1]) & (lat_oco_l1b>=sat.extent[2]) & (lat_oco_l1b<=sat.extent[3])
    sza = f['SoundingGeometry/sounding_solar_zenith'][...][logic].mean()
    saa = f['SoundingGeometry/sounding_solar_azimuth'][...][logic].mean()
    vza = f['SoundingGeometry/sounding_zenith'][...][logic].mean()
    vaa = f['SoundingGeometry/sounding_azimuth'][...][logic].mean()
    f.close()
    #sza = 60

    # run mcarats
    mca0 = mcarats_ng(
            date=date,
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            sfc_2d=sfc_2d,
            sca=sca, # newly added for phase function
            Ng=abs0.Ng,
            target='radiance',
            solar_zenith_angle   = sza,
            solar_azimuth_angle  = saa,
            sensor_zenith_angle  = vza,
            sensor_azimuth_angle = vaa,
            fdir='%s/%.4fnm/oco2/rad_%s' % (fdir, wavelength, solver.lower()),
            Nrun=3,
            weights=abs0.coef['weight']['data'],
            photons=photons,
            solver=solver,
            Ncpu=8,
            mp_mode='py',
            overwrite=overwrite
            )

    # mcarats output
    out0 = mca_out_ng(fname='%s/mca-out-rad-oco2-%s_%.4fnm.h5' % (fdir, solver.lower(), wavelength), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    oco_std0 = oco2_std(fnames=sat.fnames['oco_std'], extent=sat.extent)

    # plot
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(12, 5.5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    cs = ax1.imshow(modl1b.data['rad_2d']['data'].T, cmap='Greys_r', origin='lower', vmin=0.0, vmax=0.3, extent=sat.extent, zorder=0)
    ax1.scatter(oco_std0.data['lon']['data'], oco_std0.data['lat']['data'], s=20, c=oco_std0.data['xco2']['data'], cmap='jet', alpha=0.4, zorder=1)
    ax1.set_xlabel('Longitude [$^\circ$]')
    ax1.set_ylabel('Latitude [$^\circ$]')
    ax1.set_xlim(sat.extent[:2])
    ax1.set_ylim(sat.extent[2:])
    ax1.set_title('MODIS Chanel 1')

    cs = ax2.imshow(out0.data['rad']['data'].T, cmap='Greys_r', origin='lower', extent=sat.extent, zorder=0)
    ax2.scatter(oco_std0.data['lon']['data'], oco_std0.data['lat']['data'], s=20, c=oco_std0.data['xco2']['data'], cmap='jet', alpha=0.4, zorder=1)
    ax2.set_xlabel('Longitude [$^\circ$]')
    ax2.set_ylabel('Latitude [$^\circ$]')
    ax2.set_xlim(sat.extent[:2])
    ax2.set_ylim(sat.extent[2:])
    ax2.set_title('MCARaTS %s' % solver)
    plt.subplots_adjust(hspace=0.5)
    if cth is not None:
        plt.savefig('%s/mca-out-rad-modis-%s_cth-%.2fkm_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), cth, scale_factor, wavelength), bbox_inches='tight')
    else:
        plt.savefig('%s/mca-out-rad-modis-%s_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), scale_factor, wavelength), bbox_inches='tight')
    plt.close(fig)
    # ------------------------------------------------------------------------------------------------------


    if solver.lower() == 'ipa':

        sfc0      = sfc_sat(sat_obj=mod09, fname=fname_sfc, extent=sat.extent, verbose=True, overwrite=False)
        sfc_2d    = mca_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/mca_sfc_2d.bin' % fdir, overwrite=overwrite)

        cld0    = cld_sat(fname=fname_cld, overwrite=False)
        cld0.lay['extinction']['data'][...] = 0.0
        atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % fdir)
        atm_3ds = [atm3d0]

        # run mcarats
        mca0 = mcarats_ng(
                date=date,
                atm_1ds=atm_1ds,
                atm_3ds=atm_3ds,
                sfc_2d=sfc_2d,
                Ng=abs0.Ng,
                target='radiance',
                solar_zenith_angle   = sza,
                solar_azimuth_angle  = saa,
                sensor_zenith_angle  = vza,
                sensor_azimuth_angle = vaa,
                fdir='%s/%.4fnm/oco2/rad_%s0' % (fdir, wavelength, solver.lower()),
                Nrun=3,
                weights=abs0.coef['weight']['data'],
                photons=photons,
                solver=solver,
                Ncpu=8,
                mp_mode='py',
                overwrite=overwrite
                )

        # mcarats output
        out0 = mca_out_ng(fname='%s/mca-out-rad-oco2-%s0_%.4fnm.h5' % (fdir, solver.lower(), wavelength), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

        # plot
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        fig = plt.figure(figsize=(12, 5.5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        cs = ax1.imshow(modl1b.data['rad_2d']['data'].T, cmap='Greys_r', origin='lower', vmin=0.0, vmax=0.3, extent=sat.extent, zorder=0)
        ax1.scatter(oco_std0.data['lon']['data'], oco_std0.data['lat']['data'], s=20, c=oco_std0.data['xco2']['data'], cmap='jet', alpha=0.4, zorder=1)
        ax1.set_xlabel('Longitude [$^\circ$]')
        ax1.set_ylabel('Latitude [$^\circ$]')
        ax1.set_xlim(sat.extent[:2])
        ax1.set_ylim(sat.extent[2:])
        ax1.set_title('MODIS Chanel 1')

        cs = ax2.imshow(out0.data['rad']['data'].T, cmap='Greys_r', origin='lower', extent=sat.extent, zorder=0)
        ax2.scatter(oco_std0.data['lon']['data'], oco_std0.data['lat']['data'], s=20, c=oco_std0.data['xco2']['data'], cmap='jet', alpha=0.4, zorder=1)
        ax2.set_xlabel('Longitude [$^\circ$]')
        ax2.set_ylabel('Latitude [$^\circ$]')
        ax2.set_xlim(sat.extent[:2])
        ax2.set_ylim(sat.extent[2:])
        ax2.set_title('MCARaTS %s' % solver)
        plt.subplots_adjust(hspace=0.5)
        if cth is not None:
            plt.savefig('%s/mca-out-rad-modis-%s0_cth-%.2fkm_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), cth, scale_factor, wavelength), bbox_inches='tight')
        else:
            plt.savefig('%s/mca-out-rad-modis-%s0_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), scale_factor, wavelength), bbox_inches='tight')
        plt.close(fig)
        # ------------------------------------------------------------------------------------------------------



def convert_photon_unit(data_photon, wavelength, scale_factor=2.0):

    c = 299792458.0
    h = 6.62607015e-34
    wavelength = wavelength * 1e-9
    data = data_photon/1000.0*c*h/wavelength*scale_factor

    return data

class oco2_rad_nadir:

    def __init__(self, sat, tag):

        self.fname_l1b = sat.fnames['oco_l1b'][0]
        self.fname_std = sat.fnames['oco_std'][0]

        self.extent = sat.extent

        # =================================================================================
        self.cal_wvl()
        # after this, the following three functions will be created
        # Input: index, range from 0 to 7, e.g., 0, 1, 2, ..., 7
        # self.get_wvl_o2_a(index)
        # self.get_wvl_co2_weak(index)
        # self.get_wvl_co2_strong(index)
        # =================================================================================

        # =================================================================================
        self.get_index(self.extent)
        # after this, the following attributes will be created
        # self.index_s: starting index
        # self.index_e: ending index
        # =================================================================================

        # =================================================================================
        self.overlap(index_s=self.index_s, index_e=self.index_e)
        # after this, the following attributes will be created
        # self.logic_l1b
        # self.lon_l1b
        # self.lat_l1b
        # =================================================================================

        # =================================================================================
        self.get_data(index_s=self.index_s, index_e=self.index_e)
        # after this, the following attributes will be created
        # self.rad_o2_a
        # self.rad_co2_weak
        # self.rad_co2_strong
        # =================================================================================

    def cal_wvl(self, Nchan=1016):

        """
        Oxygen A band: centered at 765 nm
        Weak CO2 band: centered at 1610 nm
        Strong CO2 band: centered at 2060 nm
        """

        f = h5py.File(self.fname_l1b, 'r')
        wvl_coef = f['InstrumentHeader/dispersion_coef_samp'][...]
        f.close()

        Nspec, Nfoot, Ncoef = wvl_coef.shape

        wvl_o2_a       = np.zeros((Nfoot, Nchan), dtype=np.float64)
        wvl_co2_weak   = np.zeros((Nfoot, Nchan), dtype=np.float64)
        wvl_co2_strong = np.zeros((Nfoot, Nchan), dtype=np.float64)

        chan = np.arange(1, Nchan+1)
        for i in range(Nfoot):
            for j in range(Ncoef):
                wvl_o2_a[i, :]       += wvl_coef[0, i, j]*chan**j
                wvl_co2_weak[i, :]   += wvl_coef[1, i, j]*chan**j
                wvl_co2_strong[i, :] += wvl_coef[2, i, j]*chan**j

        wvl_o2_a       *= 1000.0
        wvl_co2_weak   *= 1000.0
        wvl_co2_strong *= 1000.0

        self.get_wvl_o2_a       = lambda index: wvl_o2_a[index, :]
        self.get_wvl_co2_weak   = lambda index: wvl_co2_weak[index, :]
        self.get_wvl_co2_strong = lambda index: wvl_co2_strong[index, :]

    def get_index(self, extent):

        if extent is None:
            self.index_s = 0
            self.index_e = None
        else:
            f = h5py.File(self.fname_l1b, 'r')
            lon_l1b     = f['SoundingGeometry/sounding_longitude'][...]
            lat_l1b     = f['SoundingGeometry/sounding_latitude'][...]

            logic = (lon_l1b>=extent[0]) & (lon_l1b<=extent[1]) & (lat_l1b>=extent[2]) & (lat_l1b<=extent[3])
            indices = np.where(np.sum(logic, axis=1)>0)[0]
            self.index_s = indices[0]
            self.index_e = indices[-1]

    def overlap(self, index_s=0, index_e=None, lat0=0.0, lon0=0.0):

        f       = h5py.File(self.fname_l1b, 'r')
        if index_e is None:
            lon_l1b = f['SoundingGeometry/sounding_longitude'][...][index_s:, ...]
            lat_l1b = f['SoundingGeometry/sounding_latitude'][...][index_s:, ...]
            lon_l1b_o2a = f['FootprintGeometry/footprint_longitude'][...][index_s:, ..., 0]
            lat_l1b_o2a = f['FootprintGeometry/footprint_latitude'][...][index_s:, ..., 0]
            lon_l1b_wco2 = f['FootprintGeometry/footprint_longitude'][...][index_s:, ..., 1]
            lat_l1b_wco2 = f['FootprintGeometry/footprint_latitude'][...][index_s:, ..., 1]
            lon_l1b_sco2 = f['FootprintGeometry/footprint_longitude'][...][index_s:, ..., 2]
            lat_l1b_sco2 = f['FootprintGeometry/footprint_latitude'][...][index_s:, ..., 2]
            snd_id_l1b  = f['SoundingGeometry/sounding_id'][...][index_s:, ...]
        else:
            lon_l1b     = f['SoundingGeometry/sounding_longitude'][...][index_s:index_e, ...]
            lat_l1b     = f['SoundingGeometry/sounding_latitude'][...][index_s:index_e, ...]
            lon_l1b_o2a = f['FootprintGeometry/footprint_longitude'][...][index_s:index_e, ..., 0]
            lat_l1b_o2a = f['FootprintGeometry/footprint_latitude'][...][index_s:index_e, ..., 0]
            lon_l1b_wco2 = f['FootprintGeometry/footprint_longitude'][...][index_s:index_e, ..., 1]
            lat_l1b_wco2 = f['FootprintGeometry/footprint_latitude'][...][index_s:index_e, ..., 1]
            lon_l1b_sco2 = f['FootprintGeometry/footprint_longitude'][...][index_s:index_e, ..., 2]
            lat_l1b_sco2 = f['FootprintGeometry/footprint_latitude'][...][index_s:index_e, ..., 2]
            snd_id_l1b  = f['SoundingGeometry/sounding_id'][...][index_s:index_e, ...]
        f.close()

        shape    = lon_l1b.shape
        lon_l1b  = lon_l1b
        lat_l1b  = lat_l1b
        lon_l1b_o2a  = lon_l1b_o2a
        lat_l1b_o2a  = lat_l1b_o2a
        lon_l1b_wco2  = lon_l1b_wco2
        lat_l1b_wco2  = lat_l1b_wco2
        lon_l1b_sco2  = lon_l1b_sco2
        lat_l1b_sco2  = lat_l1b_sco2

        f       = h5py.File(self.fname_std, 'r')
        lon_std = f['RetrievalGeometry/retrieval_longitude'][...]
        lat_std = f['RetrievalGeometry/retrieval_latitude'][...]
        xco2_std= f['RetrievalResults/xco2'][...]
        snd_id_std = f['RetrievalHeader/sounding_id'][...]
        sfc_pres_std = f['RetrievalResults/surface_pressure_fph'][...]
        f.close()

        self.logic_l1b = np.in1d(snd_id_l1b, snd_id_std).reshape(shape)

        self.lon_l1b   = lon_l1b
        self.lat_l1b   = lat_l1b
        #new
        self.lon_l1b_o2a  = lon_l1b_o2a
        self.lat_l1b_o2a  = lat_l1b_o2a
        self.lon_l1b_wco2  = lon_l1b_wco2
        self.lat_l1b_wco2  = lat_l1b_wco2
        self.lon_l1b_sco2  = lon_l1b_sco2
        self.lat_l1b_sco2  = lat_l1b_sco2
        self.snd_id    = snd_id_l1b

        xco2      = np.zeros_like(self.lon_l1b); xco2[...] = np.nan
        sfc_pres  = np.zeros_like(self.lon_l1b); sfc_pres[...] = np.nan

        for i in range(xco2.shape[0]):
            for j in range(xco2.shape[1]):
                logic = (snd_id_std==snd_id_l1b[i, j])
                if logic.sum() == 1:
                    xco2[i, j] = xco2_std[logic]
                    sfc_pres[i, j] = sfc_pres_std[logic]
                elif logic.sum() > 1:
                    sys.exit('Error   [oco_rad_nadir]: More than one point is found.')

        self.xco2      = xco2
        self.sfc_pres  = sfc_pres

    def get_data(self, index_s=0, index_e=None):

        f       = h5py.File(self.fname_l1b, 'r')
        if index_e is None:
            self.rad_o2_a       = f['SoundingMeasurements/radiance_o2'][...][index_s:, ...]
            self.rad_co2_weak   = f['SoundingMeasurements/radiance_weak_co2'][...][index_s:, ...]
            self.rad_co2_strong = f['SoundingMeasurements/radiance_strong_co2'][...][index_s:, ...]
            self.sza            = f['SoundingGeometry/sounding_solar_zenith'][...][index_s:, ...]
            self.saa            = f['SoundingGeometry/sounding_solar_azimuth'][...][index_s:, ...]
        else:
            self.rad_o2_a       = f['SoundingMeasurements/radiance_o2'][...][index_s:index_e, ...]
            self.rad_co2_weak   = f['SoundingMeasurements/radiance_weak_co2'][...][index_s:index_e, ...]
            self.rad_co2_strong = f['SoundingMeasurements/radiance_strong_co2'][...][index_s:index_e, ...]
            self.sza            = f['SoundingGeometry/sounding_solar_zenith'][...][index_s:index_e, ...]
            self.saa            = f['SoundingGeometry/sounding_solar_azimuth'][...][index_s:index_e, ...]

        for i in range(8):
            self.rad_o2_a[:, i, :]       = convert_photon_unit(self.rad_o2_a[:, i, :]      , self.get_wvl_o2_a(i))
            self.rad_co2_weak[:, i, :]   = convert_photon_unit(self.rad_co2_weak[:, i, :]  , self.get_wvl_co2_weak(i))
            self.rad_co2_strong[:, i, :] = convert_photon_unit(self.rad_co2_strong[:, i, :], self.get_wvl_co2_strong(i))
        f.close()

def cdata_all(date, tag, fdir_mca, fname_abs, sat):

    print(date)
    print(tag)

    # ==================================================================================================
    oco = oco2_rad_nadir(sat, tag)

    wvl_o2a  = np.zeros_like(oco.rad_o2_a      , dtype=np.float64)
    wvl_wco2 = np.zeros_like(oco.rad_co2_weak  , dtype=np.float64)
    wvl_sco2 = np.zeros_like(oco.rad_co2_strong, dtype=np.float64)
    for i in range(oco.rad_o2_a.shape[0]):
        for j in range(oco.rad_o2_a.shape[1]):
            wvl_o2a[i, j, :]  = oco.get_wvl_o2_a(j)
            wvl_wco2[i, j, :] = oco.get_wvl_co2_weak(j)
            wvl_sco2[i, j, :] = oco.get_wvl_co2_strong(j)
    # ==================================================================================================

    # ==================================================================================================
    """f = readsav(fname_idl)
    wvls  = f.lamx * 1000.0
    wvls  = np.sort(wvls)
    trans = f.tx[np.argsort(f.lamx)]"""
    f = h5py.File('data/atm_abs_o2a_11.h5', 'r')
    wvls = f['lamx'][...]*1000.0
    wvls  = np.sort(wvls)
    trans = f['tx'][...][np.argsort(f['lamx'][...])]



    modl1b = modis_l1b(fnames=sat.fnames['mod_02'], extent=sat.extent)
    lon_2d, lat_2d, rad_2d_mod = grid_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['rad']['data'][0, ...], extent=sat.extent)

    rad_mca_ipa0 = np.zeros((wvl_o2a.shape[0], wvl_o2a.shape[1], wvls.size), dtype=np.float64)
    rad_mca_ipa  = np.zeros((wvl_o2a.shape[0], wvl_o2a.shape[1], wvls.size), dtype=np.float64)
    rad_mca_3d   = np.zeros((wvl_o2a.shape[0], wvl_o2a.shape[1], wvls.size), dtype=np.float64)

    rad_mca_ipa0_std = np.zeros((wvl_o2a.shape[0], wvl_o2a.shape[1], wvls.size), dtype=np.float64)
    rad_mca_ipa_std  = np.zeros((wvl_o2a.shape[0], wvl_o2a.shape[1], wvls.size), dtype=np.float64)
    rad_mca_3d_std   = np.zeros((wvl_o2a.shape[0], wvl_o2a.shape[1], wvls.size), dtype=np.float64)

    # modified ==============================================
    fname = glob.glob('%s/mca-out-rad-oco2-ipa0_%s*nm.h5' % (fdir_mca, ('%.4f' % wvls[0])[:-1]))[0]
    f = h5py.File(fname, 'r')
    rad_ipa0     = f['mean/rad'][...]
    f.close()
    rad_mca_ipa0_domain = np.zeros((rad_ipa0.shape[0], rad_ipa0.shape[1], wvls.size), dtype=np.float64)
    rad_mca_ipa_domain  = np.zeros((rad_ipa0.shape[0], rad_ipa0.shape[1], wvls.size), dtype=np.float64)
    rad_mca_3d_domain   = np.zeros((rad_ipa0.shape[0], rad_ipa0.shape[1], wvls.size), dtype=np.float64)

    rad_mca_ipa0_domain_std = np.zeros((rad_ipa0.shape[0], rad_ipa0.shape[1], wvls.size), dtype=np.float64)
    rad_mca_ipa_domain_std  = np.zeros((rad_ipa0.shape[0], rad_ipa0.shape[1], wvls.size), dtype=np.float64)
    rad_mca_3d_domain_std   = np.zeros((rad_ipa0.shape[0], rad_ipa0.shape[1], wvls.size), dtype=np.float64)

    # =======================================================

    toa = np.zeros(wvls.size, dtype=np.float64)
    Np = np.zeros(wvls.size, dtype=np.float64);

    for k in range(wvls.size):
        print(wvls[k])

        fname = glob.glob('%s/mca-out-rad-oco2-ipa0_%s*nm.h5' % (fdir_mca, ('%.4f' % wvls[k])[:-1]))[0]
        f = h5py.File(fname, 'r')
        rad_ipa0     = f['mean/rad'][...]
        rad_ipa0_std = f['mean/rad_std'][...]
        f.close()

        fname = glob.glob('%s/mca-out-rad-oco2-ipa_%s*nm.h5' % (fdir_mca, ('%.4f' % wvls[k])[:-1]))[0]
        f = h5py.File(fname, 'r')
        rad_ipa     = f['mean/rad'][...]
        rad_ipa_std = f['mean/rad_std'][...]
        f.close()

        fname = glob.glob('%s/mca-out-rad-oco2-3d_%s*nm.h5' % (fdir_mca, ('%.4f' % wvls[k])[:-1]))[0]
        f = h5py.File(fname, 'r')
        rad_3d     = f['mean/rad'][...]
        rad_3d_std = f['mean/rad_std'][...]
        toa0       = f['mean/toa'][...]
        photons    = f['mean/N_photon'][...]
        f.close()
        toa[k] = toa0
        Np[k] = photons.sum()

        # ===================================
        rad_mca_ipa0_domain[:, :, k] = rad_ipa0.copy()
        rad_mca_ipa_domain[:, :, k]  = rad_ipa.copy()
        rad_mca_3d_domain[:, :, k]   = rad_3d.copy()

        rad_mca_ipa0_domain_std[:, :, k] = rad_ipa0_std.copy()
        rad_mca_ipa_domain_std[:, :, k]  = rad_ipa_std.copy()
        rad_mca_3d_domain_std[:, :, k]   = rad_3d_std.copy()
        # ===================================

        """
        if k == np.argmax(trans):
            rad_mca_3d_domain     = rad_3d.copy()
            rad_mca_3d_domain_std = rad_3d_std.copy()
        """
        for i in range(wvl_o2a.shape[0]):
            for j in range(wvl_o2a.shape[1]):
                lon0 = oco.lon_l1b[i, j]
                lat0 = oco.lat_l1b[i, j]
                if tag == 'o2a':
                    lon0 = oco.lon_l1b_o2a[i, j]
                    lon0 = oco.lon_l1b_o2a[i, j]
                elif tag == 'wco2':
                    lon0 = oco.lon_l1b_wco2[i, j]
                    lon0 = oco.lon_l1b_wco2[i, j]
                elif tag == 'sco2':
                    lon0 = oco.lon_l1b_sco2[i, j]
                    lon0 = oco.lon_l1b_sco2[i, j]
                index_lon = np.argmin(np.abs(lon_2d[:, 0]-lon0))
                index_lat = np.argmin(np.abs(lat_2d[0, :]-lat0))

                rad_mca_ipa0[i, j, k] = rad_ipa0[index_lon, index_lat]
                rad_mca_ipa[i, j, k]  = rad_ipa[index_lon, index_lat]
                rad_mca_3d[i, j, k]   = rad_3d[index_lon, index_lat]

                rad_mca_ipa0_std[i, j, k] = rad_ipa0_std[index_lon, index_lat]
                rad_mca_ipa_std[i, j, k]  = rad_ipa_std[index_lon, index_lat]
                rad_mca_3d_std[i, j, k]   = rad_3d_std[index_lon, index_lat]
    # ==================================================================================================


    f = h5py.File('data_all_%s_%s_%4.4d_%4.4d.h5' % (date.strftime('%Y%m%d'), tag, oco.index_s, oco.index_e), 'w')
    f['lon']          = oco.lon_l1b
    f['lat']          = oco.lat_l1b
    f['logic']        = oco.logic_l1b
    f['toa']          = toa
    f['Np']           = Np

    if tag == 'o2a':
        f['lon_fp']          = oco.lon_l1b_o2a
        f['lat_fp']          = oco.lon_l1b_o2a
        f['rad_oco']  = oco.rad_o2_a
        f['wvl_oco']  = wvl_o2a
    elif tag == 'wco2':
        f['lon_fp']          = oco.lon_l1b_wco2
        f['lat_fp']          = oco.lon_l1b_wco2
        f['rad_oco']  = oco.rad_co2_weak
        f['wvl_oco']  = wvl_wco2
    elif tag == 'sco2':
        f['lon_fp']          = oco.lon_l1b_sco2
        f['lat_fp']          = oco.lon_l1b_sco2
        f['rad_oco']  = oco.rad_co2_strong
        f['wvl_oco']  = wvl_sco2

    f['snd_id'] = oco.snd_id
    f['xco2']   = oco.xco2
    f['sfc_pres'] = oco.sfc_pres
    f['sza'] = oco.sza
    f['saa'] = oco.saa

    logic = (oco.lon_l1b>=sat.extent[0]) & (oco.lon_l1b<=sat.extent[1]) & (oco.lat_l1b>=sat.extent[2]) & (oco.lat_l1b<=sat.extent[3])
    sza_mca = np.zeros_like(oco.sza)
    saa_mca = np.zeros_like(oco.saa)
    sza_mca[...] = oco.sza[logic].mean()
    saa_mca[...] = oco.saa[logic].mean()
    f['sza_mca'] = sza_mca
    f['saa_mca'] = saa_mca

    f['rad_mca_3d']   = rad_mca_3d
    f['rad_mca_ipa']  = rad_mca_ipa
    f['rad_mca_ipa0'] = rad_mca_ipa0
    f['rad_mca_3d_std']   = rad_mca_3d_std
    f['rad_mca_ipa_std']  = rad_mca_ipa_std
    f['rad_mca_ipa0_std'] = rad_mca_ipa0_std
    # ==============
    f['lon2d']          = lon_2d
    f['lat2d']          = lat_2d
    f['rad_mca_3d_domain']   = rad_mca_3d_domain
    f['rad_mca_ipa_domain']  = rad_mca_ipa_domain
    f['rad_mca_ipa0_domain'] = rad_mca_ipa0_domain
    f['rad_mca_3d_domain_std']   = rad_mca_3d_domain_std
    f['rad_mca_ipa_domain_std']  = rad_mca_ipa_domain_std
    f['rad_mca_ipa0_domain_std'] = rad_mca_ipa0_domain_std
    # ==============
    f['wvl_mca']                = wvls
    f['tra_mca']                = trans
    #f['rad_mca_3d_domain']      = rad_mca_3d_domain
    #f['rad_mca_3d_domain_std']  = rad_mca_3d_domain_std
    f['extent_domain']          = sat.extent
    f.close()

    return 'data_all_%s_%s_%4.4d_%4.4d.h5' % (date.strftime('%Y%m%d'), tag, oco.index_s, oco.index_e)


def run_case(band_tag, cfg_info):

    # define date and region to study
    # ===============================================================
    date   = datetime.datetime(int(cfg_info['date'][:4]),    # year
                               int(cfg_info['date'][4:6]),   # month
                               int(cfg_info['date'][6:])     # day
                              )
    extent = [float(loc) for loc in cfg_info['subdomain']]
    extent[0] -= 0.15 
    extent[1] += 0.15
    extent[2] -= 0.15 
    extent[3] += 0.15
    print(extent)
    ref_threshold = float(cfg_info['ref_threshold'])

    name_tag = '%s_%s' % (cfg_info['cfg_name'], date.strftime('%Y%m%d'))
    # ===============================================================

    # create data/australia_YYYYMMDD directory if it does not exist
    # ===============================================================
    fdir_data = os.path.abspath('data/%s' % name_tag)
    if not os.path.exists(fdir_data):
        os.makedirs(fdir_data)
    # ===============================================================

    # download satellite data based on given date and region
    # ===============================================================
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(date=date, fdir_out=fdir_data, extent=extent, fname=fname_sat, overwrite=False)
    # ===============================================================
    if not ('l2' in cfg_info.keys()):
        save_h5_info(cfg_info['cfg_path'], 'l2',  sat0.fnames['oco_std'][0].split('/')[-1])
        save_h5_info(cfg_info['cfg_path'], 'met', sat0.fnames['oco_met'][0].split('/')[-1])
        save_h5_info(cfg_info['cfg_path'], 'l1b', sat0.fnames['oco_l1b'][0].split('/')[-1])
        save_h5_info(cfg_info['cfg_path'], 'lt',  sat0.fnames['oco_lite'][0].split('/')[-1])
        save_h5_info(cfg_info['cfg_path'], 'dia', sat0.fnames['oco_dia'][0].split('/')[-1])
        save_h5_info(cfg_info['cfg_path'], 'imap', sat0.fnames['oco_imap'][0].split('/')[-1])
        save_h5_info(cfg_info['cfg_path'], 'co2prior', sat0.fnames['oco_co2propr'][0].split('/')[-1])
    # create tmp-data/australia_YYYYMMDD directory if it does not exist
    # ===============================================================
    fdir_tmp = os.path.abspath('tmp-data/%s/%s' % (name_tag, band_tag))
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)
    fdir_cot_tmp = os.path.abspath('tmp-data/%s/cot' % (name_tag))
    if not os.path.exists(fdir_cot_tmp):
        os.makedirs(fdir_cot_tmp)
    # ===============================================================

    # create atmosphere based on OCO-Met and CO2_prior
    # ===============================================================
    zpt_file = f'{fdir_data}/zpt.h5'
    zpt_file = os.path.abspath(zpt_file)
    if not os.path.isfile(zpt_file):
        create_oco_atm(sat=sat0, o2mix=0.20935, output=zpt_file)
    # ===============================================================


    # read out wavelength information from Sebastian's absorption file
    # ===============================================================

    nx = int(cfg_info['nx'])
    fname_abs = f'{fdir_data}/atm_abs_{band_tag}_{(nx+1):d}.h5'
    iband_dict = {'o2a':0, 'wco2':1, 'sco2':2,}
    Trn_min = float(cfg_info['Trn_min'])
    if not os.path.isfile(fname_abs):
        oco_abs(cfg, zpt_file=zpt_file, iband=iband_dict[band_tag], nx=nx, Trn_min=Trn_min, pathout=fdir_data, reextract=False, plot=True)
    f = h5py.File(fname_abs, 'r')
    wvls = f['lamx'][...]*1000.0
    #sys.exit()

    print(wvls)
    # ===============================================================
    #"""
    for solver in ['IPA', '3D']:
        preprocessing(band_tag, sat0, cth=None, scale_factor=1.0, fdir=fdir_tmp, fdir_cot=fdir_cot_tmp, zpt_file=zpt_file,
                      sfc_scale=True, sfc_replace=True, solver=solver, ref_threshold=ref_threshold, overwrite=True)
    #"""
    #sys.exit()
    # run calculations for each wavelength
    #"""
    """
    # ===============================================================
    for wavelength in wvls:
        for solver in ['IPA', '3D']:
            cal_mca_rad_oco2(date, band_tag, sat0, wavelength, fname_idl=fname_abs, cth=None, scale_factor=1.0, sfc_scale=True, sfc_replace=True, fdir=fdir_tmp, solver=solver, overwrite=True, photons=5e7, ref_threshold=ref_threshold)
    # ===============================================================
    #"""

    # post-processing - combine the all the calculations into one dataset
    # ===============================================================
    #return cdata_all(date, band_tag, fdir_tmp, fname_abs, sat0)
    # ===============================================================
    #"""


def run_simulation(cfg):
    cfg_info = grab_cfg(cfg)
    print(cfg_info.keys())
    #"""
    if 1:#not check_h5_info(cfg, 'o2'):
        starttime = timeit.default_timer()
        o2_h5 = run_case('o2a', cfg_info)
        save_h5_info(cfg, 'o2', o2_h5)
        endtime = timeit.default_timer()
        print('O2A band duration:',(endtime-starttime)/3600.,' h')
    #""" 
    """
    if not check_h5_info(cfg, 'wco2'):
        starttime = timeit.default_timer()
        wco2_h5 = run_case('wco2', cfg_info)
        save_h5_info(cfg, 'wco2', wco2_h5)
        endtime = timeit.default_timer()
        print('WCO2 band duration:',(endtime-starttime)/3600.,' h')
    
    if not check_h5_info(cfg, 'sco2'):
        starttime = timeit.default_timer()
        sco2_h5 = run_case('sco2', cfg_info)
        save_h5_info(cfg, 'sco2', sco2_h5)
        endtime = timeit.default_timer()
        print('SCO2 band duration:',(endtime-starttime)/3600.,' h')
    #"""

if __name__ == '__main__':
    

    #cfg = 'cfg/20181018_550cloud.csv'
    #cfg = 'cfg/20170605_amazon_470cloud_aod.csv'
    #cfg = 'cfg/20170721_australia_470cloud_aod.csv'
    cfg = 'cfg/20181018_central_asia_2_470cloud_test.csv'
    #cfg = 'cfg/20190621_australia-2-470cloud_aod.csv'
    print(cfg)
    run_simulation(cfg)


    



