

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
from er3t.util.modis import modis_l1b, modis_l2, modis_03, modis_04, modis_09a1, modis_43a3
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
from oco_subroutine.oco_sfc import cal_sfc_alb_2d

import timeit
import argparse
import matplotlib.image as mpl_img
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


def sfc_alb_mask_inter(lon_alb, lat_alb, sfc_alb, lon_2d, lat_2d):
    sfc_alb[sfc_alb<0.0] = 0.0
    sfc_alb[sfc_alb>1.0] = 1.0
    mask = sfc_alb>=0
    points = np.transpose(np.vstack((lon_alb[mask].flatten(), lat_alb[mask].flatten())))
    sfc_alb_inter = interpolate.griddata(points, sfc_alb[mask].flatten(), 
                                        (lon_2d, lat_2d), method='nearest')
    return sfc_alb_inter
    


def cdata_sat_raw(oco_band, sat0, overwrite=False, plot=True):
    """
    oco_band: 'o2a', 'wco2', 'sco2'
    """
    band_list = ['o2a', 'wco2', 'sco2']
    # process wavelength
    #/----------------------------------------------------------------------------\#
    if oco_band.lower() == 'o2a':
        wvl = 650
        index_wvl = 0      # select MODIS 650 nm band radiance/reflectance for IPA cloud retrieval
        wvl_sfc = 860
        index_wvl_sfc = 1  # select MODIS 860 nm band surface albedo for scaling
    elif oco_band.lower() == 'wco2':
        wvl = 1640
        index_wvl = 5      # select MODIS 650 nm band radiance/reflectance for IPA cloud retrieval
        wvl_sfc = 1640
        index_wvl_sfc = 5  # select MODIS 860 nm band surface albedo for scaling
    elif oco_band.lower() == 'sco2':
        wvl = 2130
        index_wvl = 6      # select MODIS 650 nm band radiance/reflectance for IPA cloud retrieval
        wvl_sfc = 2130
        index_wvl_sfc = 6  # select MODIS 860 nm band surface albedo for scaling
    else:
        msg = '\nError [cdata_sat_raw]: Currently, only <oco_band=\'o2a\'> is supported.>'
        sys.exit(msg)
    #\----------------------------------------------------------------------------/#

    """ # download satellite data based on given date and region
    #/----------------------------------------------------------------------------\#
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(date=params['date'], fdir_out=fdir_data, extent=params['region'], fname=fname_sat, overwrite=False)
    #\----------------------------------------------------------------------------/#
    """

    # pre-process downloaded data
    #/----------------------------------------------------------------------------\#
    if os.path.isfile(f'{sat0.fdir_out}/pre-data.h5') and not overwrite:
        print(f'Message [pre_data]: {sat0.fdir_out}/pre-data.h5 exsit.')
        return None
    if 1:#elif not os.path.isfile(f'{sat0.fdir_out}/pre-data.h5'):
        f0 = h5py.File(f'{sat0.fdir_out}/pre-data.h5', 'w')
        f0['extent'] = sat0.extent

        # MODIS data groups in the HDF file
        #/--------------------------------------------------------------\#
        g  = f0.create_group('mod')
        g0 = g.create_group('geo')
        g1 = g.create_group('rad')
        g2 = g.create_group('cld')
        g3 = g.create_group('sfc')
        g4 = g.create_group('aod')

        #\--------------------------------------------------------------/#

        # MODIS RGB
        #/--------------------------------------------------------------\#
        mod_rgb = mpl_img.imread(sat0.fnames['mod_rgb'][0])
        g['rgb'] = mod_rgb
        print('Message [cdata_sat_raw]: the processing of MODIS RGB imagery is complete.')
        #\--------------------------------------------------------------/#


        # MODIS radiance/reflectance at 650 nm
        #/--------------------------------------------------------------\#
        modl1b = er3t.util.modis_l1b(fnames=sat0.fnames['mod_02'], extent=sat0.extent)
        lon0  = modl1b.data['lon']['data']
        lat0  = modl1b.data['lat']['data']
        ref_650_raw  = modl1b.data['ref']['data'][0, ...]
        rad_650_raw  = modl1b.data['rad']['data'][0, ...]
        lon_2d, lat_2d, ref_650_2d = er3t.util.grid_by_extent(lon0, lat0, ref_650_raw, extent=sat0.extent)
        _, _, rad_650_2d = er3t.util.grid_by_extent(lon0, lat0, rad_650_raw, extent=sat0.extent)

        g1['ref_650'] = ref_650_2d
        g1['rad_650'] = rad_650_2d

        modl1b_500m = modis_l1b(fnames=sat0.fnames['mod_02_hkm'], extent=sat0.extent)
        lon0_500m  = modl1b_500m.data['lon']['data']
        lat0_500m  = modl1b_500m.data['lat']['data']
        ref_2d_470_raw = modl1b_500m.data['ref']['data'][0, ...]
        ref_2d_555_raw = modl1b_500m.data['ref']['data'][1, ...]
        ref_2d_1640_raw = modl1b_500m.data['ref']['data'][3, ...]
        rad_2d_1640_raw = modl1b_500m.data['rad']['data'][3, ...]
        ref_2d_2130_raw = modl1b_500m.data['ref']['data'][4, ...]
        rad_2d_2130_raw = modl1b_500m.data['rad']['data'][4, ...]

        lon_2d_500m, lat_2d_500m, ref_2d_470 = er3t.util.grid_by_extent(lon0_500m, lat0_500m, ref_2d_470_raw, extent=sat0.extent)
        _, _, ref_2d_555 = er3t.util.grid_by_extent(lon0_500m, lat0_500m, ref_2d_555_raw, extent=sat0.extent)
        _, _, ref_2d_1640 = er3t.util.grid_by_extent(lon0_500m, lat0_500m, ref_2d_1640_raw, extent=sat0.extent)
        _, _, rad_2d_1640 = er3t.util.grid_by_extent(lon0_500m, lat0_500m, rad_2d_1640_raw, extent=sat0.extent)
        _, _, ref_2d_2130 = er3t.util.grid_by_extent(lon0_500m, lat0_500m, ref_2d_2130_raw, extent=sat0.extent)
        _, _, rad_2d_2130 = er3t.util.grid_by_extent(lon0_500m, lat0_500m, rad_2d_2130_raw, extent=sat0.extent)

        for var_name in ['ref_2d_470', 'ref_2d_555', 'ref_2d_1640', 'rad_2d_1640', 'ref_2d_2130', 'rad_2d_2130']:
            var = vars()[var_name]
            mask = var>=0


            points_mask = np.transpose(np.vstack((lon_2d_500m[mask].flatten(), lat_2d_500m[mask].flatten())))
            vars()[f'{var_name}_inter'] = interpolate.griddata(points_mask, var[mask].flatten(), (lon_2d, lat_2d), method='linear')

        g1['ref_470'] = vars()[f'ref_2d_470_inter']
        g1['ref_555'] = vars()[f'ref_2d_555_inter']
        g1['ref_1640'] = vars()[f'ref_2d_1640_inter']
        g1['rad_1640'] = vars()[f'ref_2d_1640_inter']
        g1['ref_2130'] = vars()[f'ref_2d_2130_inter']
        g1['rad_2130'] = vars()[f'ref_2d_2130_inter']


        print('Message [cdata_sat_raw]: the processing of MODIS L1B radiance/reflectance at %d nm is complete.' % wvl)

        f0['lon'] = lon_2d
        f0['lat'] = lat_2d

        lon_1d = lon_2d[:, 0]
        lat_1d = lat_2d[0, :]
        #\--------------------------------------------------------------/#


        # MODIS geo information - sza, saa, vza, vaa
        #/--------------------------------------------------------------\#
        mod03 = er3t.util.modis_03(fnames=sat0.fnames['mod_03'], extent=sat0.extent, vnames=['Height'])
        lon0  = mod03.data['lon']['data']
        lat0  = mod03.data['lat']['data']
        sza0  = mod03.data['sza']['data']
        saa0  = mod03.data['saa']['data']
        vza0  = mod03.data['vza']['data']
        vaa0  = mod03.data['vaa']['data']
        sfh0  = mod03.data['height']['data']/1000.0 # units: km
        sfh0[sfh0<0.0] = np.nan

        _, _, sza_2d = er3t.util.grid_by_lonlat(lon0, lat0, sza0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')
        _, _, saa_2d = er3t.util.grid_by_lonlat(lon0, lat0, saa0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')
        _, _, vza_2d = er3t.util.grid_by_lonlat(lon0, lat0, vza0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')
        _, _, vaa_2d = er3t.util.grid_by_lonlat(lon0, lat0, vaa0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')
        _, _, sfh_2d = er3t.util.grid_by_lonlat(lon0, lat0, sfh0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')

        g0['sza'] = sza_2d
        g0['saa'] = saa_2d
        g0['vza'] = vza_2d
        g0['vaa'] = vaa_2d
        g0['sfh'] = sfh_2d

        print('Message [cdata_sat_raw]: the processing of MODIS geo-info is complete.')
        #\--------------------------------------------------------------/#


        # cloud properties
        #/--------------------------------------------------------------\#
        modl2 = er3t.util.modis_l2(fnames=sat0.fnames['mod_l2'], extent=sat0.extent, vnames=['cloud_top_height_1km'])

        lon0  = modl2.data['lon']['data']
        lat0  = modl2.data['lat']['data']
        cer0  = modl2.data['cer']['data']
        cot0  = modl2.data['cot']['data']

        cth0  = modl2.data['cloud_top_height_1km']['data']/1000.0 # units: km
        cth0[cth0<=0.0] = np.nan

        lon_2d, lat_2d, cer_2d_l2 = er3t.util.grid_by_lonlat(lon0, lat0, cer0, lon_1d=lon_1d, lat_1d=lat_1d, method='nearest')
        cer_2d_l2[cer_2d_l2<=1.0] = np.nan

        lon_2d, lat_2d, cot_2d_l2 = er3t.util.grid_by_lonlat(lon0, lat0, cot0, lon_1d=lon_1d, lat_1d=lat_1d, method='nearest')
        cot_2d_l2[cot_2d_l2<=0.0] = np.nan

        lon_2d, lat_2d, cth_2d_l2 = er3t.util.grid_by_lonlat(lon0, lat0, cth0, lon_1d=lon_1d, lat_1d=lat_1d, method='linear')
        cth_2d_l2[cth_2d_l2<=0.0] = np.nan

        g2['cot_l2'] = cot_2d_l2
        g2['cer_l2'] = cer_2d_l2
        g2['cth_l2'] = cth_2d_l2

        print('Message [cdata_sat_raw]: the processing of MODIS cloud properties is complete.')
        #\--------------------------------------------------------------/#


        # surface
        #/--------------------------------------------------------------\#
        # Extract and grid MODIS surface reflectance
        #   band 1: 620  - 670  nm, index 0
        #   band 2: 841  - 876  nm, index 1
        #   band 3: 459  - 479  nm, index 2
        #   band 4: 545  - 565  nm, index 3
        #   band 5: 1230 - 1250 nm, index 4
        #   band 6: 1628 - 1652 nm, index 5
        #   band 7: 2105 - 2155 nm, index 6
        wavelength_list = [650, 860, 470, 555, 1240, 1640, 2130]
        mod09 = er3t.util.modis_09a1(fnames=sat0.fnames['mod_09'], extent=sat0.extent)
        mod43 = er3t.util.modis_43a3(fnames=sat0.fnames['mcd_43'], extent=sat0.extent)
        for wv_index in range(7):
            
            lon_2d_sfc, lat_2d_sfc, vars()[f'sfc_09_{wavelength_list[wv_index]:d}'] = er3t.util.grid_by_extent(mod09.data['lon']['data'], mod09.data['lat']['data'], mod09.data['ref']['data'][wv_index, :], extent=sat0.extent)
            vars()[f'sfc_09_{wavelength_list[wv_index]:d}'] = sfc_alb_mask_inter(lon_2d_sfc, lat_2d_sfc, vars()[f'sfc_09_{wavelength_list[wv_index]:d}'], lon_2d, lat_2d)
            lon_2d_sfc, lat_2d_sfc, vars()[f'sfc_43_{wavelength_list[wv_index]:d}'] = er3t.util.grid_by_extent(mod43.data['lon']['data'], mod43.data['lat']['data'], mod43.data['wsa']['data'][wv_index, :], extent=sat0.extent)
            vars()[f'sfc_43_{wavelength_list[wv_index]:d}'] = sfc_alb_mask_inter(lon_2d_sfc, lat_2d_sfc, vars()[f'sfc_43_{wavelength_list[wv_index]:d}'], lon_2d, lat_2d)

        """lon_2d_sfc, lat_2d_sfc, sfc_09_650 = er3t.util.grid_by_extent(mod09.data['lon']['data'], mod09.data['lat']['data'], mod09.data['ref']['data'][0, :], extent=sat0.extent)
        sfc_09_650 = sfc_alb_mask_inter(lon_2d_sfc, lat_2d_sfc, sfc_09_650, lon_2d, lat_2d)
        lon_2d_sfc, lat_2d_sfc, sfc_09_860 = er3t.util.grid_by_extent(mod09.data['lon']['data'], mod09.data['lat']['data'], mod09.data['ref']['data'][1, :], extent=sat0.extent)
        sfc_09_860 = sfc_alb_mask_inter(lon_2d_sfc, lat_2d_sfc, sfc_09_860, lon_2d, lat_2d)
        lon_2d_sfc, lat_2d_sfc, sfc_09_1640 = er3t.util.grid_by_extent(mod09.data['lon']['data'], mod09.data['lat']['data'], mod09.data['ref']['data'][5, :], extent=sat0.extent)
        sfc_09_1640 = sfc_alb_mask_inter(lon_2d_sfc, lat_2d_sfc, sfc_09_1640, lon_2d, lat_2d)
        lon_2d_sfc, lat_2d_sfc, sfc_09_2130 = er3t.util.grid_by_extent(mod09.data['lon']['data'], mod09.data['lat']['data'], mod09.data['ref']['data'][6, :], extent=sat0.extent)
        sfc_09_2130 = sfc_alb_mask_inter(lon_2d_sfc, lat_2d_sfc, sfc_09_2130, lon_2d, lat_2d)

 

        mod43 = er3t.util.modis_43a3(fnames=sat0.fnames['mcd_43'], extent=sat0.extent)
        lon_2d_sfc, lat_2d_sfc, sfc_43_650 = er3t.util.grid_by_extent(mod43.data['lon']['data'], mod43.data['lat']['data'], mod43.data['wsa']['data'][0, :], extent=sat0.extent)
        sfc_43_650 = sfc_alb_mask_inter(lon_2d_sfc, lat_2d_sfc, sfc_43_650, lon_2d, lat_2d)
        lon_2d_sfc, lat_2d_sfc, sfc_43_860 = er3t.util.grid_by_extent(mod43.data['lon']['data'], mod43.data['lat']['data'], mod43.data['wsa']['data'][1, :], extent=sat0.extent)
        sfc_43_860 = sfc_alb_mask_inter(lon_2d_sfc, lat_2d_sfc, sfc_43_860, lon_2d, lat_2d)
        lon_2d_sfc, lat_2d_sfc, sfc_43_1640 = er3t.util.grid_by_extent(mod43.data['lon']['data'], mod43.data['lat']['data'], mod43.data['wsa']['data'][5, :], extent=sat0.extent)
        sfc_43_1640 = sfc_alb_mask_inter(lon_2d_sfc, lat_2d_sfc, sfc_43_1640, lon_2d, lat_2d)
        lon_2d_sfc, lat_2d_sfc, sfc_43_2130 = er3t.util.grid_by_extent(mod43.data['lon']['data'], mod43.data['lat']['data'], mod43.data['wsa']['data'][6, :], extent=sat0.extent)
        sfc_43_2130 = sfc_alb_mask_inter(lon_2d_sfc, lat_2d_sfc, sfc_43_2130, lon_2d, lat_2d)"""

        sfc_43_o2a = vars()[f'sfc_43_860']
        sfc_43_wco2 = vars()[f'sfc_43_1640']
        sfc_43_sco2 = vars()[f'sfc_43_2130']

        g3['lon'] = lon_2d_sfc
        g3['lat'] = lat_2d_sfc
        for wavelength in wavelength_list:
            g3['alb_09_%d' % wavelength] = vars()[f'sfc_09_{wavelength:d}']
            g3['alb_43_%d' % wavelength] = vars()[f'sfc_43_{wavelength:d}']

        for band_tag in band_list:
            g3[f'alb_43_{band_tag}'] = vars()[f'sfc_43_{band_tag}']

        print('Message [cdata_sat_raw]: the processing of MODIS surface properties is complete.')
        #\--------------------------------------------------------------/#

        # aerosol
        #/--------------------------------------------------------------\#
        mcd04 = modis_04(fnames=sat0.fnames['mod_04'], extent=sat0.extent, 
                        vnames=['Deep_Blue_Spectral_Single_Scattering_Albedo_Land', ])
        AOD_lon, AOD_lat, AOD_550_land = er3t.util.grid_by_extent(mcd04.data['lon']['data'], mcd04.data['lat']['data'], mcd04.data['AOD_550_land']['data'], extent=sat0.extent)
        _, _, Angstrom_Exponent_land = er3t.util.grid_by_extent(mcd04.data['lon']['data'], mcd04.data['lat']['data'], mcd04.data['Angstrom_Exponent_land']['data'], extent=sat0.extent)
        _, _, SSA_land_660 = er3t.util.grid_by_extent(mcd04.data['lon']['data'], mcd04.data['lat']['data'], mcd04.data['deep_blue_spectral_single_scattering_albedo_land']['data'], extent=sat0.extent)

        
        #_, _, aerosol_type_land = grid_by_extent(mcd04.data['lon']['data'], mcd04.data['lat']['data'], mcd04.data['aerosol_type_land']['data'], extent=sat.extent)
        #_, _, aerosol_cloud_frac_land = grid_by_extent(mcd04.data['lon']['data'], mcd04.data['lat']['data'], mcd04.data['aerosol_cloud_frac_land']['data'], extent=sat.extent)

        AOD_550_land_nan = AOD_550_land.copy()
        AOD_550_land_nan[np.isnan(AOD_550_land_nan)] = np.nan
        AOD_550_land_nan[AOD_550_land_nan<0] = np.nan
        SSA_land_660_nan = SSA_land_660.copy()
        SSA_land_660_nan[np.isnan(SSA_land_660_nan)] = np.nan
        SSA_land_660_nan[SSA_land_660_nan<0] = np.nan

        AOD_550_land_mean = np.nanmean(AOD_550_land_nan[AOD_550_land>=0])
        Angstrom_Exponent_land_mean = np.nanmean(Angstrom_Exponent_land[AOD_550_land>=0])
        SSA_land_mean = np.nanmean(SSA_land_660_nan[AOD_550_land>=0])

        g4['AOD_550_land_mean'] = AOD_550_land_mean
        g4['Angstrom_Exponent_land_mean'] = Angstrom_Exponent_land_mean
        g4['SSA_660_land_mean'] = SSA_land_mean

        #/--------------------------------------------------------------\#


        # OCO-2 data groups in the HDF file
        #/--------------------------------------------------------------\#
        gg = f0.create_group('oco')
        gg1 = gg.create_group('o2a')
        gg2 = gg.create_group('geo')
        gg3 = gg.create_group('met')
        gg4 = gg.create_group('sfc')
        #\--------------------------------------------------------------/#

        # Read OCO-2 radiance and wavelength data
        #/--------------------------------------------------------------\#
        oco = er3t.util.oco2_rad_nadir(sat0)

        wvl_o2a  = np.zeros_like(oco.rad_o2_a, dtype=np.float64)
        for i in range(oco.rad_o2_a.shape[0]):
            for j in range(oco.rad_o2_a.shape[1]):
                wvl_o2a[i, j, :]  = oco.get_wvl_o2_a(j)
        #\--------------------------------------------------------------/#

        # OCO L1B
        #/--------------------------------------------------------------\#
        gg['lon'] = oco.lon_l1b
        gg['lat'] = oco.lat_l1b
        gg['logic']  = oco.logic_l1b
        gg['snd_id'] = oco.snd_id
        gg1['rad']   = oco.rad_o2_a
        gg1['wvl']   = wvl_o2a
        gg2['sza'] = oco.sza
        gg2['saa'] = oco.saa
        gg2['vza'] = oco.vza
        gg2['vaa'] = oco.vaa
        print('Message [cdata_sat_raw]: the processing of OCO-2 radiance is complete.')
        #\--------------------------------------------------------------/#

        # OCO wind speed
        #/--------------------------------------------------------------\#
        # extract wind speed (10m wind)
        f = h5py.File(sat0.fnames['oco_met'][0], 'r')
        lon_oco_met0 = f['SoundingGeometry/sounding_longitude'][...]
        lat_oco_met0 = f['SoundingGeometry/sounding_latitude'][...]
        u_10m0 = f['Meteorology/windspeed_u_met'][...]
        v_10m0 = f['Meteorology/windspeed_v_met'][...]
        logic = (np.abs(u_10m0)<50.0) & (np.abs(v_10m0)<50.0) & \
                (lon_oco_met0>=sat0.extent[0]) & (lon_oco_met0<=sat0.extent[1]) & \
                (lat_oco_met0>=sat0.extent[2]) & (lat_oco_met0<=sat0.extent[3])
        f.close()

        gg3['lon'] = lon_oco_met0[logic]
        gg3['lat'] = lat_oco_met0[logic]
        gg3['u_10m'] = u_10m0[logic]
        gg3['v_10m'] = v_10m0[logic]
        gg3['delta_t'] = cal_sat_delta_t(sat0)
        print('Message [cdata_sat_raw]: the processing of OCO-2 meteorological data is complete.')
        #\--------------------------------------------------------------/#


        # OCO-2 surface reflectance
        #/--------------------------------------------------------------\#
        # process wavelength
        band_list = ['o2a', 'wco2', 'sco2']
        vname_dict = {'o2a':'brdf_reflectance_o2',
                      'wco2':'brdf_reflectance_weak_co2',
                      'sco2':'brdf_reflectance_strong_co2'}
        for band_tag in band_list:
            vname = vname_dict[band_tag]
            oco = er3t.util.oco2_std(fnames=sat0.fnames['oco_std'], vnames=['BRDFResults/%s' % vname], extent=sat0.extent)

            oco_sfc_alb = oco.data[vname]['data']
            oco_sfc_alb[oco_sfc_alb<0.0] = 0.0

            oco_lon = oco.data['lon']['data']
            oco_lat = oco.data['lat']['data']
            logic = (oco_sfc_alb>0.0) & (oco_lon>=sat0.extent[0]) & (oco_lon<=sat0.extent[1]) & (oco_lat>=sat0.extent[2]) & (oco_lat<=sat0.extent[3])
            oco_lon = oco_lon[logic]
            oco_lat = oco_lat[logic]
            oco_sfc_alb = oco_sfc_alb[logic]

            gg4['alb_%s' % band_tag] = oco_sfc_alb

            oco_sfc_alb_2d = cal_sfc_alb_2d(oco_lon, oco_lat, oco_sfc_alb, lon_2d, lat_2d, vars()[f'sfc_43_{band_tag}'], scale=True, replace=True)
            gg4['alb_%s_2d' % band_tag] = oco_sfc_alb_2d

        gg4['lon'] = oco_lon
        gg4['lat'] = oco_lat
        

        
        print('Message [cdata_sat_raw]: the processing of OCO-2 surface reflectance is complete.')
        #\--------------------------------------------------------------/#

        f0.close()
    
    #/----------------------------------------------------------------------------\#

    if plot:

        f0 = h5py.File(f'{sat0.fdir_out}/pre-data.h5', 'r')
        extent = f0['extent'][...]

        rgb = f0['mod/rgb'][...]
        rad = f0['mod/rad/rad_%d' % 650][...]
        ref = f0['mod/rad/ref_%d' % 650][...]

        sza = f0['mod/geo/sza'][...]
        saa = f0['mod/geo/saa'][...]
        vza = f0['mod/geo/vza'][...]
        vaa = f0['mod/geo/vaa'][...]

        cot = f0['mod/cld/cot_l2'][...]
        cer = f0['mod/cld/cer_l2'][...]
        cth = f0['mod/cld/cth_l2'][...]
        sfh = f0['mod/geo/sfh'][...]

        alb09 = f0['mod/sfc/alb_09_%d' % wvl][...]
        alb43 = f0['mod/sfc/alb_43_%d' % wvl][...]

        f0.close()

        # figure
        #/----------------------------------------------------------------------------\#
        plt.close('all')
        rcParams['font.size'] = 12
        fig = plt.figure(figsize=(16, 16))

        fig.suptitle('MODIS Products Preview')

        # RGB
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(441)
        cs = ax1.imshow(rgb, zorder=0, extent=extent)
        ax1.set_title('RGB Imagery')

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cax.axis('off')
        #\--------------------------------------------------------------/#

        # L1B radiance
        #/----------------------------------------------------------------------------\#
        ax2 = fig.add_subplot(442)
        cs = ax2.imshow(rad.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=0.5)
        ax2.set_title('L1B Radiance (%d nm)' % wvl)

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # L1B reflectance
        #/----------------------------------------------------------------------------\#
        ax3 = fig.add_subplot(443)
        cs = ax3.imshow(ref.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=1.0)
        ax3.set_title('L1B Reflectance (%d nm)' % wvl)

        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # sza
        #/----------------------------------------------------------------------------\#
        ax5 = fig.add_subplot(445)
        cs = ax5.imshow(sza.T, origin='lower', cmap='jet', zorder=0, extent=extent)
        ax5.set_title('Solar Zenith [$^\circ$]')

        divider = make_axes_locatable(ax5)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # saa
        #/----------------------------------------------------------------------------\#
        ax6 = fig.add_subplot(446)
        cs = ax6.imshow(saa.T, origin='lower', cmap='jet', zorder=0, extent=extent)
        ax6.set_title('Solar Azimuth [$^\circ$]')

        divider = make_axes_locatable(ax6)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # vza
        #/----------------------------------------------------------------------------\#
        ax7 = fig.add_subplot(447)
        cs = ax7.imshow(vza.T, origin='lower', cmap='jet', zorder=0, extent=extent)
        ax7.set_title('Viewing Zenith [$^\circ$]')

        divider = make_axes_locatable(ax7)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # vaa
        #/----------------------------------------------------------------------------\#
        ax8 = fig.add_subplot(448)
        cs = ax8.imshow(vaa.T, origin='lower', cmap='jet', zorder=0, extent=extent)
        ax8.set_title('Viewing Azimuth [$^\circ$]')

        divider = make_axes_locatable(ax8)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cot
        #/----------------------------------------------------------------------------\#
        ax9 = fig.add_subplot(449)
        cs = ax9.imshow(cot.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=50.0)
        ax9.set_title('L2 COT')

        divider = make_axes_locatable(ax9)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cer
        #/----------------------------------------------------------------------------\#
        ax10 = fig.add_subplot(4, 4, 10)
        cs = ax10.imshow(cer.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=30.0)
        ax10.set_title('L2 CER [$\mu m$]')

        divider = make_axes_locatable(ax10)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cth
        #/----------------------------------------------------------------------------\#
        ax11 = fig.add_subplot(4, 4, 11)
        cs = ax11.imshow(cth.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=15.0)
        ax11.set_title('L2 CTH [km]')

        divider = make_axes_locatable(ax11)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # sfh
        #/----------------------------------------------------------------------------\#
        ax12 = fig.add_subplot(4, 4, 12)
        cs = ax12.imshow(sfh.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=5.0)
        ax12.set_title('Surface Height [km]')

        divider = make_axes_locatable(ax12)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # surface albedo (MYD09A1, reflectance)
        #/----------------------------------------------------------------------------\#
        ax13 = fig.add_subplot(4, 4, 13)
        cs = ax13.imshow(alb09.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=0.4)
        ax13.set_title('09A1 Reflectance at %d nm' % wvl_sfc)

        divider = make_axes_locatable(ax13)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # surface albedo (MYD43A3, white sky albedo)
        #/----------------------------------------------------------------------------\#
        ax14 = fig.add_subplot(4, 4, 14)
        cs = ax14.imshow(alb43.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=0.4)
        ax14.set_title('43A3 WSA at %d nm' % wvl_sfc)

        divider = make_axes_locatable(ax14)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        ax_list = [f'ax{num}' for num in range(1, 15)]
        ax_list.remove('ax4')
        for num in range(len(ax_list)):
            ax = vars()[ax_list[num]]
            ax.set_xlim((extent[:2]))
            ax.set_ylim((extent[2:]))
            ax.set_xlabel('Longitude [$^\circ$]')
            ax.set_ylabel('Latitude [$^\circ$]')

        # save figure
        #/--------------------------------------------------------------\#
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s/<%s>.png' % (sat0.fdir_out, _metadata['Function']), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        #\----------------------------------------------------------------------------/#
