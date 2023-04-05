#!/bin/env python

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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_16g, abs_oco_idl
from er3t.pre.cld import cld_sat
from er3t.pre.sfc import sfc_sat
from er3t.pre.pha import pha_mie_wc as pha_mie # newly added for phase function
from er3t.util.modis import modis_l1b, modis_l2, modis_03, modis_09a1, modis_43a3, grid_modis_by_extent, grid_modis_by_lonlat, download_modis_https, get_sinusoidal_grid_tag, get_filename_tag, download_modis_rgb
from er3t.util.oco2 import oco2_std, download_oco2_https
from er3t.util import cal_r_twostream

from er3t.rtm.mca_v010 import mca_atm_1d, mca_atm_3d, mca_sfc_2d
from er3t.rtm.mca_v010 import mcarats_ng
from er3t.rtm.mca_v010 import mca_out_ng
from er3t.rtm.mca_v010 import mca_sca # newly added for phase function

import timeit
import argparse
import matplotlib.image as mpl_img

class satellite_download:

    def __init__(
            self,
            date=None,
            extent=None,
            fname=None,
            fdir_out='data',
            overwrite=False,
            quiet=False,
            verbose=False):

        self.date     = date
        self.extent   = extent
        self.extent_simulation = extent
        self.fdir_out = fdir_out
        self.quiet    = quiet
        self.verbose  = verbose

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif (((date is not None) and (extent is not None)) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             (((date is not None) and (extent is not None)) and (fname is not None) and (not os.path.exists(fname))):

            self.run()
            self.dump(fname)

        elif (((date is not None) and (extent is not None)) and (fname is None)):

            self.run()

        else:

            sys.exit('Error   [satellite_download]: Please check if \'%s\' exists or provide \'date\' and \'extent\' to proceed.' % fname)

    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'fnames') and hasattr(obj, 'extent') and hasattr(obj, 'fdir_out') and hasattr(obj, 'date'):
                if self.verbose:
                    print('Message [satellite_download]: Loading %s ...' % fname)
                self.date     = obj.date
                self.extent   = obj.extent
                self.fnames   = obj.fnames
                self.fdir_out = obj.fdir_out
            else:
                sys.exit('Error   [satellite_download]: File \'%s\' is not the correct pickle file to load.' % fname)

    def run(self, run=True):

        lon = np.array(self.extent[:2])
        lat = np.array(self.extent[2:])

        self.fnames = {}

        self.fnames['mod_rgb'] = [download_modis_rgb(self.date, self.extent, fdir=self.fdir_out, which='aqua', coastline=True)]

        # MODIS Level 2 Cloud Product and MODIS 03 geo file
        self.fnames['mod_l2'] = []
        self.fnames['mod_02_1km'] = []
        self.fnames['mod_02'] = []
        self.fnames['mod_03'] = []

        modis_parent_fdir = 'data/modis'
        if not os.path.exists(modis_parent_fdir):
            os.makedirs(modis_parent_fdir)
        modis_fdir = f"{modis_parent_fdir}/{self.date.strftime('%Y%m%d')}"
        if not os.path.exists(modis_fdir):
            os.makedirs(modis_fdir)

        filename_tags_03 = get_filename_tag(self.date, lon, lat, satID='aqua')
        for filename_tag in filename_tags_03:
            fnames_l2 = download_modis_https(self.date, '61/MYD06_L2', filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            fnames_02_1km = download_modis_https(self.date, '61/MYD021KM', filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            fnames_02 = download_modis_https(self.date, '61/MYD02QKM', filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            fnames_03 = download_modis_https(self.date, '61/MYD03'   , filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            self.fnames['mod_l2'] += fnames_l2
            self.fnames['mod_02_1km'] += fnames_02_1km
            self.fnames['mod_02'] += fnames_02
            self.fnames['mod_03'] += fnames_03

        # MOD09A1 surface reflectance product
        self.fnames['mod_09'] = []
        filename_tags_09 = get_sinusoidal_grid_tag(lon, lat)
        for filename_tag in filename_tags_09:
            fnames_09 = download_modis_https(self.date, '61/MOD09A1', filename_tag, day_interval=8, fdir_out=modis_fdir, run=run)
            self.fnames['mod_09'] += fnames_09
        # MOD43A3 surface reflectance product
        self.fnames['mcd_43'] = []
        filename_tags_43 = get_sinusoidal_grid_tag(lon, lat)
        for filename_tag in filename_tags_43:
            fnames_43 = download_modis_https(self.date, '61/MCD43A3', filename_tag, day_interval=16, fdir_out=modis_fdir, run=run)
            self.fnames['mcd_43'] += fnames_43

        # OCO2 std and met file
        self.fnames['oco_std'] = []
        self.fnames['oco_met'] = []
        self.fnames['oco_l1b'] = []
        oco_fdir = 'data/oco'
        if not os.path.exists(oco_fdir):
            os.makedirs(oco_fdir)
        for filename_tag in filename_tags_03:
            dtime = datetime.datetime.strptime(filename_tag, 'A%Y%j.%H%M') + datetime.timedelta(minutes=7.0)
            fnames_std = download_oco2_https(dtime, 'OCO2_L2_Standard.10r', fdir_out=oco_fdir, run=run)
            fnames_met = download_oco2_https(dtime, 'OCO2_L2_Met.10r'     , fdir_out=oco_fdir, run=run)
            fnames_l1b = download_oco2_https(dtime, 'OCO2_L1B_Science.10r', fdir_out=oco_fdir, run=run)
            self.fnames['oco_std'] += fnames_std
            self.fnames['oco_met'] += fnames_met
            self.fnames['oco_l1b'] += fnames_l1b

    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [satellite_download]: Saving object into %s ...' % fname)
            pickle.dump(self, f)



def para_corr(lon0, lat0, vza, vaa, cld_h, sfc_h, R_earth=6378000.0, verbose=True):

    """
    Parallax correction for the cloud positions

    lon0: input longitude
    lat0: input latitude
    vza : sensor zenith angle [degree]
    vaa : sensor azimuth angle [degree]
    cld_h: cloud height [meter]
    sfc_h: surface height [meter]
    """

    if verbose:
        print('Message [para_corr]: Please make sure the units of \'cld_h\' and \'sfc_h\' are in \'meter\'.')

    dist = (cld_h-sfc_h)*np.tan(np.deg2rad(vza))

    delta_lon = dist*np.sin(np.deg2rad(vaa)) / (np.pi*R_earth) * 180.0
    delta_lat = dist*np.cos(np.deg2rad(vaa)) / (np.pi*R_earth) * 180.0

    lon = lon0 + delta_lon
    lat = lat0 + delta_lat

    return lon, lat

def wind_corr(lon0, lat0, u, v, dt, R_earth=6378000.0, verbose=True):

    """
    Wind correction for the cloud positions

    lon0: input longitude
    lat0: input latitude
    u   : meridional wind [meter/second], positive when eastward
    v   : zonal wind [meter/second], positive when northward
    dt  : delta time [second]
    """

    if verbose:
        print('Message [wind_corr]: Please make sure the units of \'u\' and \'v\' are in \'meter/second\' and \'dt\' in \'second\'.')
        print('Message [wind_corr]: U: %.4f m/s; V: %.4f m/s; Time offset: %.2f s' % (u, v, dt))

    delta_lon = (u*dt) / (np.pi*R_earth) * 180.0
    delta_lat = (v*dt) / (np.pi*R_earth) * 180.0

    lon = lon0 + delta_lon
    lat = lat0 + delta_lat

    return lon, lat

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

def rgb2hsv(rgb):
    """ convert RGB to HSV color space

    :param rgb: np.ndarray
    :return: np.ndarray
    """

    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv

    return hsv

def pre_cld(sat, tag, cth=None, cot_source='2stream', ref_threshold=0.1, scale_factor=1.0, solver='3D'):

    # retrieve 1. cloud top height; 2. sensor zenith; 3. sensor azimuth for MODIS L1B (250nm) data from MODIS L2 (5km resolution)
    
    # Extract
    #   1. cloud top height (cth, 1km resolution);
    #   2. solar zenith and azimuth angles (sza and saa, 1km resolution);
    #   3. sensor zenith and azimuth angles (vza and vaa, 1km resolution);
    #   4. surface height (sfc, 1km resolution)
    # ===================================================================================
 
    # ===================================================================================
    now = time.time()
    modl2      = modis_l2(fnames=sat.fnames['mod_l2'], extent=sat.extent, vnames=['Sensor_Zenith', 'Sensor_Azimuth', 'Cloud_Top_Height'])
    #logic_cth  = (modl2.data['cloud_top_height']['data']>0.0)
    logic_cth  = np.logical_and(modl2.data['cloud_top_height']['data']>0.0, ~np.isnan(modl2.data['cloud_top_height']['data']))
    lon0       = modl2.data['lon_5km']['data']
    lat0       = modl2.data['lat_5km']['data']
    cth0       = modl2.data['cloud_top_height']['data']/1000.0 # units: km
    cot0       = modl2.data['cot']['data']
    # for MODIS cer
    lon_cer    = modl2.data['lon']['data']
    lat_cer    = modl2.data['lat']['data']
    cer0       = modl2.data['cer']['data']
    print(f'Load modis L2 data: {(time.time()-now)/60} min')

    #"""
    # 1km cth
    now = time.time()
    modl2_1km      = modis_l2(fnames=sat.fnames['mod_l2'], extent=sat.extent,
                              vnames=['cloud_top_height_1km', 'cloud_top_temperature_1km',], 
                              resolution='1km')
    lon_1km       = modl2_1km.data['lon']['data']
    lat_1km       = modl2_1km.data['lat']['data']
    cth_1km       = modl2_1km.data['cloud_top_height_1km']['data']/1000.0 
    #logic_cth_1km  = np.logical_and(modl2_1km.data['cloud_top_height_1km']['data']>0.0, ~np.isnan(modl2_1km.data['cloud_top_height_1km']['data']))
    logic_cth_1km  = np.logical_and(modl2_1km.data['cloud_top_height_1km']['data']>0.0, modl2_1km.data['cloud_top_height_1km']['data']<14000)
    print(f'Load modis L2 1km data: {(time.time()-now)/60} min')
    #"""
    
    now = time.time()
    mod03      = modis_03(fnames=sat.fnames['mod_03'], extent=sat.extent, vnames=['Height'])
    logic_sfh  = (mod03.data['height']['data']>0.0)
    lon1       = mod03.data['lon']['data']
    lat1       = mod03.data['lat']['data']
    sfh1       = mod03.data['height']['data']/1000.0 # units: km
    sza1       = mod03.data['sza']['data']
    saa1       = mod03.data['saa']['data']
    vza1       = mod03.data['vza']['data']
    vaa1       = mod03.data['vaa']['data']
    print(f'Load modis 03 data: {(time.time()-now)/60} min')

    # Process MODIS radiance and reflectance at 650 nm (250m resolution)
    # ===================================================================================
    now = time.time()
    modl1b = modis_l1b(fnames=sat.fnames['mod_02'], extent=sat.extent)
    lon_2d, lat_2d, ref_2d = grid_modis_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['ref']['data'][0, ...], extent=sat.extent)
    lon_2d, lat_2d, rad_2d = grid_modis_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['rad']['data'][0, ...], extent=sat.extent)
    print(f'Load modis l1b data: {(time.time()-now)/60} min')

    """    
    #print('wavelength: {}nm'.format(modl1b.data['wvl']['data'][0]))
    _, _, ref_650_2d = grid_modis_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['ref']['data'][0, ...], extent=sat.extent)#, NxNy=(110*2, 148*2))
    _, _, rad_650_2d = grid_modis_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['rad']['data'][0, ...], extent=sat.extent)#, NxNy=(110*2, 148*2))
    #print('wavelength: {}nm'.format(modl1b.data['wvl']['data'][1]))
    _, _, ref_860_2d = grid_modis_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['ref']['data'][1, ...], extent=sat.extent)#, NxNy=(110*2, 148*2))
    _, _, rad_860_2d = grid_modis_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['rad']['data'][1, ...], extent=sat.extent)#, NxNy=(110*2, 148*2))
    """    
    # ===================================================================================


    # Process MODIS radiance and reflectance at 650/860 nm (250m resolution)
    # ===================================================================================
    now = time.time()
    mod09 = modis_09a1(fnames=sat.fnames['mod_09'], extent=sat.extent)
    mod09_lon_2d, mod09_lat_2d, mod_sfc_alb_2d = grid_modis_by_extent(mod09.data['lon']['data'], mod09.data['lat']['data'], mod09.data['ref']['data'][0, :], extent=sat.extent)
    
    mcd43 = modis_43a3(fnames=sat.fnames['mcd_43'], extent=sat.extent)
    points = np.transpose(np.vstack((mcd43.data['lon']['data'], mcd43.data['lat']['data'])))
    _, _, mcd_bsa_2d = grid_modis_by_extent(mcd43.data['lon']['data'], mcd43.data['lat']['data'], mcd43.data['bsa']['data'][0, :], extent=sat.extent)
    _, _, mcd_wsa_2d = grid_modis_by_extent(mcd43.data['lon']['data'], mcd43.data['lat']['data'], mcd43.data['wsa']['data'][0, :], extent=sat.extent)
    
    mcd_bsa_2d[mcd_bsa_2d<0] = np.nan
    mcd_wsa_2d[mcd_wsa_2d<0] = np.nan

    points = np.transpose(np.vstack((mod09_lon_2d.flatten(), mod09_lat_2d.flatten())))
    mod09_inter = interpolate.griddata(points, mod_sfc_alb_2d.flatten(), 
                                           (lon_2d, lat_2d), method='nearest')
    
    mask = mcd_bsa_2d>=0
    points_mask = np.transpose(np.vstack((mod09_lon_2d[mask].flatten(), mod09_lat_2d[mask].flatten())))
    mcd_bsa_inter = interpolate.griddata(points_mask, mcd_bsa_2d[mask].flatten(), 
                                           (lon_2d, lat_2d), method='nearest')
    mask = mcd_wsa_2d>=0
    points_mask = np.transpose(np.vstack((mod09_lon_2d[mask].flatten(), mod09_lat_2d[mask].flatten())))
    mcd_wsa_inter = interpolate.griddata(points_mask, mcd_wsa_2d[mask].flatten(), 
                                           (lon_2d, lat_2d), method='nearest')
    print(f'Load modis 09 and 43 data: {(time.time()-now)/60} min')

    now = time.time()
    sfc_alb_inter = mcd_bsa_inter
    sza_lon, sza_lat, sza_2d = grid_modis_by_extent(lon1, lat1, sza1, extent=sat.extent)
    points = np.transpose(np.vstack((sza_lon.flatten(), sza_lat.flatten())))
    sza_inter = interpolate.griddata(points, sza_2d.flatten(), 
                                           (lon_2d, lat_2d), method='linear')
    print(f'Interpolating sfc alb and sza: {(time.time()-now)/60} min')

    # ===================================================================================

    #----------------------------------------------
    mod_rgb = mpl_img.imread(sat.fnames['mod_rgb'][0])

    """
    lon_rgb0 = np.linspace(sat.extent[0], sat.extent[1], mod_rgb.shape[1]+1)
    lat_rgb0 = np.linspace(sat.extent[2], sat.extent[3], mod_rgb.shape[0]+1)
    lon_rgb = (lon_rgb0[1:]+lon_rgb0[:-1])/2.0
    lat_rgb = (lat_rgb0[1:]+lat_rgb0[:-1])/2.0
    lon_rgb_2d, lat_rgb_2d = np.meshgrid(lon_rgb, lat_rgb[::-1], indexing='xy')

    mod_r = mod_rgb[:, :, 0]
    mod_g = mod_rgb[:, :, 1]
    mod_b = mod_rgb[:, :, 2]
    mod_hsv = rgb2hsv(mod_rgb[:,:,:3])
    mod_h = mod_hsv[:, :, 0]
    mod_s = mod_hsv[:, :, 1]
    mod_v = mod_hsv[:, :, 2]
    mod_v_over_s = mod_v/mod_s
    """
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

    now = time.time()
    logic_ref_nan = (ref_2d-sfc_alb_inter) < ref_threshold
    indices    = np.where(logic_ref_nan!=1)
    indices_x  = indices[0]
    indices_y  = indices[1]
    lon        = lon_2d[indices_x, indices_y]
    lat        = lat_2d[indices_x, indices_y]
    print(f'Cloud detection time: {(time.time()-now)/60} min')



    print('max cloud top height:', np.max(cth_1km))
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
        _, _, cer_2d_l2 = grid_modis_by_lonlat(lon_cer, lat_cer, cer0, lon_1d=lon_2d[:, 0], lat_1d=lat_2d[0, :], method='linear')
        cer_2d_l2[cer_2d_l2<=1] = 1         # make sure the minimum cer is 1
        print(f'CER griding time: {(time.time()-now)/60} min')

        # =============================================================
        
        # Upscale cloud optical thickness from 1km (L2) to 250m resolution
        # =============================================================
        now = time.time()
        cot        = interpolate.griddata(points_cer, cot0, (lon, lat), method='nearest')
        _, _, cot_2d_l2 = grid_modis_by_lonlat(lon_cer, lat_cer, cot0, lon_1d=lon_2d[:, 0], lat_1d=lat_2d[0, :], method='nearest')

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
    u_oco = f['Meteorology/windspeed_u_met'][...][logic]
    v_oco = f['Meteorology/windspeed_v_met'][...][logic]
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

    xx_2stream = np.linspace(0.0, 200.0, 10000)
    yy_2stream = cal_r_twostream(xx_2stream, a=a0, mu=mu0)  #g0 still constant for now

    # lon/lat shift due to parallax and wind correction
    lon_1d = lon_2d[:, 0]
    indices_x_new = np.int_(np.round((lon_corr-lon_1d[0])/(((lon_1d[1:]-lon_1d[:-1])).mean()), decimals=0))
    lat_1d = lat_2d[0, :]
    indices_y_new = np.int_(np.round((lat_corr-lat_1d[0])/(((lat_1d[1:]-lat_1d[:-1])).mean()), decimals=0))



    Nx, Ny = ref_2d.shape
    cot_2d_l1b = np.zeros_like(ref_2d)
    cer_2d_l1b = np.ones_like(ref_2d)# minimum cer is set to 1
    cth_2d_l1b = np.zeros_like(ref_2d)
    now = time.time()
    for i in range(indices_x.size):
        if 0<=indices_x_new[i]<Nx and 0<=indices_y_new[i]<Ny:
            cer_2d_l1b[indices_x_new[i], indices_y_new[i]] = cer_2d_l2[indices_x[i], indices_y[i]]
            cth_2d_l1b[indices_x_new[i], indices_y_new[i]] = cth_2d_l2[indices_x[i], indices_y[i]]
            if cot_source=='2stream':
                # COT from two-stream
                cot_2d_l1b[indices_x_new[i], indices_y_new[i]] = xx_2stream[np.argmin(np.abs(yy_2stream-ref_2d[indices_x[i], indices_y[i]]))]
            elif cot_source=='MODIS':
                # COT from closest COT from MODIS L2 cloud product
                cot_2d_l1b[indices_x_new[i], indices_y_new[i]] = cot_2d_l2[indices_x[i], indices_y[i]]
            else:
                sys.exit('Error  [pre_cld_modis]: cot_select must be either "2stream" or "MODIS" ')
    print(f'New Cloud position correction time: {(time.time()-now)/60} min')
    modl1b.data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=lon_2d)
    modl1b.data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=lat_2d)
    modl1b.data['rad_2d'] = dict(name='Gridded radiance'                , units='W/m^2/nm/sr', data=rad_2d)
    modl1b.data['ref_2d'] = dict(name='Gridded reflectance'             , units='N/A'        , data=ref_2d)
    modl1b.data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=cot_2d_l1b)#scale_factor)
    modl1b.data['cot_2d_1km'] = dict(name='Gridded cloud optical thickness' , units='N/A'    , data=cot_2d_l2)
    modl1b.data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=cth_2d_l1b)
    modl1b.data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=cer_2d_l1b)
    #modl1b.data['wvl']    = dict(name='Wavelength'                      , units='nm'         , data=wvl)
    modl1b.data['sza']    = dict(name='Solar Zenith Angle'              , units='degree'     , data=np.nanmean(sza1))
    modl1b.data['saa']    = dict(name='Solar Azimuth Angle'             , units='degree'     , data=np.nanmean(saa1))
    modl1b.data['vza']    = dict(name='Sensor Zenith Angle'             , units='degree'     , data=np.nanmean(vza1))
    modl1b.data['vaa']    = dict(name='Sensor Azimuth Angle'            , units='degree'     , data=np.nanmean(vaa1))

    return modl1b

def pre_sfc(sat, tag, version='10r', scale=True, replace=True):

    if version == '7':
        vnames = [
                'AlbedoResults/albedo_o2_fph',              # 0.77 microns
                'AlbedoResults/albedo_slope_o2',
                'AlbedoResults/albedo_strong_co2_fph',      # 2.06 microns
                'AlbedoResults/albedo_slope_strong_co2',
                'AlbedoResults/albedo_weak_co2_fph',        # 1.615 microns
                'AlbedoResults/albedo_slope_weak_co2'
                  ]
    elif version == '10' or version == '10r':
        vnames = [
                'BRDFResults/brdf_reflectance_o2',              # 0.77 microns
                'BRDFResults/brdf_reflectance_slope_o2',
                'BRDFResults/brdf_reflectance_strong_co2',      # 2.06 microns
                'BRDFResults/brdf_reflectance_slope_strong_co2',
                'BRDFResults/brdf_reflectance_weak_co2',        # 1.615 microns
                'BRDFResults/brdf_reflectance_slope_weak_co2'
                  ]
    else:
        exit('Error   [pre_sfc]: Cannot recognize version \'%s\'.' % version)
    #print(sat.fnames['oco_std'])
    oco = oco2_std(fnames=sat.fnames['oco_std'], vnames=vnames, extent=sat.extent)
    oco_lon = oco.data['lon']['data']
    oco_lat = oco.data['lat']['data']

    if version == '7':

        if tag.lower() == 'o2a':
            oco_sfc_alb = oco.data['albedo_o2_fph']['data']
        elif tag.lower() == 'wco2':
            oco_sfc_alb = oco.data['albedo_weak_co2_fph']['data']
        elif tag.lower() == 'sco2':
            oco_sfc_alb = oco.data['albedo_strong_co2_fph']['data']

    elif version == '10' or version == '10r':

        if tag.lower() == 'o2a':
            oco_sfc_alb = oco.data['brdf_reflectance_o2']['data']
        elif tag.lower() == 'wco2':
            oco_sfc_alb = oco.data['brdf_reflectance_weak_co2']['data']
        elif tag.lower() == 'sco2':
            oco_sfc_alb = oco.data['brdf_reflectance_strong_co2']['data']

    else:
        exit('Error   [cdata_sfc_alb]: Cannot recognize version \'%s\'.' % version)
    logic = (oco_sfc_alb>0.0) & (oco_lon>=sat.extent[0]) & (oco_lon<=sat.extent[1]) & (oco_lat>=sat.extent[2]) & (oco_lat<=sat.extent[3])
    oco_lon = oco_lon[logic]
    oco_lat = oco_lat[logic]
    oco_sfc_alb = oco_sfc_alb[logic]


    # MODIS band information
    # band 1: 620  - 670  nm, index 0
    # band 2: 841  - 876  nm, index 1
    # band 3: 459  - 479  nm, index 2
    # band 4: 545  - 565  nm, index 3
    # band 5: 1230 - 1250 nm, index 4
    # band 6: 1628 - 1652 nm, index 5
    # band 7: 2105 - 2155 nm, index 6

    #mod = modis_09a1(fnames=sat.fnames['mod_09'], extent=sat.extent)
    mod = modis_43a3(fnames=sat.fnames['mcd_43'], extent=sat.extent)
    #lon_2d, lat_2d, mcd_bsa_2d = grid_modis_by_extent(mod.data['lon']['data'], mod.data['lat']['data'], mod.data['bsa']['data'][index[wvl], :], extent=sat.extent)
    #lon_2d, lat_2d, mcd_wsa_2d = grid_modis_by_extent(mod.data['lon']['data'], mod.data['lat']['data'], mod.data['bsa']['data'][index[wvl], :], extent=sat.extent)
    mod43_mode = 'bsa'
    points = np.transpose(np.vstack((mod.data['lon']['data'], mod.data['lat']['data'])))
    if tag.lower() == 'o2a':
        lon_2d, lat_2d, mod_sfc_alb_2d = grid_modis_by_extent(mod.data['lon']['data'], mod.data['lat']['data'], mod.data[mod43_mode]['data'][1, :], extent=sat.extent)
    elif tag.lower() == 'wco2':
        lon_2d, lat_2d, mod_sfc_alb_2d = grid_modis_by_extent(mod.data['lon']['data'], mod.data['lat']['data'], mod.data[mod43_mode]['data'][5, :], extent=sat.extent)
    elif tag.lower() == 'sco2':
        lon_2d, lat_2d, mod_sfc_alb_2d = grid_modis_by_extent(mod.data['lon']['data'], mod.data['lat']['data'], mod.data[mod43_mode]['data'][6, :], extent=sat.extent)

    mask = mod_sfc_alb_2d>=0
    points = np.transpose(np.vstack((lon_2d[mask].flatten(), lat_2d[mask].flatten())))
    mod_sfc_alb_2d_inter = interpolate.griddata(points, mod_sfc_alb_2d[mask].flatten(), 
                                           (lon_2d, lat_2d), method='nearest')

    mod03      = modis_03(fnames=sat.fnames['mod_03'], extent=sat.extent, vnames=['Height'])
    lon1       = mod03.data['lon']['data']
    lat1       = mod03.data['lat']['data']
    sza1       = mod03.data['sza']['data']

    sza_lon, sza_lat, sza_2d = grid_modis_by_extent(lon1, lat1, sza1, extent=sat.extent)
    mask = sza_2d>0
    points = np.transpose(np.vstack((sza_lon[mask].flatten(), sza_lat[mask].flatten())))
    print('sza_ori filtered albedo NaN:', np.isnan(sza_2d[mask]).any())
    sza_inter = interpolate.griddata(points, sza_2d[mask].flatten(), 
                                           (lon_2d, lat_2d), method='nearest')
    print('sza albedo NaN:', np.isnan(sza_inter).any())
    print('sfc albedo NaN:', np.isnan(mod_sfc_alb_2d_inter).any())
    mod_sfc_alb_2d = mod_sfc_alb_2d_inter/np.cos(sza_inter/180*np.pi)

    oco_sfc_alb_2d = create_sfc_alb_2d(oco_lon, oco_lat, oco_sfc_alb, lon_2d, lat_2d, mod_sfc_alb_2d, scale=scale, replace=replace)

    mod.data['alb_2d'] = dict(data=oco_sfc_alb_2d, name='Surface albedo', units='N/A')
    mod.data['mod43_2d'] = dict(data=mod_sfc_alb_2d, name='Black-sky albedo', units='N/A')
    mod.data['lon_2d'] = dict(data=lon_2d, name='Longitude', units='degrees')
    mod.data['lat_2d'] = dict(data=lat_2d, name='Latitude' , units='degrees')

    return mod

def preprocessing(tag, sat, cth=None, scale_factor=1.0, fdir='tmp-data',
                  sfc_scale=True, sfc_replace=True, solver='3D', ref_threshold=0.1, overwrite=True):

    if os.path.isfile(f'{sat.fdir_out}/pre-data.h5') and not overwrite:
        print(f'Message [pre_data]: {sat.fdir_out}/pre-data.h5 exsit.')
        return None
    elif not os.path.isfile(f'{sat.fdir_out}/pre-data.h5'):
        f0 = h5py.File(f'{sat.fdir_out}/pre-data.h5', 'w')
        f0['extent'] = sat.extent
        # MODIS data groups in the HDF file
        # ==================================================================================================
        g = f0.create_group('mod')
        g1 = g.create_group('rad')
        g2 = g.create_group('cld')
        g3 = g.create_group('sfc')
        # ==================================================================================================

        # MODIS RGB
        # ==================================================================================================
        mod_rgb = mpl_img.imread(sat.fnames['mod_rgb'][0])
        g['rgb'] = mod_rgb
        print('Message [pre_data]: the processing of MODIS RGB imagery is complete.')
        # ==================================================================================================
    else:
        f0 = h5py.File(f'{sat.fdir_out}/pre-data.h5', 'a')
        g = f0['mod']
        g1 = g['rad']
        g2 = g['cld']
        g3 = g['sfc']
     
    # cloud optical properties
    # ==================================================================================================
    if not all([g1.__contains__('lon'), g1.__contains__(f'rad_{tag}'), g2.__contains__(f'cot_{solver}')]):
        modl1b    = pre_cld(sat, tag, cth=cth, scale_factor=scale_factor, solver=solver, ref_threshold=ref_threshold)
        if not g1.__contains__('lon'):
            g1['lon'] = modl1b.data['lon_2d']['data']
            g1['lat'] = modl1b.data['lat_2d']['data']
            g1['sza'] = modl1b.data['sza']['data']
            g1['saa'] = modl1b.data['saa']['data']
            g1['vza'] = modl1b.data['vza']['data']
            g1['vaa'] = modl1b.data['vaa']['data']
            g2['cot_l2'] = modl1b.data['cot_2d_1km']['data'] 
        if not g1.__contains__(f'rad_{tag}'):
            g1[f'rad_{tag}'] = modl1b.data['rad_2d']['data']
            g1[f'ref_{tag}'] = modl1b.data['ref_2d']['data']
        if not g2.__contains__(f'cot_{solver}'):
            g2[f'cot_{solver}'] = modl1b.data['cot_2d']['data']
            g2[f'cer_{solver}'] = modl1b.data['cer_2d']['data']
            g2[f'cth_{solver}'] = modl1b.data['cth_2d']['data']
    print('Message [pre_data]: the processing of cloud optical properties is complete.')
    # ==================================================================================================

    # surface albedo
    # ==================================================================================================
    if not all([g3.__contains__('lon'), g3.__contains__(f'alb_{tag}')]):
        mod_sfc = pre_sfc(sat, tag, scale=sfc_scale, replace=sfc_replace)
        if not g3.__contains__('lon'):
            g3['lon'] = mod_sfc.data['lon_2d']['data']
            g3['lat'] = mod_sfc.data['lat_2d']['data']
        if not g3.__contains__(f'alb_{tag}'):
            g3[f'alb_{tag}'] = mod_sfc.data['alb_2d']['data']
            g3[f'mod43_{tag}'] = mod_sfc.data['mod43_2d']['data']
    print('Message [pre_data]: the processing of surface albedo is complete.')
    # ==================================================================================================

    f0.close()

    return None


class sat_tmp:

    def __init__(self, data):

        self.data = data

def cal_mca_rad_oco2(date, tag, sat, wavelength, fname_idl=None, cth=None, photons=1e8, scale_factor=1.0, 
                     sfc_scale=True, sfc_replace=True, fdir='tmp-data', solver='3D', ref_threshold=0.1, overwrite=True):

    """
    Calculate OCO2 radiance using cloud (MODIS level 1b) and surface properties (MOD09A1) from MODIS
    """

    # atm object
    # =================================================================================
    levels    = np.array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. , 5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. , 11. , 12. , 13. , 14. , 20. , 25. , 30. , 35. , 40. ])
    fname_atm = '%s/atm.pk' % fdir
    atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    # =================================================================================

    # abs object, in the future, we will implement OCO2 MET file for this
    # =================================================================================
    fname_abs = '%s/abs.pk' % fdir
    abs0      = abs_oco_idl(wavelength=wavelength, fname=fname_abs, fname_idl=fname_idl, atm_obj=atm0, overwrite=overwrite)
    # =================================================================================

    # mca_sfc object
    # =================================================================================
    #"""
    data = {}
    f = h5py.File(f'{sat.fdir_out}/pre-data.h5', 'r')
    data['alb_2d'] = dict(data=f[f'mod/sfc/alb_{tag}'][...], name='Surface albedo', units='N/A')
    data['lon_2d'] = dict(data=f['mod/sfc/lon'][...], name='Longitude', units='degrees')
    data['lat_2d'] = dict(data=f['mod/sfc/lat'][...], name='Latitude' , units='degrees')
    f.close()
    #"""

    #mod09     = pre_sfc(sat, tag, scale=sfc_scale, replace=sfc_replace)
    mod09     = sat_tmp(data)
    fname_sfc = '%s/sfc.pk' % fdir
    sfc0      = sfc_sat(sat_obj=mod09, fname=fname_sfc, extent=sat.extent, verbose=True, overwrite=overwrite)
    sfc_2d    = mca_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/mca_sfc_2d.bin' % fdir, overwrite=overwrite)
    # =================================================================================


    # mca_sca object (newly added for phase function)
    # =================================================================================
    pha0 = pha_mie(wvl0=wavelength)
    sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)
    # =================================================================================


    # cld object
    # =================================================================================
    data = {}
    f = h5py.File(f'{sat.fdir_out}/pre-data.h5', 'r')
    data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=f['mod/rad/lon'][...])
    data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=f['mod/rad/lat'][...])
    data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=f[f'mod/cld/cot_{solver}'][...])
    data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=f[f'mod/cld/cer_{solver}'][...])
    data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=f[f'mod/cld/cth_{solver}'][...])
    f.close()

    
    #modl1b    = pre_cld(sat, tag, cth=cth, scale_factor=scale_factor, solver=solver, ref_threshold=ref_threshold)
    modl1b    =  sat_tmp(data)
    fname_cld = '%s/cld.pk' % fdir

    if cth is None:
        cth0 = modl1b.data['cth_2d']['data']
    else:
        cth0 = cth
    cld0      = cld_sat(sat_obj=modl1b, fname=fname_cld, cth=cth0, cgt=0.5, dz=np.unique(atm0.lay['thickness']['data'])[0], overwrite=overwrite)
    # =================================================================================


    # mca_cld object
    # =================================================================================
    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir) # newly modified for phase function
    atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
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
            Nrun=5,
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

def cdata_all(date, tag, fdir_mca, fname_idl, sat):

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
    f = readsav(fname_idl)
    wvls  = f.lamx * 1000.0
    wvls  = np.sort(wvls)
    trans = f.tx[np.argsort(f.lamx)]

    modl1b = modis_l1b(fnames=sat.fnames['mod_02'], extent=sat.extent)
    lon_2d, lat_2d, rad_2d_mod = grid_modis_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['rad']['data'][0, ...], extent=sat.extent)

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
    ref_threshold = 0.06

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

    # create tmp-data/australia_YYYYMMDD directory if it does not exist
    # ===============================================================
    fdir_tmp = os.path.abspath('tmp-data/%s/%s' % (name_tag, band_tag))
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)
    # ===============================================================

    # read out wavelength information from Sebastian's absorption file
    # ===============================================================
    fname_idl = 'data/atm_abs_%s_11.out' % band_tag
    f = readsav(fname_idl)
    wvls = f.lamx*1000.0
    print(wvls)
    # ===============================================================
    #"""
    for solver in ['IPA', '3D']:
        preprocessing(band_tag, sat0, cth=None, scale_factor=1.0, 
                      sfc_scale=True, sfc_replace=True, solver=solver, ref_threshold=0.1, overwrite=True)
    #"""
    #sys.exit()
    # run calculations for each wavelength
    # ===============================================================
    for wavelength in wvls:
        for solver in ['IPA', '3D']:
            cal_mca_rad_oco2(date, band_tag, sat0, wavelength, fname_idl=fname_idl, cth=None, scale_factor=1.0, sfc_scale=True, sfc_replace=True, fdir=fdir_tmp, solver=solver, overwrite=True, photons=1e8, ref_threshold=ref_threshold)
    # ===============================================================

    # post-processing - combine the all the calculations into one dataset
    # ===============================================================
    return cdata_all(date, band_tag, fdir_tmp, fname_idl, sat0)
    # ===============================================================


def grab_cfg(path):
    """
    Read the setting information in the assigned csv file.

    path: relative or absolute path to the setting csv file.
    """
    cfg_file = pd.read_csv(path, header=None, index_col=0)
    result = {'cfg_name':path.split('/')[-1].replace('.csv', '')}
    for ind in cfg_file.index.dropna():
        contents = [str(i) for i in cfg_file.loc[ind].dropna() if str(i)[0] != '#']
        if len(contents) == 1:
            result[ind] = contents[0]
        elif len(contents) > 1:
            result[ind] = contents
    

    # check whether julian day exists and is correct
    date   = datetime.datetime(int(result['date'][:4]),    # year
                               int(result['date'][4:6]),   # month
                               int(result['date'][6:])     # day
                              )
    if 'juld' in result.keys():
        if not result['juld'] == date.timetuple().tm_yday:
            result['juld'] = date.timetuple().tm_yday
            save_h5_info(path, 'juld', date.timetuple().tm_yday)
    else: 
        save_h5_info(path, 'juld', date.timetuple().tm_yday)
        result['juld'] = date.timetuple().tm_yday 

    return result

def save_h5_info(cfg, index, filename):
    """
    Save the output h5 name into cfg file
    """
    cfg_file = pd.read_csv(cfg, header=None, index_col=0)
    cfg_file.loc[index, 1] = filename
    cfg_file.to_csv(cfg, header=False)
    return None

def check_h5_info(cfg, index):
    """
    Check whether the output h5 name is saved in cfg file
    """
    try: 
        cfg_file = grab_cfg(cfg)
    except OSError as err:
        print('{} not exists!'.format(cfg))
        return None
    if index in cfg_file.keys():
        if cfg_file[index][-2:] == 'h5':
            print('Output file {} exists.'.format(cfg_file[index]))
            return True
        return False
    else:
        return False
    
def save_subdomain_info(cfg, subdomain):
    """
    Save the subdomain info into cfg file
    """
    cfg_file = pd.read_csv(cfg, header=None, index_col=0)
    for j in range(4):
        cfg_file.loc['subdomain', j+1] = subdomain[j]
    cfg_file.to_csv(cfg, header=False)
    return None

def run_simulation(cfg):
    cfg_info = grab_cfg(cfg)

    #"""
    if not check_h5_info(cfg, 'o2'):
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
    

    #cfg = 'cfg/20181018_2.csv'
    #cfg = 'cfg/20150622_163_225.csv'
    #cfg = 'cfg/20150622_107_168.csv'
    #cfg = 'cfg/20170718_amazon_2.csv'
    cfg = 'cfg/20170605_amazon.csv'
    print(cfg)
    run_simulation(cfg)


    



