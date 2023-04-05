

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

import timeit
import argparse
import matplotlib.image as mpl_img


class func_cot_vs_rad:

    def __init__(self,
                sat,
                modl1b,
                fdir,
                wavelength,
                sfc_albedo=0,
                cth=3, 
                ctt=250, 
                cot=np.concatenate((np.arange(0.0, 1.0, 0.1),
                                    np.arange(1.0, 10.0, 1.0),
                                    np.arange(10.0, 20.0, 2.0),
                                    np.arange(20.0, 50.0, 5.0),
                                    np.arange(50.0, 100.0, 10.0),
                                    np.arange(100.0, 200.0, 20.0),
                                    np.arange(200.0, 401.0, 50.0))),
                run=False,
                ):

        if not os.path.exists(fdir):
            os.makedirs(fdir)

        self.fdir       = fdir
        self.wavelength = wavelength
        self.cot        = cot
        self.rad        = np.array([])

        if run:
            self.run_all(sat, modl1b, sfc_albedo, cth, ctt)

        for i in range(self.cot.size):
            cot0 = self.cot[i]
            fname = '%s/mca-out-rad-3d_cot-%.2f.h5' % (self.fdir, cot0)
            out0  = mca_out_ng(fname=fname, mode='all', squeeze=True)
            self.rad = np.append(self.rad, out0.data['rad']['data'].mean())

    def run_all(self, sat, modl1b, sfc_albedo, cth, ctt):
        f = h5py.File(sat.fnames['oco_l1b'][0], 'r')
        lon_oco_l1b = f['SoundingGeometry/sounding_longitude'][...]
        lat_oco_l1b = f['SoundingGeometry/sounding_latitude'][...]
        logic = (lon_oco_l1b>=sat.extent[0]) & (lon_oco_l1b<=sat.extent[1]) & (lat_oco_l1b>=sat.extent[2]) & (lat_oco_l1b<=sat.extent[3])
        sza = f['SoundingGeometry/sounding_solar_zenith'][...][logic].mean()
        saa = f['SoundingGeometry/sounding_solar_azimuth'][...][logic].mean()
        vza = f['SoundingGeometry/sounding_zenith'][...][logic].mean()
        vaa = f['SoundingGeometry/sounding_azimuth'][...][logic].mean()
        f.close()

        for cot0 in self.cot:
            print(cot0)
            self.run_mca_one(modl1b, cot0, sza, saa, vza, vaa, sfc_albedo, cth, ctt)

    def run_mca_one(self, modl1b, cot, sza, saa, vza, vaa, sfc_albedo, cth, ctt):

        

        """
        levels    = np.linspace(0.0, 20.0, 21)
        fname_atm = '%s/atm.pk' % self.fdir
        atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=False)

        fname_abs = '%s/abs.pk' % self.fdir
        abs0      = abs_16g(wavelength=self.wavelength, fname=fname_abs, atm_obj=atm0, overwrite=False)
        """

        # atm object
        # =================================================================================
        levels    = np.array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. , 5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. , 11. , 12. , 13. , 14. , 20. , 25. , 30. , 35. , 40. ])
        fname_atm = '%s/atm.pk' % self.fdir
        atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=True)
        # =================================================================================

        # abs object, in the future, we will implement OCO2 MET file for this
        # =================================================================================
        fname_abs = '%s/abs.pk' % self.fdir
        abs0      = abs_16g(wavelength=650, fname=fname_abs, atm_obj=atm0, overwrite=True)
        # =================================================================================

        cot_2d    = np.zeros((2, 2), dtype=np.float64); cot_2d[...] = cot
        cer_2d    = np.zeros((2, 2), dtype=np.float64); cer_2d[...] = 12.0
        ext_3d    = np.zeros((2, 2, 2), dtype=np.float64)

        fname_cld  = '%s/cld.pk' % self.fdir
        #cld0          = cld_les(fname_nc=fname_les, fname=fname_les_pk, coarsen=[1, 1, 1, 1], overwrite=False)
        cld0      = cld_sat(sat_obj=modl1b, fname=fname_cld, cth=modl1b.data['cth_2d']['data'], cgt=0.5, dz=np.unique(atm0.lay['thickness']['data'])[0], overwrite=False)

        cld0.lev['altitude']['data']    = cld0.lay['altitude']['data'][2:5]

        cld0.lay['x']['data']           = np.array([0, 1])
        cld0.lay['y']['data']           = np.array([0, 1])
        cld0.lay['nx']['data']          = 2
        cld0.lay['ny']['data']          = 2
        cth_hist, cth_edges = np.histogram(cth, density=False, bins=25)
        cth_index = np.argmax(cth_hist)
        cld0.lay['altitude']['data']    = cth_edges[cth_index:cth_index+2]
        #cld0.lay['pressure']['data']    = cld0.lay['pressure']['data'][2:4]
        ctt_hist, ctt_edges = np.histogram(ctt, density=False, bins=25)
        ctt_index = np.argmax(ctt_hist)
        cld0.lay['temperature']['data'] = np.zeros((2, 2, 2))
        cld0.lay['temperature']['data'][:,:, 0] = ctt_edges[ctt_index]
        cld0.lay['temperature']['data'][:,:, 1] = ctt_edges[ctt_index+1]
        cld0.lay['cot']['data']         = cot_2d
        cld0.lay['thickness']['data']   = cld0.lay['thickness']['data'][2:4]

        ext_3d[:, :, 0]  = cal_ext(cot_2d, cer_2d)/(cld0.lay['thickness']['data'].sum()*1000.0)
        ext_3d[:, :, 1]  = cal_ext(cot_2d, cer_2d)/(cld0.lay['thickness']['data'].sum()*1000.0)
        cld0.lay['extinction']['data']  = ext_3d

        atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)

        fname_atm3d = '%s/mca_atm_3d.bin' % self.fdir
        atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % self.fdir, overwrite=True)

        atm_1ds   = [atm1d0]
        atm_3ds   = [atm3d0]

        mca0 = mcarats_ng(
                date=datetime.datetime(2016, 8, 29),
                atm_1ds=atm_1ds,
                atm_3ds=atm_3ds,
                Ng=abs0.Ng,
                target='radiance',
                surface_albedo=sfc_albedo,
                solar_zenith_angle=sza,
                solar_azimuth_angle=-saa,
                sensor_zenith_angle=vza,
                sensor_azimuth_angle=vaa,
                fdir='%s/%.2f/les_rad_3d' % (self.fdir, cot),
                Nrun=1,
                photons=1e6,
                solver='3D',
                Ncpu=24,
                mp_mode='py',
                overwrite=True)

        out0 = mca_out_ng(fname='%s/mca-out-rad-3d_cot-%.2f.h5' % (self.fdir, cot), mca_obj=mca0, abs_obj=abs0, mode='all', squeeze=True, verbose=True)

    def interp_from_rad(self, rad, method='cubic'):

        f = interp1d(self.rad, self.cot, kind=method, bounds_error=False)

        return f(rad)

    def interp_from_cot(self, cot, method='cubic'):

        f = interp1d(self.cot, self.rad, kind=method, bounds_error=False)

        return f(cot)


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
        self.fnames['mod_02_hkm'] = []
        self.fnames['mod_02'] = []
        self.fnames['mod_03'] = []
        self.fnames['mod_04'] = []

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
            fnames_02_hkm = download_modis_https(self.date, '61/MYD02HKM', filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            fnames_02 = download_modis_https(self.date, '61/MYD02QKM', filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            fnames_03 = download_modis_https(self.date, '61/MYD03'   , filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            fnames_04 = download_modis_https(self.date, '61/MYD04_L2'   , filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            self.fnames['mod_l2'] += fnames_l2
            self.fnames['mod_02_1km'] += fnames_02_1km
            self.fnames['mod_02_hkm'] += fnames_02_hkm
            self.fnames['mod_02'] += fnames_02
            self.fnames['mod_03'] += fnames_03
            self.fnames['mod_04'] += fnames_04

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
        self.fnames['oco_co2prior'] = []
        self.fnames['oco_imap'] = []
        oco_fdir = 'data/oco'
        if not os.path.exists(oco_fdir):
            os.makedirs(oco_fdir)
        for filename_tag in filename_tags_03:
            dtime = datetime.datetime.strptime(filename_tag, 'A%Y%j.%H%M') + datetime.timedelta(minutes=7.0)
            fnames_std = download_oco2_https(dtime, 'OCO2_L2_Standard.10r', fdir_out=oco_fdir, run=run)
            fnames_met = download_oco2_https(dtime, 'OCO2_L2_Met.10r'     , fdir_out=oco_fdir, run=run)
            fnames_l1b = download_oco2_https(dtime, 'OCO2_L1B_Science.10r', fdir_out=oco_fdir, run=run)
            fnames_co2prior = download_oco2_https(dtime, 'OCO2_L2_CO2Prior.10r', fdir_out=oco_fdir, run=run)
            fnames_imap = download_oco2_https(dtime, 'OCO2_L2_IMAPDOAS.10r', fdir_out=oco_fdir, run=run)
            self.fnames['oco_std'] += fnames_std
            self.fnames['oco_met'] += fnames_met
            self.fnames['oco_l1b'] += fnames_l1b
            self.fnames['oco_co2prior'] += fnames_co2prior
            self.fnames['oco_imap'] += fnames_imap

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

def pre_cld(sat, tag, cth=None, cot_source='2stream', ref_threshold=0.1, scale_factor=1.0, solver='3D', fdir_cot='tmp-data'):

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
                              vnames=['cloud_top_height_1km', 'cloud_top_temperature_1km',])
    lon_1km       = modl2_1km.data['lon']['data']
    lat_1km       = modl2_1km.data['lat']['data']
    cth_1km       = modl2_1km.data['cloud_top_height_1km']['data']/1000.0 
    ctt_1km       = modl2_1km.data['cloud_top_temperature_1km']['data']
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
    lon_2d, lat_2d, ref_2d = grid_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['ref']['data'][0, ...], extent=sat.extent)
    lon_2d, lat_2d, rad_2d = grid_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['rad']['data'][0, ...], extent=sat.extent)
    print(f'Load modis l1b data: {(time.time()-now)/60} min')

    modl1b_500m = modis_l1b(fnames=sat.fnames['mod_02_hkm'], extent=sat.extent)
    lon_2d_470, lat_2d_470, ref_2d_470_raw = grid_by_extent(modl1b_500m.data['lon']['data'], modl1b_500m.data['lat']['data'], modl1b_500m.data['ref']['data'][0, ...], extent=sat.extent)
    lon_2d_550, lat_2d_550, ref_2d_550_raw = grid_by_extent(modl1b_500m.data['lon']['data'], modl1b_500m.data['lat']['data'], modl1b_500m.data['ref']['data'][1, ...], extent=sat.extent)
    mask = ref_2d_470_raw>=0
    points_mask = np.transpose(np.vstack((lon_2d_470[mask].flatten(), lat_2d_470[mask].flatten())))
    ref_2d_470 = interpolate.griddata(points_mask, ref_2d_470_raw[mask].flatten(), 
                                           (lon_2d, lat_2d), method='nearest')
    
    mask = ref_2d_470_raw>=0
    points_mask = np.transpose(np.vstack((lon_2d_470[mask].flatten(), lat_2d_470[mask].flatten())))
    ref_2d_470 = interpolate.griddata(points_mask, ref_2d_470_raw[mask].flatten(), 
                                           (lon_2d, lat_2d), method='nearest')

    """    
    #print('wavelength: {}nm'.format(modl1b.data['wvl']['data'][0]))
    _, _, ref_650_2d = grid_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['ref']['data'][0, ...], extent=sat.extent)#, NxNy=(110*2, 148*2))
    _, _, rad_650_2d = grid_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['rad']['data'][0, ...], extent=sat.extent)#, NxNy=(110*2, 148*2))
    #print('wavelength: {}nm'.format(modl1b.data['wvl']['data'][1]))
    _, _, ref_860_2d = grid_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['ref']['data'][1, ...], extent=sat.extent)#, NxNy=(110*2, 148*2))
    _, _, rad_860_2d = grid_by_extent(modl1b.data['lon']['data'], modl1b.data['lat']['data'], modl1b.data['rad']['data'][1, ...], extent=sat.extent)#, NxNy=(110*2, 148*2))
    """    
    # ===================================================================================

    # Process mcd04 AOD data
    # =========================================================================
    mcd04 = modis_04(fnames=sat.fnames['mod_04'], extent=sat.extent, 
                     vnames=['Deep_Blue_Spectral_Single_Scattering_Albedo_Land', ])
    AOD_lon, AOD_lat, AOD_550_land = grid_by_extent(mcd04.data['lon']['data'], mcd04.data['lat']['data'], mcd04.data['AOD_550_land']['data'], extent=sat.extent)
    _, _, Angstrom_Exponent_land = grid_by_extent(mcd04.data['lon']['data'], mcd04.data['lat']['data'], mcd04.data['Angstrom_Exponent_land']['data'], extent=sat.extent)
    print(AOD_550_land.shape)
    print(mcd04.data['deep_blue_spectral_single_scattering_albedo_land']['data'].shape)
    _, _, SSA_land_660 = grid_by_extent(mcd04.data['lon']['data'], mcd04.data['lat']['data'], mcd04.data['deep_blue_spectral_single_scattering_albedo_land']['data'], extent=sat.extent)
    print( SSA_land_660.shape)
    
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


    # =========================================================================

    # Process MODIS radiance and reflectance at 650/860 nm (250m resolution)
    # ===================================================================================
    now = time.time()
    mod09 = modis_09a1(fnames=sat.fnames['mod_09'], extent=sat.extent)
    mod09_lon_2d, mod09_lat_2d, mod_sfc_alb_2d = grid_by_extent(mod09.data['lon']['data'], mod09.data['lat']['data'], mod09.data['ref']['data'][0, :], extent=sat.extent)
    
    mcd43 = modis_43a3(fnames=sat.fnames['mcd_43'], extent=sat.extent)
    points = np.transpose(np.vstack((mcd43.data['lon']['data'], mcd43.data['lat']['data'])))
    mod43_lon_2d, mod43_lat_2d, mcd_bsa_2d_650 = grid_by_extent(mcd43.data['lon']['data'], mcd43.data['lat']['data'], mcd43.data['bsa']['data'][0, :], extent=sat.extent)
    _, _, mcd_bsa_2d_860 = grid_by_extent(mcd43.data['lon']['data'], mcd43.data['lat']['data'], mcd43.data['bsa']['data'][1, :], extent=sat.extent)
    _, _, mcd_bsa_2d_470 = grid_by_extent(mcd43.data['lon']['data'], mcd43.data['lat']['data'], mcd43.data['bsa']['data'][2, :], extent=sat.extent)
    _, _, mcd_bsa_2d_550 = grid_by_extent(mcd43.data['lon']['data'], mcd43.data['lat']['data'], mcd43.data['bsa']['data'][3, :], extent=sat.extent)
       
    mcd_bsa_2d_650[mcd_bsa_2d_650<0] = np.nan
    mcd_bsa_2d_860[mcd_bsa_2d_860<0] = np.nan
    mcd_bsa_2d_470[mcd_bsa_2d_470<0] = np.nan
    mcd_bsa_2d_550[mcd_bsa_2d_550<0] = np.nan
            
    mask_650 = mcd_bsa_2d_650>=0
    points_mask = np.transpose(np.vstack((mod43_lon_2d[mask_650].flatten(), mod43_lat_2d[mask_650].flatten())))
    mcd_bsa_inter_650 = interpolate.griddata(points_mask, mcd_bsa_2d_650[mask_650].flatten(), 
                                           (lon_2d, lat_2d), method='nearest')

    mask_860 = mcd_bsa_2d_860>=0
    points_mask = np.transpose(np.vstack((mod43_lon_2d[mask_860].flatten(), mod43_lat_2d[mask_860].flatten())))
    mcd_bsa_inter_860 = interpolate.griddata(points_mask, mcd_bsa_2d_860[mask_860].flatten(), 
                                           (lon_2d, lat_2d), method='nearest')
    
    mask_470 = mcd_bsa_2d_470>=0
    points_mask = np.transpose(np.vstack((mod43_lon_2d[mask_470].flatten(), mod43_lat_2d[mask_470].flatten())))
    mcd_bsa_inter_470 = interpolate.griddata(points_mask, mcd_bsa_2d_470[mask_470].flatten(), 
                                           (lon_2d, lat_2d), method='nearest')
    
    mask_550 = mcd_bsa_2d_550>=0
    points_mask = np.transpose(np.vstack((mod43_lon_2d[mask_550].flatten(), mod43_lat_2d[mask_550].flatten())))
    mcd_bsa_inter_550 = interpolate.griddata(points_mask, mcd_bsa_2d_550[mask_550].flatten(), 
                                           (lon_2d, lat_2d), method='nearest')
    print(f'Load modis 09 and 43 data: {(time.time()-now)/60} min')

    now = time.time()
    sfc_alb_inter = mcd_bsa_inter_650
    sza_lon, sza_lat, sza_2d = grid_by_extent(lon1, lat1, sza1, extent=sat.extent)
    points = np.transpose(np.vstack((sza_lon.flatten(), sza_lat.flatten())))
    sza_inter = interpolate.griddata(points, sza_2d.flatten(), 
                                           (lon_2d, lat_2d), method='linear')
    print(f'Interpolating sfc alb and sza: {(time.time()-now)/60} min')

    # ===================================================================================

    #----------------------------------------------
    mod_rgb = mpl_img.imread(sat.fnames['mod_rgb'][0])
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
    logic_ref_nan = (ref_2d_470-mcd_bsa_inter_470) < ref_threshold
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

def pre_sfc(sat, tag, version='10r', scale=True, replace=True):

    if version == '10' or version == '10r':
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

    if version == '10' or version == '10r':

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
    #lon_2d, lat_2d, mcd_bsa_2d = grid_by_extent(mod.data['lon']['data'], mod.data['lat']['data'], mod.data['bsa']['data'][index[wvl], :], extent=sat.extent)
    #lon_2d, lat_2d, mcd_wsa_2d = grid_by_extent(mod.data['lon']['data'], mod.data['lat']['data'], mod.data['bsa']['data'][index[wvl], :], extent=sat.extent)
    mod43_mode = 'wsa'
    points = np.transpose(np.vstack((mod.data['lon']['data'], mod.data['lat']['data'])))
    if tag.lower() == 'o2a':
        lon_2d, lat_2d, mod_sfc_alb_2d = grid_by_extent(mod.data['lon']['data'], mod.data['lat']['data'], mod.data[mod43_mode]['data'][1, :], extent=sat.extent)
    elif tag.lower() == 'wco2':
        lon_2d, lat_2d, mod_sfc_alb_2d = grid_by_extent(mod.data['lon']['data'], mod.data['lat']['data'], mod.data[mod43_mode]['data'][5, :], extent=sat.extent)
    elif tag.lower() == 'sco2':
        lon_2d, lat_2d, mod_sfc_alb_2d = grid_by_extent(mod.data['lon']['data'], mod.data['lat']['data'], mod.data[mod43_mode]['data'][6, :], extent=sat.extent)

    mask = mod_sfc_alb_2d>=0
    points = np.transpose(np.vstack((lon_2d[mask].flatten(), lat_2d[mask].flatten())))
    mod_sfc_alb_2d_inter = interpolate.griddata(points, mod_sfc_alb_2d[mask].flatten(), 
                                           (lon_2d, lat_2d), method='nearest')

    mod03      = modis_03(fnames=sat.fnames['mod_03'], extent=sat.extent, vnames=['Height'])
    lon1       = mod03.data['lon']['data']
    lat1       = mod03.data['lat']['data']
    sza1       = mod03.data['sza']['data']

    sza_lon, sza_lat, sza_2d = grid_by_extent(lon1, lat1, sza1, extent=sat.extent)
    mask = sza_2d>0
    points = np.transpose(np.vstack((sza_lon[mask].flatten(), sza_lat[mask].flatten())))
    print('sza_ori filtered albedo NaN:', np.isnan(sza_2d[mask]).any())
    sza_inter = interpolate.griddata(points, sza_2d[mask].flatten(), 
                                           (lon_2d, lat_2d), method='nearest')
    print('sza albedo NaN:', np.isnan(sza_inter).any())
    print('sfc albedo NaN:', np.isnan(mod_sfc_alb_2d_inter).any())
    mod_sfc_alb_2d = mod_sfc_alb_2d_inter#/np.cos(sza_inter/180*np.pi)

    oco_sfc_alb_2d = create_sfc_alb_2d(oco_lon, oco_lat, oco_sfc_alb, lon_2d, lat_2d, mod_sfc_alb_2d, scale=scale, replace=replace)

    mod.data['alb_2d'] = dict(data=oco_sfc_alb_2d, name='Surface albedo', units='N/A')
    mod.data['mod43_2d'] = dict(data=mod_sfc_alb_2d, name='Black-sky albedo', units='N/A')
    mod.data['lon_2d'] = dict(data=lon_2d, name='Longitude', units='degrees')
    mod.data['lat_2d'] = dict(data=lat_2d, name='Latitude' , units='degrees')

    return mod



class sat_tmp:

    def __init__(self, data):

        self.data = data





if __name__ == '__main__':
    None


    



