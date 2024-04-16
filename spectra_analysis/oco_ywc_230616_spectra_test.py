#!/bin/env python
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=OCO2_test

import os
from pathlib import Path
import sys
import h5py
import numpy as np
import pandas as pd
import datetime
from scipy import stats as st
from scipy.special import wofz
from scipy.optimize import curve_fit
import xarray as xr
import pickle
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from er3t.util.modis import download_modis_https, get_sinusoidal_grid_tag, get_filename_tag, download_modis_rgb
from er3t.util.daac import download_oco2_https
from er3t.util import upscale_modis_lonlat

from oco_subroutine.oco_cfg import grab_cfg, save_h5_info


import geopy.distance
from haversine import haversine, Unit, haversine_vector

import timeit

class sat_tmp:

    def __init__(self, data):

        self.data = data

class satellite_download:
    """
    This class is used to download satellite data from MODIS and OCO-2
    """
    def __init__(self,
                 date=None,
                 extent=None,
                 fname=None,
                 fdir_out='data',
                 fdir_pre_data='data',
                 overwrite=False,
                 quiet=False,
                 verbose=False):
        """
        Initialize the SatelliteDownload class.
        """
        self.date     = date
        self.extent   = extent
        self.extent_simulation = extent
        self.fdir_out = fdir_out
        self.quiet    = quiet
        self.verbose  = verbose

        if (fname is not None) and os.path.exists(fname) and (not overwrite):
            self.load(fname)
        elif ((date is not None) and (extent is not None) and (fname is not None) and os.path.exists(fname) and overwrite) or \
             ((date is not None) and (extent is not None) and (fname is not None) and not os.path.exists(fname)):
            self.run()
            self.dump(fname)

        elif date is not None and extent is not None and fname is None:
            self.run()
        else:
            raise FileNotFoundError('Error   [satellite_download]: Please check if \'%s\' exists or provide \'date\' and \'extent\' to proceed.' % fname)

    def load(self, fname):
        """
        Load the satellite data from the pickle file.
        """
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
        """
        Run the satellite download process.
        """
        lon = np.array(self.extent[:2])
        lat = np.array(self.extent[2:])

        self.fnames = {}

        self.fnames['mod_rgb'] = [download_modis_rgb(self.date, self.extent, fdir=self.fdir_out, which='aqua', coastline=True)]

        # MODIS Level 2 Cloud Product and MODIS 03 geo file
        self.fnames['mod_l2'] = []
        # self.fnames['mod_02_1km'] = []
        # self.fnames['mod_02_hkm'] = []
        # self.fnames['mod_02'] = []


        # modis_fdir = '../simulation/data/modis'
        modis_parent_fdir = '/'.join([self.fdir_out, 'modis'])
        if not os.path.exists(modis_parent_fdir):
            os.makedirs(modis_parent_fdir)
        modis_fdir = f"{modis_parent_fdir}/{self.date.strftime('%Y%m%d')}"
        if not os.path.exists(modis_fdir):
            os.makedirs(modis_fdir)

        filename_tags_03 = get_filename_tag(self.date, lon, lat, satID='aqua')
        for filename_tag in filename_tags_03:
            fnames_l2 = download_modis_https(self.date, '61/MYD06_L2', filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            # fnames_02_1km = download_modis_https(self.date, '61/MYD021KM', filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            # fnames_02_hkm = download_modis_https(self.date, '61/MYD02HKM', filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            # fnames_02 = download_modis_https(self.date, '61/MYD02QKM', filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            self.fnames['mod_l2'] += fnames_l2
            # self.fnames['mod_02_1km'] += fnames_02_1km
            # self.fnames['mod_02_hkm'] += fnames_02_hkm
            # self.fnames['mod_02'] += fnames_02

        # OCO2 std and met file
        self.fnames['oco_std'] = []
        self.fnames['oco_met'] = []
        self.fnames['oco_l1b'] = []
        self.fnames['oco_lite'] = []
        # oco_fdir = '../simulation/data/oco'
        oco_fdir = '/'.join([self.fdir_out, 'oco'])
        for filename_tag in filename_tags_03:
            dtime = datetime.datetime.strptime(filename_tag, 'A%Y%j.%H%M') + datetime.timedelta()#minutes=7.0)
            fnames_std = download_oco2_https(dtime, 'OCO2_L2_Standard.10r', fdir_out=oco_fdir, run=run)
            fnames_met = download_oco2_https(dtime, 'OCO2_L2_Met.10r'     , fdir_out=oco_fdir, run=run)
            fnames_l1b = download_oco2_https(dtime, 'OCO2_L1B_Science.10r', fdir_out=oco_fdir, run=run)
            fnames_lt = download_oco2_https(dtime, 'OCO2_L2_Lite_FP.10r', fdir_out=oco_fdir, run=run)
            self.fnames['oco_std'] += fnames_std
            self.fnames['oco_met'] += fnames_met
            self.fnames['oco_l1b'] += fnames_l1b
            self.fnames['oco_lite'] += fnames_lt

    def dump(self, fname):
        """
        Save the SatelliteDownload object into a pickle file
        """
        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [satellite_download]: Saving object into %s ...' % fname)
            pickle.dump(self, f)

def solar(file):
    """
    # to solve: kurudz.dat and val?
    """
    h = 6.62607004e-34
    c = 299792458.
    hc = h*c


    # the setting is specific to the assigned solar.txt filw
    data = pd.read_csv(file, skiprows=6, header=None, sep='     ', engine='python')
    data['Wavenumber'] = data[0].astype(float)
    data['Irradiance'] = data[1].astype(float)
    wn = np.array(data['Wavenumber'])
    ss = np.array(data['Irradiance']) # photons/sec/m2/micron
    sx = 1e-1*hc*wn*ss # W/m2/nm
    wl = 1.0e4/wn # convert wavenumer to wavelength in micron


    return wl[::-1], sx[::-1]

def add_solar(s_wl, s_fx, lam, ilsx, ilsy, fp=0):

    wl = s_wl
    fx = s_fx
    wl=np.array(wl)
    fx=np.array(fx)
    idx=np.argsort(wl)
    solar_wl=np.array(wl[idx])
    solar_irradiance=np.array(fx[idx])

    ch1 = np.interp(lam[fp,:, 0],solar_wl,solar_irradiance)
    ch2 = np.interp(lam[fp,:, 1],solar_wl,solar_irradiance)        
    ch3 = np.interp(lam[fp,:, 2],solar_wl,solar_irradiance)        

    ox=np.zeros((8, 1016))
    wc=np.zeros((8, 1016))
    sc=np.zeros((8, 1016))
    for fp in range(8):
        for i in range(1016):
            temp=np.interp(lam[fp,:, 0], ilsx[0,fp,i,:]+lam[fp, i, 0], ilsy[0,fp,i,:])
            temp=temp/np.sum(temp)
            ox[fp, i]=np.sum(ch1*temp)
            temp=np.interp(lam[fp,:, 1], ilsx[1,fp,i,:]+lam[fp, i, 1], ilsy[1,fp,i,:])
            temp=temp/np.sum(temp)
            wc[fp, i]=np.sum(ch2*temp)
            temp=np.interp(lam[fp,:, 2], ilsx[2,fp,i,:]+lam[fp, i, 2], ilsy[2,fp,i,:])
            temp=temp/np.sum(temp)
            sc[fp, i]=np.sum(ch3*temp)

    """nz =1#self.rad1.shape[0]
    nfp=1#self.rad1.shape[1]
    #mu = np.stack([self.mu]*1016,axis=2)
    oxs= np.stack([ox]*nfp,axis=0)
    oxs= np.stack([oxs]*nz,axis=0)
    wcs= np.stack([wc]*nfp,axis=0)
    wcs= np.stack([wcs]*nz,axis=0)
    scs= np.stack([sc]*nfp,axis=0)
    scs= np.stack([scs]*nz,axis=0)
    
    return oxs, wcs, scs"""
    
    return ox, wc, sc


def cld_dist_calc(lon_oco, lat_oco, cth, cth_lon, cth_lat, cfg_info, fdir_data):

    if os.path.isfile(f"{fdir_data}/dist_data_modis_cth_oco_cld_distance_nearest_{cfg_info['cfg_name']}.npy"):
        cloud_dist = np.load(f"{fdir_data}/dist_data_modis_cth_oco_cld_distance_nearest_{cfg_info['cfg_name']}.npy")
    else:

        lon_cld, lat_cld = cth_lon, cth_lat
        cld_list = cth>0
        cld_X, cld_Y = np.where(cld_list==1)[0], np.where(cld_list==1)[1]
        cld_position = []
        cld_position_lonlat = []
        for i in range(len(cld_X)):
            cld_position.append(np.array([cld_X[i], cld_Y[i]]))
            cld_position_lonlat.append(np.array([cth_lon[cld_X[i], cld_Y[i]], cth_lat[cld_X[i], cld_Y[i]]]))
        cld_position = np.array(cld_position)
        cld_position_lonlat = np.array(cld_position_lonlat)

        cloud_dist = np.zeros_like(lon_oco)
    #     for j in range(cloud_dist.shape[1]):
    #         for i in range(cloud_dist.shape[0]):
    # #             if cld_list[i, j] == 1:
    # #                 cloud_dist[i, j] = 0
    # #             else:
    #             tmp_lon = lon_oco[i, j]
    #             tmp_lat = lat_oco[i, j]
    #             if np.logical_and(np.logical_and(tmp_lon >= np.min(cth_lon), tmp_lon <= np.max(cth_lon)), 
    #                               np.logical_and(tmp_lat >= np.min(cth_lat), tmp_lat <= np.max(cth_lat))):
    #                 min_ind = np.argmin(np.sum(np.square(cld_position_lonlat-np.array([tmp_lon, tmp_lat])), axis=1))
    #                 #print(min_ind)
    #                 #print(cld_position[min_ind])
    #                 cld_x, cld_y = cld_position_lonlat[min_ind][0], cld_position_lonlat[min_ind][1]

    #                 dist = geopy.distance.distance((tmp_lat, tmp_lon), (cld_y, cld_x)).km
    #                 #print(dist)
    #                 cloud_dist[i, j] = dist
    #             else:
    #                 cloud_dist[i, j] = np.nan
        for i in range(cloud_dist.shape[0]):

                tmp_lon = lon_oco[i]
                tmp_lat = lat_oco[i]
                if np.logical_and(np.logical_and(tmp_lon >= np.min(cth_lon), tmp_lon <= np.max(cth_lon)), 
                                  np.logical_and(tmp_lat >= np.min(cth_lat), tmp_lat <= np.max(cth_lat))):
                    min_ind = np.argmin(np.sum(np.square(cld_position_lonlat-np.array([tmp_lon, tmp_lat])), axis=1))
                    #print(min_ind)
                    #print(cld_position[min_ind])
                    cld_x, cld_y = cld_position_lonlat[min_ind][0], cld_position_lonlat[min_ind][1]

                    dist = geopy.distance.distance((tmp_lat, tmp_lon), (cld_y, cld_x)).km
                    #print(dist)
                    cloud_dist[i] = dist
                else:
                    cloud_dist[i] = np.nan

        np.save(f"{fdir_data}/dist_data_modis_cth_oco_cld_distance_nearest_{cfg_info['cfg_name']}.npy", cloud_dist)
    
    return cloud_dist

def weighted_cld_dist_calc(lon_oco, lat_oco, cth, cth_lon, cth_lat, cfg_info, fdir_data):

    if os.path.isfile(f"{fdir_data}/dist_data_modis_cth_oco_cld_distance_weighted_{cfg_info['cfg_name']}.npy"):
        cloud_dist = np.load(f"{fdir_data}/dist_data_modis_cth_oco_cld_distance_weighted_{cfg_info['cfg_name']}.npy")
    else:
        lon_cld, lat_cld = cth_lon, cth_lat
        cld_list = cth>0
        cld_X, cld_Y = np.where(cld_list==1)[0], np.where(cld_list==1)[1]
        cld_position = []
        cld_position_latlon = []
        for i in range(len(cld_X)):
            cld_position.append(np.array([cld_X[i], cld_Y[i]]))
            cld_position_latlon.append(np.array([cth_lat[cld_X[i], cld_Y[i]], cth_lon[cld_X[i], cld_Y[i]]]))
        cld_position = np.array(cld_position)
        cld_position_latlon = np.array(cld_position_latlon)

        cloud_dist = np.zeros_like(lon_oco)


        # for j in range(cloud_dist.shape[1]):
        #     for i in range(cloud_dist.shape[0]):
        #         tmp_lon = lon_oco[i, j]
        #         tmp_lat = lat_oco[i, j]

        #         if np.logical_and(np.logical_and(tmp_lon >= np.min(cth_lon), tmp_lon <= np.max(cth_lon)), 
        #                           np.logical_and(tmp_lat >= np.min(cth_lat), tmp_lat <= np.max(cth_lat))):
                    
        #             point = np.array([tmp_lat, tmp_lon])

        #             # distances = np.array([haversine(point, p, unit=Unit.KILOMETERS) for p in cld_latlon])
        #             distances = haversine_vector(point, cld_position_latlon, unit=Unit.KILOMETERS, comb=True)
        #             # Calculate the inverse distance weights
                    
        #             weights = 1 / distances**2 #np.exp(-distances)
        #             weights[distances>100] = 0
        #             #weights = 1 / distances                
                    
        #             # Calculate the weighted average distance
        #             weighted_avg_distance = np.sum(distances * weights) / np.sum(weights)
                    
        #             cloud_dist[i, j] = weighted_avg_distance
        #         else:
        #             cloud_dist[i, j] = np.nan
        for i in range(cloud_dist.shape[0]):
                tmp_lon = lon_oco[i]
                tmp_lat = lat_oco[i]

                if np.logical_and(np.logical_and(tmp_lon >= np.min(cth_lon), tmp_lon <= np.max(cth_lon)), 
                                  np.logical_and(tmp_lat >= np.min(cth_lat), tmp_lat <= np.max(cth_lat))):
                    
                    point = np.array([tmp_lat, tmp_lon])

                    # distances = np.array([haversine(point, p, unit=Unit.KILOMETERS) for p in cld_latlon])
                    distances = haversine_vector(point, cld_position_latlon, unit=Unit.KILOMETERS, comb=True)
                    # Calculate the inverse distance weights
                    
                    weights = 1 / distances**2 #np.exp(-distances)
                    weights[distances>50] = 0
                    #weights = 1 / distances                
                    
                    # Calculate the weighted average distance
                    weighted_avg_distance = np.sum(distances * weights) / np.sum(weights)
                    
                    cloud_dist[i] = weighted_avg_distance
                else:
                    cloud_dist[i] = np.nan

        np.save(f"{fdir_data}/dist_data_modis_cth_oco_cld_distance_weighted_{cfg_info['cfg_name']}.npy", cloud_dist) 

    return cloud_dist



def convert_photon_unit(data_photon, wavelength, scale_factor=1):
    # original: 
    # Ph sec^{-1} m^{-2} sr^{-1} um^{-1}
    # wavelength: nm
    
    c = 299792458.0
    h = 6.62607015e-34
    wavelength = wavelength * 1e-9
    data = data_photon/1000.0*c*h/wavelength*scale_factor

    return data

def preprocess(cfg_info):
    # define date and region to study
    # ===============================================================
    date   = datetime.datetime(int(cfg_info['date'][:4]),    # year
                               int(cfg_info['date'][4:6]),   # month
                               int(cfg_info['date'][6:]))    # day
    extent = [float(loc) for loc in cfg_info['subdomain']]
    print(f'simulation extent: {extent}')
    ref_threshold = float(cfg_info['ref_threshold'])

    name_tag = f"{cfg_info['cfg_name']}_{date.strftime('%Y%m%d')}"
    # ===============================================================

    # create data/name_tag directory if it does not exist
    # ===============================================================
    fdir_data = os.path.abspath('data/%s' % name_tag)
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists(fdir_data):
        os.makedirs(fdir_data)
    # ===============================================================

    # download satellite data based on given date and region
    # ===============================================================
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(date=date, 
                              fdir_out=cfg_info['path_sat_data'], 
                              fdir_pre_data=fdir_data,
                              extent=extent,
                              fname=fname_sat, overwrite=False)
    # ===============================================================
    if not ('l2' in cfg_info.keys()):
        oco_data_dict = {'l2': 'oco_std',
                         'met': 'oco_met',
                         'l1b': 'oco_l1b',}
        for key, value in oco_data_dict.items():
            save_h5_info(cfg_info['cfg_path'], key, sat0.fnames[value][0].split('/')[-1])



def V0(x, center, alpha, gamma, adj):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))

    return (np.real(wofz(((x-center) + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi))/adj

def plt_map_cld_dis(sat, cth0, lon, lat, cloud_dist, snd_lon, snd_lat, fdir_data):
    png       = [sat.fnames['mod_rgb'][0], sat.extent]
    img = png[0]
    wesn= png[1]


    f,frame=plt.subplots(figsize=(12, 10))
    img = png[0]
    wesn= png[1]
    img = mpl.image.imread(img)
    frame.imshow(img,extent=wesn)
    lon_dom = [wesn[0], wesn[1]]
    lat_dom = [wesn[2], wesn[3]]
    frame.set_xlim(np.min(lon_dom), np.max(lon_dom))
    frame.set_ylim(np.min(lat_dom), np.max(lat_dom))
    mask = cth0[:]>=0
    c = frame.scatter(lon[mask], lat[mask], s=1,
                    marker='o', 
                    #c=cld_mask0_bin_cloud_int,
                    c=cth0[mask],
                    alpha=0.4, vmin=0, vmax=10)

    cbar = f.colorbar(c, extend='max')
    cbar.set_label('Cloud top height', fontsize=18)

    c2 = frame.scatter(snd_lon, snd_lat, s=5, c=cloud_dist, cmap='OrRd', 
                       vmin=0, vmax=50)
    cbar2 = f.colorbar(c2, extend='max')
    cbar2.set_label('Cloud distance', fontsize=18, )
    #frame.scatter(lon_cld[cld_list>0], lat_cld[cld_list>0], s=5, color='r')
    #for i in range(len(boundary_list)):
    #    boundary = boundary_list[i]
    #    plot_rec(np.mean(boundary[0][:2]), np.mean(boundary[0][2:]), 
    #             0.5, lat_interval, 
    #             frame, 'r')
    #plt.legend(fontsize=16, facecolor='white')
    frame.set_xlabel('Longitude')
    frame.set_ylabel('Latitude')
    f.tight_layout()
    f.savefig(f'{fdir_data}/modis_cth_oco_cld_distance_nearest.png')
    # plt.show()

def plt_map_weight_cld_dis(sat, cth0, lon, lat, weighted_cloud_dist, snd_lon, snd_lat, fdir_data):
    png       = [sat.fnames['mod_rgb'][0], sat.extent]
    img = png[0]
    wesn= png[1]


    f,frame=plt.subplots(figsize=(12, 10))
    img = png[0]
    wesn= png[1]
    img = mpl.image.imread(img)
    frame.imshow(img,extent=wesn)
    lon_dom = [wesn[0], wesn[1]]
    lat_dom = [wesn[2], wesn[3]]
    frame.set_xlim(np.min(lon_dom), np.max(lon_dom))
    frame.set_ylim(np.min(lat_dom), np.max(lat_dom))
    mask = cth0[:]>=0
    c = frame.scatter(lon[mask], lat[mask], s=1,
                    marker='o', 
                    #c=cld_mask0_bin_cloud_int,
                    c=cth0[mask],
                    alpha=0.4, vmin=0, vmax=10)

    cbar = f.colorbar(c, extend='max')
    cbar.set_label('Cloud top height', fontsize=18)

    c2 = frame.scatter(snd_lon, snd_lat, s=5, c=weighted_cloud_dist, cmap='OrRd', 
                       vmin=0, vmax=50)
    cbar2 = f.colorbar(c2, extend='max')
    cbar2.set_label('Weighted Cloud distance', fontsize=18, )
    #frame.scatter(lon_cld[cld_list>0], lat_cld[cld_list>0], s=5, color='r')
    #for i in range(len(boundary_list)):
    #    boundary = boundary_list[i]
    #    plot_rec(np.mean(boundary[0][:2]), np.mean(boundary[0][2:]), 
    #             0.5, lat_interval, 
    #             frame, 'r')
    #plt.legend(fontsize=16, facecolor='white')
    frame.set_xlabel('Longitude')
    frame.set_ylabel('Latitude')
    f.tight_layout()
    f.savefig(f'{fdir_data}/modis_cth_oco_cld_distance_weighted.png')
    # plt.show()

def hist_cld_dist(sat, cloud_dist, weighted_cloud_dist, fdir_data):
    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(cloud_dist[~np.isnan(cloud_dist)], bins=100, alpha=0.5, label='Nearest')
    ax.hist(weighted_cloud_dist[~np.isnan(weighted_cloud_dist)], bins=100, alpha=0.5, label='Weighted')
    ax.set_xlabel('Cloud distance (km)', fontsize=18)
    ax.set_ylabel('Frequency', fontsize=18)
    ax.legend(fontsize=18)
    f.tight_layout()
    f.savefig(f'{fdir_data}/modis_cth_oco_cld_distance_hist.png')

def run_case_modis_650(cfg_info):
    # define date and region to study
    # ===============================================================
    date   = datetime.datetime(int(cfg_info['date'][:4]),    # year
                               int(cfg_info['date'][4:6]),   # month
                               int(cfg_info['date'][6:]))    # day
    extent = [float(loc) + offset for loc, offset in zip(cfg_info['subdomain'], [-0.15, 0.15, -0.15, 0.15])]

    name_tag = f"{cfg_info['cfg_name']}_{date.strftime('%Y%m%d')}"
    # ======================================================================

    # create data/{name_tag} directory if it does not exist
    # ======================================================================
    fdir_data = os.path.abspath(f'data/{name_tag}')
    print(f'fdir_data: {fdir_data}')
    # sys.exit()
    fname_sat = f'{fdir_data}/sat.pk'
    sat0 = satellite_download(date=date, 
                              fdir_out=cfg_info['path_sat_data'], 
                              
                              fdir_pre_data=fdir_data,
                              extent=extent,
                              fname=fname_sat, overwrite=False)
    
    oco_l1b_file = sat0.fnames['oco_l1b'][0]#.replace('../simulation/data/', '../sat_data/')
    l1b = h5py.File(oco_l1b_file, 'r')
    oco_l2_file = sat0.fnames['oco_std'][0]#.replace('../simulation/data/', '../sat_data/')
    l2 = h5py.File(oco_l2_file, 'r')
    oco_met_file = sat0.fnames['oco_met'][0]#.replace('../simulation/data/', '../sat_data/')
    met = h5py.File(oco_met_file, 'r')

    dis = l1b["InstrumentHeader/dispersion_coef_samp"][...]
    lam = np.zeros([8,1016,3]) # Those are the wavelengths in the radiance file
    wli = np.arange(1,1017,dtype=float)
    for i in range(8): 
        for j in range(3):
            for k in range(5):
                lam[i,:,j]=lam[i,:,j] + dis[j,i,k]*wli**k  
    ilsx = l1b["InstrumentHeader/ils_delta_lambda"][...]
    ilsy = l1b["InstrumentHeader/ils_relative_response"][...]
    s_wl, s_fx = solar('solar.txt')
    oxs, wcs, scs = add_solar(s_wl, s_fx, lam, ilsx, ilsy)

    o2a_rad = l1b['SoundingMeasurements/radiance_o2'][...]
    wco2_rad = l1b['SoundingMeasurements/radiance_weak_co2'][...]
    sco2_rad = l1b['SoundingMeasurements/radiance_strong_co2'][...]

    o2a_qf = l1b['FootprintGeometry/footprint_o2_qual_flag'][...]

    c = 299792458.0
    h = 6.62607015e-34

    o2a_con = l1b['SoundingMeasurements']['rad_continuum_o2'][...]
    o2a_sza = l1b['FootprintGeometry/footprint_solar_zenith'][...][:, :, 0]
    o2a_vza = l1b['FootprintGeometry/footprint_zenith'][...][:, :, 0]
    o2a_alt = l1b['FootprintGeometry/footprint_altitude'][...][:, :, 0]

    wco2_sza = l1b['FootprintGeometry/footprint_solar_zenith'][...][:, :, 1]
    wco2_vza = l1b['FootprintGeometry/footprint_zenith'][...][:, :, 1]
    wco2_alt = l1b['FootprintGeometry/footprint_altitude'][...][:, :, 1]

    sco2_sza = l1b['FootprintGeometry/footprint_solar_zenith'][...][:, :, 2]
    sco2_vza = l1b['FootprintGeometry/footprint_zenith'][...][:, :, 2]
    sco2_alt = l1b['FootprintGeometry/footprint_altitude'][...][:, :, 2]

    o2a_mu = np.cos(o2a_sza/180*np.pi)
    wco2_mu = np.cos(wco2_sza/180*np.pi)
    sco2_mu = np.cos(sco2_sza/180*np.pi)

    o2a_mu_v = np.cos(o2a_vza/180*np.pi)
    wco2_mu_v = np.cos(wco2_vza/180*np.pi)
    sco2_mu_v = np.cos(sco2_vza/180*np.pi)

    o2a_mu_r = o2a_mu.repeat(1016).reshape((o2a_mu.shape[0], o2a_mu.shape[1], 1016))
    wco2_mu_r = wco2_mu.repeat(1016).reshape((o2a_mu.shape[0], o2a_mu.shape[1], 1016))
    sco2_mu_r = sco2_mu.repeat(1016).reshape((o2a_mu.shape[0], o2a_mu.shape[1], 1016))
    o2a_mu_v_r = o2a_mu_v.repeat(1016).reshape((o2a_mu.shape[0], o2a_mu.shape[1], 1016))
    wco2_mu_v_r = wco2_mu_v.repeat(1016).reshape((o2a_mu.shape[0], o2a_mu.shape[1], 1016))
    sco2_mu_v_r = sco2_mu_v.repeat(1016).reshape((o2a_mu.shape[0], o2a_mu.shape[1], 1016))

    # modl2 = modis_l2(fnames=sat0.fnames['mod_l2'], extent=sat0.extent, vnames=['cloud_top_height_1km'])
    # lon0, lat0 = [modl2.data[var]['data'] for var in ['lon', 'lat']]
    # lon, lat  = upscale_modis_lonlat(lon0, lat0, scale=5, extra_grid=True)
    # cth0  = modl2.data['cloud_top_height_1km']['data']/1000.0 # units: km
    # cth0[cth0<=0.0] = np.nan
    from pyhdf.SD import SD, SDC
    f     = SD(sat0.fnames['mod_l2'][0].replace('../simulation/data/', '../sat_data/'), SDC.READ)
    lat0       = f.select('Latitude')
    lon0       = f.select('Longitude')
    cth0 = f.select('cloud_top_height_1km')[:]/1000
    lon, lat  = upscale_modis_lonlat(lon0[:], lat0[:], scale=5, extra_grid=True)

    snd_id = l1b['SoundingGeometry']['sounding_id'][...]
    snd_lon = l1b['SoundingGeometry']['sounding_longitude'][...]
    snd_lat = l1b['SoundingGeometry']['sounding_latitude'][...]
    l2_id = l2['RetrievalHeader']['sounding_id'][...]
    l2_co2 = l2['RetrievalResults']['xco2'][...]
    valid_id = l2_id[l2_co2>0]
    xco2_mask = np.isin(snd_id, valid_id)
    loc_mask = np.logical_and(np.logical_and(snd_lon>=extent[0], snd_lon<=extent[1]),
                              np.logical_and(snd_lat>=extent[2], snd_lat<=extent[3]))
    snd_mask = np.logical_and(xco2_mask, loc_mask)
    snd_lon = snd_lon[snd_mask]
    snd_lat = snd_lat[snd_mask]
    sfc_p_met = met["Meteorology"]["surface_pressure_met"][...]
    sfc_p_met[sfc_p_met<0] = np.nan
    
    sfc_T_met = met["Meteorology"]["skin_temperature_met"][...]

    nearest_cloud_dist = cld_dist_calc(snd_lon, snd_lat, cth0, lon, lat, cfg_info, fdir_data)
    weighted_cloud_dist = weighted_cld_dist_calc(snd_lon, snd_lat, cth0, lon, lat, cfg_info, fdir_data)
    large_weighted_cloud_dist = np.logical_and(np.isnan(weighted_cloud_dist), ~np.isnan(nearest_cloud_dist))
    weighted_cloud_dist[large_weighted_cloud_dist] = nearest_cloud_dist[large_weighted_cloud_dist]
    plt_map_cld_dis(sat0, cth0, lon, lat, nearest_cloud_dist, snd_lon, snd_lat, fdir_data)
    plt_map_weight_cld_dis(sat0, cth0, lon, lat, weighted_cloud_dist, snd_lon, snd_lat, fdir_data)
    hist_cld_dist(sat0, nearest_cloud_dist, weighted_cloud_dist, fdir_data)
    sys.exit()
    cloud_dist = weighted_cloud_dist
    o2a_rad_convert = convert_photon_unit(o2a_rad, lam[:, :, 0]*1e3)
    o2a_ref_convert = o2a_rad_convert*np.pi/(oxs*o2a_mu_r)

    wco2_rad_convert = convert_photon_unit(wco2_rad, lam[:, :, 1]*1e3)
    wco2_ref_convert = wco2_rad_convert*np.pi/(wcs*wco2_mu_r)

    sco2_rad_convert = convert_photon_unit(sco2_rad, lam[:, :, 2]*1e3)
    sco2_ref_convert = sco2_rad_convert*np.pi/(scs*sco2_mu_r)

    ref_min, ref_max = 92, 95
    o2a_ref_con_convert = np.percentile(o2a_ref_convert, np.linspace(ref_min, ref_max, 11), axis=2).mean(axis=0)
    wco2_ref_con_convert = np.percentile(wco2_ref_convert, np.linspace(ref_min, ref_max, 11), axis=2).mean(axis=0)
    sco2_ref_con_convert = np.percentile(sco2_ref_convert, np.linspace(ref_min, ref_max, 11), axis=2).mean(axis=0)

    o2a_ref_con_expand = np.repeat(o2a_ref_con_convert, 1016).reshape(o2a_ref_con_convert.shape[0], o2a_ref_con_convert.shape[1], 1016)
    wco2_ref_con_expand = np.repeat(wco2_ref_con_convert, 1016).reshape(wco2_ref_con_convert.shape[0], wco2_ref_con_convert.shape[1], 1016)
    sco2_ref_con_expand = np.repeat(sco2_ref_con_convert, 1016).reshape(sco2_ref_con_convert.shape[0], sco2_ref_con_convert.shape[1], 1016)

    o2a_absorbed = (o2a_ref_con_expand-o2a_ref_convert)
    o2a_abs_3d = -np.log10(1-o2a_absorbed)

    wco2_absorbed = (wco2_ref_con_expand-wco2_ref_convert)
    wco2_abs_3d = -np.log10(1-wco2_absorbed)

    sco2_absorbed = (sco2_ref_con_expand-sco2_ref_convert)
    sco2_abs_3d = -np.log10(1-sco2_absorbed)

    A = o2a_abs_3d
    A_o2a = o2a_abs_3d
    A_wco2 = wco2_abs_3d
    A_sco2 = sco2_abs_3d

    o2a_fwhm = o2a_fwhm_calc(A_o2a, lam, fdir_data)
    wco2_fwhm = wco2_fwhm_calc(A_wco2, lam, fdir_data)
    sco2_fwhm = sco2_fwhm_calc(A_sco2, lam, fdir_data)

    litefile = sat0.fnames['oco_lite'][0].replace('../simulation/data/', '../sat_data/')
    lite = xr.open_dataset(litefile)
    lite_snd = xr.open_dataset(litefile, group='Sounding')
    lite_retrieve = xr.open_dataset(litefile, group='Retrieval')
    lon_w, lon_e = extent[0], extent[1]
    lat_s, lat_n = extent[2], extent[3]
    lon_range = np.logical_and(lite.longitude >= lon_w, lite.longitude <= lon_e)
    lat_range = np.logical_and(lite.latitude >= lat_s, lite.latitude <= lat_n)

    select = np.where(np.logical_and(lon_range, lat_range))
    lite_snd_list = np.array(lite.sounding_id[select], dtype=int)
    xco2_array = np.zeros((A.shape[0], A.shape[1]))
    qf_array = np.zeros((A.shape[0], A.shape[1], 3))
    sfc_p = np.zeros((A.shape[0], A.shape[1]), dtype=float)
    qf_lon, qf_lat = np.zeros((A.shape[0], A.shape[1])), np.zeros((A.shape[0], A.shape[1]))
    # qf, qf_bitflag, qf_simpleflag\
    sfc_p_l2 = l2["RetrievalResults/surface_pressure_fph"][...]
    sfc_p_l2[sfc_p_l2<0] = np.nan
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            snd = snd_id[i, j]
            #print('-'*15)
            #print(snd)
            if snd in lite_snd_list:
                index = np.where(lite_snd_list == snd)[0][0]

                xco2_quality_flag = float(lite.xco2_quality_flag[select][index])
                xco2_qf_bitflag = float(lite.xco2_qf_bitflag[select][index])
                xco2_qf_simple_bitflag = float(lite.xco2_qf_simple_bitflag[select][index])
                xco2_array[i, j] = float(lite.xco2[select][index])
                qf_array[i, j, :] = (xco2_quality_flag, xco2_qf_bitflag, xco2_qf_simple_bitflag)
                qf_lat[i, j] = float(lite.latitude[select][index])
                qf_lon[i, j] = float(lite.longitude[select][index])
                sfc_p[i, j] = float(lite_retrieve.psurf[select][index])
                # alt_array[i, j] = float(lite_snd.altitude[select][index])
            else:
                xco2_array[i, j] = np.nan
                qf_array[i, j, :] = (np.nan,)*3
                qf_lat[i, j] = np.nan
                qf_lon[i, j] = np.nan
                sfc_p[i, j] = np.nan
                # alt_array[i, j] = np.nan

    vmin = 10
    vmax = 500
    select = np.isnan(snd_lat[:, :])==False
    marker_list = ['o', 's', 'd', 'v', '^', '<', '>', 'p', 'h', '8', 'D', 'P', 'X']
    
    plt.clf()
    plt.scatter(sfc_p.flatten(), sfc_p_met.flatten(), alpha=0.7)
    plt.show()

    ### with valid sfc_p
    plt.clf()
    plt.figure(figsize=(8, 6))
    for i in range(8):
        select_fp = np.isnan(snd_lat[:, i])==False
        plt.scatter((o2a_fwhm)[:, i][select_fp], sfc_p[:, i][select_fp], c=cloud_dist[:, i][select_fp],
                    marker=marker_list[i],
                    norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), alpha=0.7)
    effective = np.logical_and(np.logical_and((o2a_fwhm)[:, :][select]>0, sfc_p[:, :][select]>0),
                            cloud_dist[:, :][select]>0)
    plt.colorbar(extend='both')
    plt.title(f'counts: {np.sum(effective)}')
    plt.savefig(f'fig/{cfg_info["cfg_name"]}_valid_pressure_o2a_fwhm_cloud_dist_{extent[0]:.2f}_{extent[1]:.2f}_{extent[2]:.2f}_{extent[3]:.2f}.png')

    plt.clf()
    plt.figure(figsize=(8, 6))
    for i in range(8):
        select_fp = np.isnan(snd_lat[:, i])==False
        plt.scatter(((o2a_fwhm)/(o2a_mu+o2a_mu_v))[:, i][select_fp], sfc_p[:, i][select_fp], c=xco2_array[:, i][select_fp],
                    marker=marker_list[i],
                    # norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), 
                    alpha=0.7)
    effective = np.logical_and(np.logical_and((o2a_fwhm)[:, :][select]>0, sfc_p[:, :][select]>0),
                            cloud_dist[:, :][select]>0)
    plt.colorbar(extend='both')
    plt.title(f'counts: {np.sum(effective)}')
    plt.savefig(f'fig/{cfg_info["cfg_name"]}_valid_pressure_o2a_fwhm_cloud_dist_{extent[0]:.2f}_{extent[1]:.2f}_{extent[2]:.2f}_{extent[3]:.2f}_test_xco2.png')

    plt.clf()
    plt.figure(figsize=(8, 6))
    for i in range(1):
        select_fp = np.isnan(snd_lat[:, i])==False
        plt.scatter(((o2a_fwhm)/(o2a_mu+o2a_mu_v))[:, i][select_fp], sfc_p[:, i][select_fp], c=cloud_dist[:, i][select_fp],
                    marker=marker_list[i],
                    norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), alpha=0.7)
    effective = np.logical_and(np.logical_and((o2a_fwhm)[:, :][select]>0, sfc_p[:, :][select]>0),
                            cloud_dist[:, :][select]>0)
    plt.colorbar(extend='both')
    plt.title(f'counts: {np.sum(effective)}')
    plt.savefig(f'fig/{cfg_info["cfg_name"]}_valid_pressure_o2a_fwhm_cloud_dist_{extent[0]:.2f}_{extent[1]:.2f}_{extent[2]:.2f}_{extent[3]:.2f}_test.png')


    plt.clf()
    plt.figure(figsize=(8, 6))
    for i in range(8):
        select_fp = np.isnan(snd_lat[:, i])==False
        plt.scatter((wco2_fwhm)[:, i][select_fp], sfc_p[:, i][select_fp], c=cloud_dist[:, i][select_fp],
                    marker=marker_list[i],
                    norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), alpha=0.7)
    effective = np.logical_and(np.logical_and((wco2_fwhm)[:, :][select]>0, sfc_p[:, :][select]>0),
                            cloud_dist[:, :][select]>0)
    plt.colorbar(extend='both')
    plt.title(f'counts: {np.sum(effective)}')
    plt.savefig(f'fig/{cfg_info["cfg_name"]}_valid_pressure_wco2_fwhm_cloud_dist_{extent[0]:.2f}_{extent[1]:.2f}_{extent[2]:.2f}_{extent[3]:.2f}.png')

    plt.clf()
    plt.figure(figsize=(8, 6))
    for i in range(8):
        select_fp = np.isnan(snd_lat[:, i])==False
        plt.scatter((sco2_fwhm)[:, i][select_fp], sfc_p[:, i][select_fp], c=cloud_dist[:, i][select_fp],
                    marker=marker_list[i],
                    norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), alpha=0.7)
    effective = np.logical_and(np.logical_and((sco2_fwhm)[:, :][select]>0, sfc_p[:, :][select]>0),
                            cloud_dist[:, :][select]>0)
    plt.colorbar(extend='both')
    plt.title(f'counts: {np.sum(effective)}')
    plt.savefig(f'fig/{cfg_info["cfg_name"]}_valid_pressure_sco2_fwhm_cloud_dist_{extent[0]:.2f}_{extent[1]:.2f}_{extent[2]:.2f}_{extent[3]:.2f}.png')

    ### only with valid xco2
    select = np.logical_and(np.isnan(snd_lat[:, :])==False, 
                            np.isnan(xco2_array[:, :])==False)
    plt.clf()
    plt.figure(figsize=(8, 6))
    for i in range(8):
        select_fp = np.logical_and(np.isnan(snd_lat[:, i])==False, 
                                   np.isnan(xco2_array[:, i])==False)
        plt.scatter((o2a_fwhm)[:, i][select_fp], sfc_p[:, i][select_fp], c=cloud_dist[:, i][select_fp],
                    marker=marker_list[i],
                    norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), alpha=0.7)
    effective = np.logical_and(np.logical_and((o2a_fwhm)[:, :][select]>0, sfc_p[:, :][select]>0),
                            cloud_dist[:, :][select]>0)
    plt.colorbar(extend='both')
    plt.title(f'counts: {np.sum(effective)}')
    plt.savefig(f'fig/{cfg_info["cfg_name"]}_valid_xco2_o2a_fwhm_cloud_dist_{extent[0]:.2f}_{extent[1]:.2f}_{extent[2]:.2f}_{extent[3]:.2f}.png')

    plt.clf()
    plt.figure(figsize=(8, 6))
    for i in range(8):
        select_fp = np.logical_and(np.isnan(snd_lat[:, i])==False, 
                                   np.isnan(xco2_array[:, i])==False)
        plt.scatter(((o2a_fwhm)/(o2a_mu+o2a_mu_v))[:, i][select_fp], sfc_p[:, i][select_fp], c=cloud_dist[:, i][select_fp],
                    marker=marker_list[i],
                    norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), alpha=0.7)
    effective = np.logical_and(np.logical_and((o2a_fwhm)[:, :][select]>0, sfc_p[:, :][select]>0),
                            cloud_dist[:, :][select]>0)
    plt.colorbar(extend='both')
    plt.title(f'counts: {np.sum(effective)}')
    plt.savefig(f'fig/{cfg_info["cfg_name"]}_valid_xco2_o2a_fwhm_cloud_dist_{extent[0]:.2f}_{extent[1]:.2f}_{extent[2]:.2f}_{extent[3]:.2f}_test.png')


    plt.clf()
    plt.figure(figsize=(8, 6))
    for i in range(8):
        select_fp = np.logical_and(np.isnan(snd_lat[:, i])==False, 
                                   np.isnan(xco2_array[:, i])==False)
        plt.scatter((wco2_fwhm)[:, i][select_fp], sfc_p[:, i][select_fp], c=cloud_dist[:, i][select_fp],
                    marker=marker_list[i],
                    norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), alpha=0.7)
    effective = np.logical_and(np.logical_and((wco2_fwhm)[:, :][select]>0, sfc_p[:, :][select]>0),
                            cloud_dist[:, :][select]>0)
    plt.colorbar(extend='both')
    plt.title(f'counts: {np.sum(effective)}')
    plt.savefig(f'fig/{cfg_info["cfg_name"]}_valid_xco2_wco2_fwhm_cloud_dist_{extent[0]:.2f}_{extent[1]:.2f}_{extent[2]:.2f}_{extent[3]:.2f}.png')

    plt.clf()
    plt.figure(figsize=(8, 6))
    for i in range(8):
        select_fp = np.logical_and(np.isnan(snd_lat[:, i])==False, 
                                   np.isnan(xco2_array[:, i])==False)
        plt.scatter((sco2_fwhm)[:, i][select_fp], sfc_p[:, i][select_fp], c=cloud_dist[:, i][select_fp],
                    marker=marker_list[i],
                    norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), alpha=0.7)
    effective = np.logical_and(np.logical_and((sco2_fwhm)[:, :][select]>0, sfc_p[:, :][select]>0),
                            cloud_dist[:, :][select]>0)
    plt.colorbar(extend='both')
    plt.title(f'counts: {np.sum(effective)}')
    plt.savefig(f'fig/{cfg_info["cfg_name"]}_valid_xco2_sco2_fwhm_cloud_dist_{extent[0]:.2f}_{extent[1]:.2f}_{extent[2]:.2f}_{extent[3]:.2f}.png')


def o2a_fwhm_calc(A, lam, fdir_data):
    # o2a
    filename = f'{fdir_data}/o2a_fwhm.npy'
    if os.path.isfile(filename):
        FWHM_array = np.load(filename)
    else:
        print('Calculate O2-A band FWHM...')
        FWHM_array = np.zeros((A.shape[0], A.shape[1]))

        start, end = 268, 287
        wl = lam[:, :, 0]
        plt_test = 0
        for i in range(A.shape[0]):
            for fp in range(8):
                max_ind = np.argmax((A[i, fp, start:end]))
                half_interval = 4
                part_start, part_end = start+max_ind-6, start+max_ind+4+1
                if (A[i , fp, part_start:part_end]<5e-4).all():
                    FWHM_array[i,fp] = np.nan
                else:
                    try:
                        popt, pcov = curve_fit(V0, lam[fp, part_start:part_end, 0], A[i , fp, part_start:part_end],
                                            p0=[0.7623, 1e-5, 1e-5, 50000],
                                            bounds=([0.7622, 1e-11, 1e-11, 1e2], 
                                                        [0.7624, 1e-3, 1e-3, 5e8]),
                                            max_nfev=2500)
                        wvl_linspace = np.linspace(lam[fp, start, 0], lam[fp, end, 0], 401)


                        A_min = 0#popt[-1]
                        A_max = V0(popt[0], *popt)
                        hm = (A_max-A_min)/2+A_min

                        xx = wvl_linspace
                        yy = V0(wvl_linspace, *popt)

                        yy_max_ind = np.argmax(yy)

                        xx_l = xx[:yy_max_ind]
                        yy_l = yy[:yy_max_ind]
                        hm_ind_l = np.argmin(np.abs(yy_l-hm))

                        xx_r = xx[yy_max_ind-1:]
                        yy_r = yy[yy_max_ind-1:]
                        hm_ind_r = np.argmin(np.abs(yy_r-hm))


                        FWHM = np.abs(xx_r[hm_ind_r]-xx_l[hm_ind_l])*1e3 # to nm
                        FWHM_array[i,fp] = FWHM

                        if FWHM < 0.002:
                            plt.clf()
                            print('i, fp:', i, fp)
                            print('A max:', A_max)
                            print('A min:', A_min)
                            print(popt)
                            print(FWHM)
                            #plt.vlines(wl[fp, start+max_ind], 0, A[i,fp,max_ind+start],'green')
                            plt.plot(wl[fp, start:end], A[i,fp,start:end])
                            plt.plot(wvl_linspace, yy, 'g--')
                            plt.vlines(xx_r[hm_ind_r], 0, hm*2, 'r')
                            plt.vlines(xx_l[hm_ind_l], 0, hm*2, 'r')
                            plt.fill_betweenx([0, hm*2], lam[fp, part_start, 0], lam[fp, part_end-1, 0], color='orange', alpha=0.25)
                            # plt.show()
                        if FWHM >= 0.19 and plt_test<5:
                            print('-'*15)
                            print('i, fp:', i, fp)
                            print('A max:', A_max)
                            print('A min:', A_min)
                            print(popt)
                            print(FWHM)

                            plt.clf()
                            plt.plot(wvl_linspace, yy, 'g--')
                            plt.plot(wl[fp, start:end], A[i,fp,start:end])
                            plt.vlines(xx_r[hm_ind_r], 0, hm*2, 'r')
                            plt.vlines(xx_l[hm_ind_l], 0, hm*2, 'r')
                            plt.fill_betweenx([0, hm*2], lam[fp, part_start, 0], lam[fp, part_end-1, 0], color='orange', alpha=0.25)
                            # plt.show()

                            plt_test += 1
                    except:
                        FWHM_array[i,fp] = np.nan
            np.save(filename, FWHM_array)
    return FWHM_array

def wco2_fwhm_calc(A_wco2, lam, fdir_data):
    # wco2
    filename = f'{fdir_data}/wco2_fwhm.npy'
    if os.path.isfile(filename):
        FWHM_array_wco2 = np.load(filename)
    else:
        print('Calculate WCO2 band FWHM...')
        FWHM_array_wco2 = np.zeros((A_wco2.shape[0], A_wco2.shape[1]))

        start, end = 351, 364+1
        wl = lam[:, :, 1]
        plt_test = 0
        plt_small_test = 0

        for i in range(A_wco2.shape[0]):
            for fp in range(8):
                max_ind = np.argmax((A_wco2[i, fp, start:end]))
                half_interval = 4
                part_start, part_end = start+max_ind-3, start+max_ind+3+1
                if (A_wco2[i , fp, part_start:part_end]<5e-4).all():
                    FWHM_array_wco2[i,fp] = np.nan
                else:
                    try:
                        popt, pcov = curve_fit(V0, lam[fp, part_start:part_end, 1], A_wco2[i , fp, part_start:part_end],
                                            p0=[1.60285, 1e-5, 1e-5, 2e5],
                                            bounds=([1.60275, 1e-6, 1e-6, 1e3], 
                                                    [1.60295, 1e-4, 1e-4, 1e8]),
                                            max_nfev=10000)
                        wvl_linspace = np.linspace(lam[fp, start, 1], lam[fp, end, 1], 401)

                        max_ind = np.argmax((A_wco2[i,fp,start:end]))

                        A_min = 0
                        A_max = V0(popt[0], *popt)
                        hm = (A_max-A_min)/2+A_min

                        xx = wvl_linspace
                        yy = V0(wvl_linspace, *popt)

                        yy_max_ind = np.argmax(yy)

                        xx_l = xx[:yy_max_ind]
                        yy_l = yy[:yy_max_ind]
                        hm_ind_l = np.argmin(np.abs(yy_l-hm))

                        xx_r = xx[yy_max_ind-1:]
                        yy_r = yy[yy_max_ind-1:]
                        hm_ind_r = np.argmin(np.abs(yy_r-hm))


                        FWHM = np.abs(xx_r[hm_ind_r]-xx_l[hm_ind_l])*1e3 # to nm
                        FWHM_array_wco2[i,fp] = FWHM

                        if FWHM < 0.05 and plt_small_test<5:
                            FWHM_array_wco2[i,fp] = np.nan
                            plt.clf()
                            print('i, fp:', i, fp)
                            print('A max:', A_max)
                            print('A min:', A_min)
                            print(popt)
                            print(FWHM)
                            #plt.vlines(wl[fp, start+max_ind], 0, A[i,fp,max_ind+start],'green')
                            plt.plot(wl[fp, start:end], A_wco2[i,fp,start:end])
                            plt.plot(wvl_linspace, V0(wvl_linspace, *popt), 'g--')
                            plt.vlines(xx_r[hm_ind_r], 0, hm*2, 'r')
                            plt.vlines(xx_l[hm_ind_l], 0, hm*2, 'r')
                            plt.fill_betweenx([0, hm*2], lam[fp, part_start, 1], lam[fp, part_end-1, 1], color='orange', alpha=0.25)
                            # plt.show()
                            plt_small_test += 1
                        if FWHM >= 0.22 and plt_test<5:
                            print('-'*15)
                            print('plt_test:', plt_test)
                            print('i, fp:', i, fp)
                            print('A max:', A_max)
                            print('A min:', A_min)
                            print(popt)
                            print(FWHM)

                            plt.clf()
                            plt.plot(wvl_linspace, V0(wvl_linspace, *popt), 'g--')
                            plt.plot(wl[fp, start:end], A_wco2[i,fp,start:end])
                            plt.vlines(xx_r[hm_ind_r], 0, hm*2, 'r')
                            plt.vlines(xx_l[hm_ind_l], 0, hm*2, 'r')
                            plt.fill_betweenx([0, hm*2], lam[fp, part_start, 1], lam[fp, part_end-1, 1], color='orange', alpha=0.25)
                            # plt.show()

                            plt_test += 1
                    except:
                        FWHM_array_wco2[i,fp] = np.nan
        np.save(filename, FWHM_array_wco2)
    return FWHM_array_wco2

def sco2_fwhm_calc(A_sco2, lam, fdir_data):
    # sco2
    filename = f'{fdir_data}/sco2_fwhm.npy'
    if os.path.isfile(filename):
        FWHM_array_sco2 = np.load(filename)
    else:
        print('Calculate SCO2 band FWHM...')
        FWHM_array_sco2 = np.zeros((A_sco2.shape[0], A_sco2.shape[1]))

        start, end = 418, 434+1
        wl = lam[:, :, 2]
        plt_test = 0
        for i in range(A_sco2.shape[0]):
            for fp in range(8):
                max_ind = np.argmax((A_sco2[i, fp, start:end]))
                half_interval = 4
                part_start, part_end = start+max_ind-4, start+max_ind+4+1
                if (A_sco2[i , fp, start:end]<5e-4).all():
                    FWHM_array_sco2[i,fp] = np.nan
                else:
                    try:
                        popt, pcov = curve_fit(V0, lam[fp, part_start:part_end, 2], A_sco2[i , fp, part_start:part_end],
                                            p0=[2.0617, 1e-4, 1e-4, 1e5],
                                            bounds=([2.0616, 1e-6, 1e-6, 1e3], 
                                                    [2.0618, 1e-3, 1e-3, 1e8]),
                                            max_nfev=5000)
                        wvl_linspace = np.linspace(lam[fp, start, 2], lam[fp, end, 2], 401)

                        max_ind = np.argmax((A_sco2[i,fp,start:end]))

                        A_min = 0
                        A_max = V0(popt[0], *popt)
                        hm = (A_max-A_min)/2+A_min

                        xx = wvl_linspace
                        yy = V0(wvl_linspace, *popt)

                        yy_max_ind = np.argmax(yy)

                        xx_l = xx[:yy_max_ind]
                        yy_l = yy[:yy_max_ind]
                        hm_ind_l = np.argmin(np.abs(yy_l-hm))

                        xx_r = xx[yy_max_ind-1:]
                        yy_r = yy[yy_max_ind-1:]
                        hm_ind_r = np.argmin(np.abs(yy_r-hm))


                        FWHM = np.abs(xx_r[hm_ind_r]-xx_l[hm_ind_l])*1e3 # to nm
                        FWHM_array_sco2[i,fp] = FWHM

                        if FWHM < 0.005:
                            plt.clf()
                            print('i, fp:', i, fp)
                            print('A max:', A_max)
                            print('A min:', A_min)
                            print(popt)
                            print(FWHM)
                            #plt.vlines(wl[fp, start+max_ind], 0, A[i,fp,max_ind+start],'green')
                            plt.plot(wl[fp, start:end], A_sco2[i,fp,start:end])
                            plt.plot(wvl_linspace, yy, 'g--')
                            plt.vlines(xx_r[hm_ind_r], 0, hm*2, 'r')
                            plt.vlines(xx_l[hm_ind_l], 0, hm*2, 'r')
                            plt.fill_betweenx([0, hm*2], lam[fp, part_start, 2], lam[fp, part_end-1, 2], color='orange', alpha=0.25)
                            # plt.show()
                        if FWHM >= 0.38 and plt_test<5:
                            print('-'*15)
                            print('i, fp:', i, fp)
                            print('A max:', A_max)
                            print('A min:', A_min)
                            print(popt)
                            print(FWHM)

                            plt.clf()
                            plt.plot(wvl_linspace, yy, 'g--')
                            plt.plot(wl[fp, start:end], A_sco2[i,fp,start:end])
                            plt.vlines(xx_r[hm_ind_r], 0, hm*2, 'r')
                            plt.vlines(xx_l[hm_ind_l], 0, hm*2, 'r')
                            plt.fill_betweenx([0, hm*2], lam[fp, part_start, 2], lam[fp, part_end-1, 2], color='orange', alpha=0.25)
                            # plt.show()

                            plt_test += 1
                    except:
                        FWHM_array_sco2[i,fp] = np.nan
        np.save(filename, FWHM_array_sco2)
    return FWHM_array_sco2

def run_simulation(cfg, sfc_alb=None, sza=None):
    cfg_info = grab_cfg(cfg)
    preprocess(cfg_info)
    starttime = timeit.default_timer()
    run_case_modis_650(cfg_info)
    endtime = timeit.default_timer()
    print(f'Duration: {(endtime-starttime)/3600.:.2f} h')

    


if __name__ == '__main__':
    
    # cfg = 'cfg/20181018_central_asia_2_470cloud_test2.csv'
    # cfg = 'cfg/20151219_north_italy_470cloud_test.csv'
    #cfg = 'cfg/20190621_australia-2-470cloud_aod.csv'
    #cfg = 'cfg/20161023_north_france_test.csv'
    # cfg = 'cfg/20170605_amazon.csv'
    # cfg = 'cfg/20190209_dryden.csv'
    # cfg = 'cfg/20181018_central_asia.csv'
    # cfg = 'cfg/20190621_australia.csv'
    cfg = 'cfg/20151220_afric_east.csv'
    cfg = 'cfg/20151201_ocean_2.csv'
    print(cfg)
    run_simulation(cfg) #done
    