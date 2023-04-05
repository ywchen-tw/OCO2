

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
from er3t.pre.cld import cld_sat
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
from oco_subroutine.oco_cloud import para_corr, wind_corr
from oco_subroutine.oco_cfg import grab_cfg, save_h5_info, check_h5_info, save_subdomain_info
from oco_subroutine.oco_abs_snd_sat import oco_abs
from oco_subroutine.oco_modis_time import cal_sat_delta_t
from oco_subroutine.oco_raw_collect import cdata_sat_raw
from oco_subroutine.oco_cloud import cdata_cld_ipa
from oco_subroutine.oco_modis_650 import cal_mca_rad_650, modis_650_simulation_plot

import timeit
import argparse
import matplotlib.image as mpl_img



class sat_tmp:

    def __init__(self, data):

        self.data = data


def cal_mca_rad_oco2(date, tag, sat, zpt_file, wavelength, fname_idl=None, cth=None, 
                     photons=1e6, scale_factor=1.0, 
                     fdir='tmp-data', solver='3D', 
                     sfc_alb_abs=None, sza_abs=None, overwrite=True):

    """
    Calculate OCO2 radiance using cloud (MODIS level 1b) and surface properties (MOD09A1) from MODIS
    """

    # atm object
    # =================================================================================
    oco_zpt = h5py.File(zpt_file, 'r')
    levels = oco_zpt['h_edge'][...]
    oco_zpt.close()
    fname_atm = '%s/atm.pk' % fdir
    atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    # =================================================================================

    # abs object, in the future, we will implement OCO2 MET file for this
    # =================================================================================
    fname_abs = '%s/abs.pk' % fdir
    abs0      = abs_oco_h5(wavelength=wavelength, fname=fname_abs, fname_h5=fname_idl, atm_obj=atm0, overwrite=overwrite)
    # =================================================================================

    # mca_sfc object
    # =================================================================================
    #"""
    data = {}
    f_pre_data = h5py.File(f'{sat.fdir_out}/pre-data.h5', 'r')
    data['alb_2d'] = dict(data=f_pre_data[f'oco/sfc/alb_{tag}_2d'][...], name='Surface albedo', units='N/A')
    data['lon_2d'] = dict(data=f_pre_data['mod/sfc/lon'][...], name='Longitude', units='degrees')
    data['lat_2d'] = dict(data=f_pre_data['mod/sfc/lat'][...], name='Latitude' , units='degrees')

    #"""
    avg_sfc_alb = np.nanmean(data['alb_2d']['data'][...])
    print('Average sfc albedo: ', avg_sfc_alb)
    simulated_sfc_alb = avg_sfc_alb
    if sfc_alb_abs != None:
        simulated_sfc_alb = sfc_alb_abs
        data['alb_2d']['data'][...] = simulated_sfc_alb

    mod43     = sat_tmp(data)
    fname_sfc = '%s/sfc.pk' % fdir
    sfc0      = sfc_sat(sat_obj=mod43, fname=fname_sfc, extent=sat.extent, verbose=True, overwrite=overwrite)
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
    data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=f_pre_data['lon'][...])
    data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=f_pre_data['lat'][...])
    data['rad_2d'] = dict(name='Gridded radiance'                , units='km'         , data=f_pre_data[f'mod/rad/rad_650'][...])
    if solver.lower() == 'ipa':
        data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=f_pre_data['mod/cld/cot_ipa0'][...])
        data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=f_pre_data['mod/cld/cer_ipa0'][...])
        data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=f_pre_data['mod/cld/cth_ipa0'][...])
    elif solver.lower() == '3d':
        data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=f_pre_data['mod/cld/cot_ipa'][...])
        data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=f_pre_data['mod/cld/cer_ipa'][...])
        data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=f_pre_data['mod/cld/cth_ipa'][...])

    # =================================================================================

    # aod object
    # =================================================================================
    AOD_550_land_mean = f_pre_data['mod/aod/AOD_550_land_mean'][...]
    Angstrom_Exponent_land_mean = f_pre_data['mod/aod/Angstrom_Exponent_land_mean'][...]
    SSA_land_mean = f_pre_data['mod/aod/SSA_660_land_mean'][...]
    # =================================================================================

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

    f_pre_data.close()
    # =================================================================================


    f_oco_l1b = h5py.File(sat.fnames['oco_l1b'][0], 'r')
    lon_oco_l1b = f_oco_l1b['SoundingGeometry/sounding_longitude'][...]
    lat_oco_l1b = f_oco_l1b['SoundingGeometry/sounding_latitude'][...]
    logic = (lon_oco_l1b>=sat.extent[0]) & (lon_oco_l1b<=sat.extent[1]) & (lat_oco_l1b>=sat.extent[2]) & (lat_oco_l1b<=sat.extent[3])
    sza = f_oco_l1b['SoundingGeometry/sounding_solar_zenith'][...][logic].mean()
    saa = f_oco_l1b['SoundingGeometry/sounding_solar_azimuth'][...][logic].mean()
    vza = f_oco_l1b['SoundingGeometry/sounding_zenith'][...][logic].mean()
    vaa = f_oco_l1b['SoundingGeometry/sounding_azimuth'][...][logic].mean()
    f_oco_l1b.close()
    if sza_abs != None:
        sza = sza_abs

    if simulated_sfc_alb <= 0.2:
        photons = photons*2

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

        sfc0      = sfc_sat(sat_obj=mod43, fname=fname_sfc, extent=sat.extent, verbose=True, overwrite=False)
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
        ax1.set_title('MODIS Chanel 1')

        cs = ax2.imshow(out0.data['rad']['data'].T, cmap='Greys_r', origin='lower', extent=sat.extent, zorder=0)
        ax2.scatter(oco_std0.data['lon']['data'], oco_std0.data['lat']['data'], s=20, c=oco_std0.data['xco2']['data'], cmap='jet', alpha=0.4, zorder=1)
        ax2.set_title('MCARaTS %s' % solver)

        for ax in [ax1, ax2]:
            ax.set_xlabel('Longitude [$^\circ$]')
            ax.set_ylabel('Latitude [$^\circ$]')
            ax.set_xlim(sat.extent[:2])
            ax.set_ylim(sat.extent[2:])

        plt.subplots_adjust(hspace=0.5)
        if cth is not None:
            plt.savefig('%s/mca-out-rad-modis-%s0_cth-%.2fkm_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), cth, scale_factor, wavelength), bbox_inches='tight')
        else:
            plt.savefig('%s/mca-out-rad-modis-%s0_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), scale_factor, wavelength), bbox_inches='tight')
        plt.close(fig)
        # ------------------------------------------------------------------------------------------------------
    return simulated_sfc_alb, sza


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

def cdata_all(date, tag, fdir_mca, fname_abs, sat, sfc_alb, sza):

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
    f = h5py.File(fname_abs, 'r')
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
    Np = np.zeros(wvls.size, dtype=np.float64)

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

    output_file = 'data_all_%s_%s_%4.4d_%4.4d_sfc_alb_%.3f_sza_%.1f.h5' % (date.strftime('%Y%m%d'), tag, oco.index_s, oco.index_e, sfc_alb, sza)
    f = h5py.File(output_file, 'w')
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

    return output_file


def run_case(band_tag, cfg_info, sfc_alb=None, sza=None):

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
        save_h5_info(cfg_info['cfg_path'], 'co2prior', sat0.fnames['oco_co2prior'][0].split('/')[-1])
    # create tmp-data/australia_YYYYMMDD directory if it does not exist
    # ===============================================================
    #fdir_tmp = os.path.abspath('tmp-data/%s/%s' % (name_tag, band_tag))
    if sfc_alb != None:
        fdir_tmp = os.path.abspath('tmp-data/%s_alb_%.3f_saz_%.1f/%s' % (name_tag, sfc_alb, sza, band_tag))
    else:
        fdir_tmp = os.path.abspath('tmp-data/%s/%s' % (name_tag, band_tag))
    print(fdir_tmp)
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

    if not os.path.isfile(f'{sat０.fdir_out}/pre-data.h5') :
        cdata_sat_raw(band_tag, sat0=sat０, overwrite=True, plot=True)

    # ===============================================================
    #"""
    for solver in ['IPA', '3D']:
        cdata_cld_ipa(band_tag, sat０, fdir_tmp, fdir_cot_tmp, zpt_file, ref_threshold=ref_threshold, photons=1e6, plot=True)
    #"""

    
    """
    # run calculations for 650 nm
    # ===============================================================
    fdir_tmp_650 = os.path.abspath('tmp-data/%s/%s' % (name_tag, 'modis_650'))
    if not os.path.exists(fdir_tmp_650):
        os.makedirs(fdir_tmp_650)
    cal_mca_rad_650(sat0, zpt_file, 650, fdir=fdir_tmp_650, solver='IPA', overwrite=False, case_name_tag=name_tag, photons=1e8)
    modis_650_simulation_plot(extent, case_name_tag=name_tag, fdir=fdir_tmp_650, solver='IPA', wvl=650, ref_threshold=ref_threshold, plot=True)
    #"""

    #"""
    # run calculations for each wavelength
    # ===============================================================
    for wavelength in wvls:
        for solver in ['IPA', '3D']:
            simulated_sfc_alb, sza = cal_mca_rad_oco2(date, band_tag, sat0, zpt_file, wavelength,
                                                      fname_idl=fname_abs, cth=None, scale_factor=1.0, 
                                                      fdir=fdir_tmp, solver=solver, 
                                                      sfc_alb_abs=sfc_alb, sza_abs=sza,
                                                      overwrite=True, photons=5e7)
    # ===============================================================
    #"""

    #"""
    # post-processing - combine the all the calculations into one dataset
    # ===============================================================
    return cdata_all(date, band_tag, fdir_tmp, fname_abs, sat0, simulated_sfc_alb, sza)
    # ===============================================================
    #"""


def run_simulation(cfg, sfc_alb=None, sza=None):
    cfg_info = grab_cfg(cfg)
    #"""
    if 1:#not check_h5_info(cfg, 'o2'):
        starttime = timeit.default_timer()
        o2_h5 = run_case('o2a', cfg_info, sfc_alb=sfc_alb, sza=sza)
        save_h5_info(cfg, 'o2', o2_h5)
        endtime = timeit.default_timer()
        print('O2A band duration:',(endtime-starttime)/3600.,' h')
        time.sleep(120)
    #""" 
    
    #"""
    if 1:#not check_h5_info(cfg, 'wco2'):
        starttime = timeit.default_timer()
        wco2_h5 = run_case('wco2', cfg_info, sfc_alb=sfc_alb, sza=sza)
        save_h5_info(cfg, 'wco2', wco2_h5)
        endtime = timeit.default_timer()
        print('WCO2 band duration:',(endtime-starttime)/3600.,' h')
    #"""
    #""""
    time.sleep(120)
    if 1:#not check_h5_info(cfg, 'sco2'):
        starttime = timeit.default_timer()
        sco2_h5 = run_case('sco2', cfg_info, sfc_alb=sfc_alb, sza=sza)
        save_h5_info(cfg, 'sco2', sco2_h5)
        endtime = timeit.default_timer()
        print('SCO2 band duration:',(endtime-starttime)/3600.,' h')
    #"""

if __name__ == '__main__':
    
    # cfg = 'cfg/20181018_central_asia_2_470cloud_test2.csv'
    # cfg = 'cfg/20151219_north_italy_470cloud_test.csv'
    #cfg = 'cfg/20190621_australia-2-470cloud_aod.csv'
    cfg = 'cfg/20190209_dryden_470cloud.csv'
    print(cfg)
    run_simulation(cfg) #done
    # run_simulation(cfg, sfc_alb=0.3, sza=45) #done
    # run_simulation(cfg, sfc_alb=0.5, sza=45)
    # run_simulation(cfg, sfc_alb=0.2, sza=45)
    # run_simulation(cfg, sfc_alb=0.1, sza=45)
    # run_simulation(cfg, sfc_alb=0.05, sza=45)


    # run_simulation(cfg, sfc_alb=0.3, sza=30)
    # run_simulation(cfg, sfc_alb=0.5, sza=30)
    # run_simulation(cfg, sfc_alb=0.2, sza=30)
    # run_simulation(cfg, sfc_alb=0.1, sza=30)
    # run_simulation(cfg, sfc_alb=0.05, sza=30)

    # run_simulation(cfg, sfc_alb=0.3, sza=60)
    # run_simulation(cfg, sfc_alb=0.5, sza=60)
    # run_simulation(cfg, sfc_alb=0.2, sza=60)
    # run_simulation(cfg, sfc_alb=0.1, sza=60)
    # run_simulation(cfg, sfc_alb=0.05, sza=60)

    # run_simulation(cfg, sfc_alb=0.15, sza=30)
    # run_simulation(cfg, sfc_alb=0.15, sza=45)
    # run_simulation(cfg, sfc_alb=0.15, sza=60)

    # run_simulation(cfg, sfc_alb=0.25, sza=30)
    # run_simulation(cfg, sfc_alb=0.25, sza=45)
    # run_simulation(cfg, sfc_alb=0.25, sza=60)

    # run_simulation(cfg, sfc_alb=0.4, sza=30)
    # run_simulation(cfg, sfc_alb=0.4, sza=45)
    # run_simulation(cfg, sfc_alb=0.4, sza=60)

    # run_simulation(cfg, sfc_alb=0.025, sza=30)
    # run_simulation(cfg, sfc_alb=0.025, sza=45)
    # run_simulation(cfg, sfc_alb=0.025, sza=60)

    
    # run_simulation(cfg, sfc_alb=0.5, sza=15)
    # run_simulation(cfg, sfc_alb=0.4, sza=15)
    # run_simulation(cfg, sfc_alb=0.3, sza=15)
    # run_simulation(cfg, sfc_alb=0.25, sza=15)
    # run_simulation(cfg, sfc_alb=0.2, sza=15)
    # run_simulation(cfg, sfc_alb=0.15, sza=15)
    # run_simulation(cfg, sfc_alb=0.1, sza=15)
    # run_simulation(cfg, sfc_alb=0.05, sza=15)
    # run_simulation(cfg, sfc_alb=0.025, sza=15)

    # run_simulation(cfg, sfc_alb=0.5, sza=75)
    # run_simulation(cfg, sfc_alb=0.4, sza=75)
    # run_simulation(cfg, sfc_alb=0.3, sza=75)
    # run_simulation(cfg, sfc_alb=0.25, sza=75)
    # run_simulation(cfg, sfc_alb=0.2, sza=75)
    # run_simulation(cfg, sfc_alb=0.15, sza=75)
    # run_simulation(cfg, sfc_alb=0.1, sza=75)
    # run_simulation(cfg, sfc_alb=0.05, sza=75)
    # run_simulation(cfg, sfc_alb=0.025, sza=75)







    



