#!/bin/env python
#SBATCH --partition=amilan
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=OCO2_test

import os
import sys
import h5py
import numpy as np
import pandas as pd
import datetime
from scipy import stats as st
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_oco_h5
from er3t.pre.cld import cld_sat
from er3t.pre.sfc import sfc_sat
from er3t.pre.pha import pha_mie_wc # newly added for phase function
from er3t.util.oco2 import oco2_std
from er3t.rtm.mca import mca_atm_1d, mca_atm_3d, mca_sfc_2d
from er3t.rtm.mca import mcarats_ng
from er3t.rtm.mca import mca_out_ng
from er3t.rtm.mca import mca_sca # newly added for phase function

from oco_subroutine.oco_create_atm import create_oco_atm
from oco_subroutine.oco_satellite import satellite_download
from oco_subroutine.oco_cfg import grab_cfg, save_h5_info
from oco_subroutine.oco_abs_snd_sat import oco_abs
from oco_subroutine.oco_raw_collect import cdata_sat_raw
from oco_subroutine.oco_cloud import cdata_cld_ipa
from oco_subroutine.oco_post_process import cdata_all
from oco_subroutine.oco_modis_650 import cal_mca_rad_650, modis_650_simulation_plot

import timeit

class sat_tmp:

    def __init__(self, data):

        self.data = data


def cal_mca_rad_oco2(date, tag, sat, zpt_file, wavelength, fname_idl=None, cth=None, 
                     photons=1e6, scale_factor=1.0, 
                     fdir='tmp-data', solver='3D', 
                     sfc_alb_abs=None, sza_abs=None, aod_550=None,
                     overwrite=True):

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
    with h5py.File(f'{sat.fdir_out}/pre-data.h5', 'r') as f_pre_data:
        data = {
            'alb_2d': {'data': f_pre_data[f'oco/sfc/alb_{tag}_2d'][...], 'name': 'Surface albedo', 'units': 'N/A'},
            'lon_2d': {'data': f_pre_data['mod/sfc/lon'][...], 'name': 'Longitude', 'units': 'degrees'},
            'lat_2d': {'data': f_pre_data['mod/sfc/lat'][...], 'name': 'Latitude', 'units': 'degrees'}
        }

    #"""
    if sfc_alb_abs is not None:
        # avg_sfc_alb = np.nanmean(data['alb_2d']['data'])
        # print('Average sfc albedo: ', avg_sfc_alb)
        simulated_sfc_alb = sfc_alb_abs #if sfc_alb_abs is not None else avg_sfc_alb
        data['alb_2d']['data'][...] = simulated_sfc_alb
    else:
        avg_sfc_alb = np.nanmean(data['alb_2d']['data'])
        print('Average sfc albedo: ', avg_sfc_alb)
        simulated_sfc_alb = avg_sfc_alb
        
    mod43     = sat_tmp(data)
    fname_sfc = '%s/sfc.pk' % fdir
    sfc0      = sfc_sat(sat_obj=mod43, fname=fname_sfc, extent=sat.extent, verbose=True, overwrite=overwrite)
    sfc_2d    = mca_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/mca_sfc_2d.bin' % fdir, overwrite=overwrite)
    # =================================================================================


    # mca_sca object (newly added for phase function)
    # =================================================================================
    pha0 = pha_mie_wc(wavelength=wavelength, overwrite=overwrite)
    sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)
    # =================================================================================


    # cld object
    # =================================================================================
    with h5py.File(f'{sat.fdir_out}/pre-data.h5', 'r') as f_pre_data:
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
        if aod_550 is not None:
            AOD_550_land_mean = aod_550
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

    atm1d0.add_mca_1d_atm(ext1d=aer_ext, omg1d=ssa, apf1d=asy, z_bottom=z_bot, z_top=z_top)
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

    

    with h5py.File(sat.fnames['oco_l1b'][0], 'r') as f_oco_l1b:
        lon_oco_l1b = f_oco_l1b['SoundingGeometry/sounding_longitude'][...]
        lat_oco_l1b = f_oco_l1b['SoundingGeometry/sounding_latitude'][...]
        logic = (lon_oco_l1b>=sat.extent[0]) & (lon_oco_l1b<=sat.extent[1]) & (lat_oco_l1b>=sat.extent[2]) & (lat_oco_l1b<=sat.extent[3])
        sza = f_oco_l1b['SoundingGeometry/sounding_solar_zenith'][...][logic].mean()
        saa = f_oco_l1b['SoundingGeometry/sounding_solar_azimuth'][...][logic].mean()
        vza = f_oco_l1b['SoundingGeometry/sounding_zenith'][...][logic].mean()
        vaa = f_oco_l1b['SoundingGeometry/sounding_azimuth'][...][logic].mean()

    if sza_abs != None:
        sza = sza_abs

    if simulated_sfc_alb <= 0.2:
        photons = photons*2

    # run mcarats
    run = False if os.path.isdir('%s/%.4fnm/oco2/rad_%s' % (fdir, wavelength, solver.lower())) and overwrite==False else True
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
            overwrite=run
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
        run = False if os.path.isdir('%s/%.4fnm/oco2/rad_%s' % (fdir, wavelength, solver.lower())) and overwrite==False else True
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
                overwrite=run,
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
    return simulated_sfc_alb, sza, AOD_550_land_mean

def preprocess(cfg_info, sfc_alb=None, sza=None):
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
    print(f'simulation extent: {extent}')
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
        oco_data_dict = {'l2': 'oco_std',
                         'met': 'oco_met',
                         'l1b': 'oco_l1b',
                         'lt': 'oco_lite',
                         'dia': 'oco_dia',
                         'imap': 'oco_imap',
                         'co2prior': 'oco_co2prior'}
        for key in oco_data_dict.keys():
            save_h5_info(cfg_info['cfg_path'], key, sat0.fnames[oco_data_dict[key]][0].split('/')[-1])
    # create tmp-data/australia_YYYYMMDD directory if it does not exist
    # ===============================================================
    fdir_cot_tmp = os.path.abspath('tmp-data/%s/cot' % (name_tag))
    if not os.path.exists(fdir_cot_tmp):
        os.makedirs(fdir_cot_tmp)
    # ===============================================================

    # create atmosphere based on OCO-Met and CO2_prior
    # ===============================================================
    zpt_file = os.path.abspath(f'{fdir_data}/zpt.h5')
    if not os.path.isfile(zpt_file):
        create_oco_atm(sat=sat0, o2mix=0.20935, output=zpt_file)
    # ===============================================================


    # read out wavelength information from absorption file
    # ===============================================================
    nx = int(cfg_info['nx'])
    iband_dict = {'o2a':0, 'wco2':1, 'sco2':2,}
    Trn_min = float(cfg_info['Trn_min'])
    for band_tag in ['o2a', 'wco2', 'sco2']:
        fname_abs = f'{fdir_data}/atm_abs_{band_tag}_{(nx+1):d}.h5'
        if not os.path.isfile(fname_abs):
            oco_abs(cfg, zpt_file=zpt_file, iband=iband_dict[band_tag], nx=nx, Trn_min=Trn_min, pathout=fdir_data, reextract=False, plot=True)

    if not os.path.isfile(f'{sat０.fdir_out}/pre-data.h5') :
        cdata_sat_raw(sat0=sat０, overwrite=True, plot=True)

    # ===============================================================
    cdata_cld_ipa(sat０, fdir_cot_tmp, zpt_file, ref_threshold=ref_threshold, photons=1e7, plot=True)
    # ===============================================================



def run_case(band_tag, cfg_info, sfc_alb=None, sza=None, aod_550=None):

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
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(date=date, fdir_out=fdir_data, extent=extent, fname=fname_sat, overwrite=False)
    
    if sfc_alb != None:
        fdir_tmp = os.path.abspath('tmp-data/%s_alb_%.3f_sza_%.1f_aod550_%.2f/%s' % (name_tag, sfc_alb, sza, aod_550, band_tag))
    else:
        fdir_tmp = os.path.abspath('tmp-data/%s/%s' % (name_tag, band_tag))
    print(fdir_tmp)
    if not os.path.exists(fdir_tmp):
        os.makedirs(fdir_tmp)
    zpt_file = os.path.abspath(f'{fdir_data}/zpt.h5')


    # read out wavelength information from absorption file
    # ===============================================================
    fname_abs = f'{fdir_data}/atm_abs_{band_tag}_{(int(cfg_info["nx"])+1):d}.h5'
    with h5py.File(fname_abs, 'r') as f:
        wvls = f['lamx'][...]*1000.0
    print(wvls)
    # ===============================================================

    """
    # run calculations for 650 nm
    # ===============================================================
    fdir_tmp_650 = os.path.abspath('tmp-data/%s/%s' % (name_tag, 'modis_650'))
    if not os.path.exists(fdir_tmp_650):
        os.makedirs(fdir_tmp_650)
    for solver in ['IPA', '3D']:
        cal_mca_rad_650(sat0, zpt_file, 650, fdir=fdir_tmp_650, solver=solver, overwrite=False, case_name_tag=name_tag, photons=1e9)
        modis_650_simulation_plot(extent, case_name_tag=name_tag, fdir=fdir_tmp_650, solver=solver, wvl=650, ref_threshold=ref_threshold, plot=True)
    #"""

    #"""
    # run calculations for each wavelength
    # ===============================================================
    for wavelength in wvls:
        for solver in ['IPA', '3D']:
            simulated_sfc_alb, sza = cal_mca_rad_oco2(date, band_tag, sat0, zpt_file, wavelength,
                                                      fname_idl=fname_abs, cth=None, scale_factor=1.0, 
                                                      fdir=fdir_tmp, solver=solver, 
                                                      sfc_alb_abs=sfc_alb, sza_abs=sza, aod_550=aod_550,
                                                      overwrite=True, photons=2e8)
    # ===============================================================
    #"""

    #"""
    # post-processing - combine the all the calculations into one dataset
    # ===============================================================
    return cdata_all(date, band_tag, fdir_tmp, fname_abs, sat0, simulated_sfc_alb, sza)
    # ===============================================================
    #"""


def run_simulation(cfg, sfc_alb=None, sza=None, aod_550=None):
    cfg_info = grab_cfg(cfg)
    preprocess(cfg_info, sfc_alb=sfc_alb, sza=sza)
    #"""
    if 1:#not check_h5_info(cfg, 'o2'):
        starttime = timeit.default_timer()
        o2_h5 = run_case('o2a', cfg_info, sfc_alb=sfc_alb, sza=sza, aod_550=aod_550)
        save_h5_info(cfg, 'o2', o2_h5)
        endtime = timeit.default_timer()
        print('O2A band duration:',(endtime-starttime)/3600.,' h')
        # time.sleep(120)
    #""" 
    
    #"""
    if 1:#not check_h5_info(cfg, 'wco2'):
        starttime = timeit.default_timer()
        wco2_h5 = run_case('wco2', cfg_info, sfc_alb=sfc_alb, sza=sza, aod_550=aod_550)
        save_h5_info(cfg, 'wco2', wco2_h5)
        endtime = timeit.default_timer()
        print('WCO2 band duration:',(endtime-starttime)/3600.,' h')
    #"""
    #""""
    #time.sleep(120)
    if 1:#not check_h5_info(cfg, 'sco2'):
        starttime = timeit.default_timer()
        sco2_h5 = run_case('sco2', cfg_info, sfc_alb=sfc_alb, sza=sza, aod_550=aod_550)
        save_h5_info(cfg, 'sco2', sco2_h5)
        endtime = timeit.default_timer()
        print('SCO2 band duration:',(endtime-starttime)/3600.,' h')
    #"""

if __name__ == '__main__':
    
    cfg = 'cfg/20181018_central_asia_2_470cloud_test2.csv'
    # cfg = 'cfg/20151219_north_italy_470cloud_test.csv'
    #cfg = 'cfg/20190621_australia-2-470cloud_aod.csv'
    # cfg = 'cfg/20190209_dryden_470cloud.csv'
    print(cfg)
    #run_simulation(cfg) #done

    run_simulation(cfg, sfc_alb=0.5, sza=45, aod_550=0.1)
    # run_simulation(cfg, sfc_alb=0.4, sza=45)
    # run_simulation(cfg, sfc_alb=0.3, sza=45) #done
    # run_simulation(cfg, sfc_alb=0.25, sza=45)
    # run_simulation(cfg, sfc_alb=0.2, sza=45)
    # run_simulation(cfg, sfc_alb=0.15, sza=45)
    # run_simulation(cfg, sfc_alb=0.1, sza=45)
    # run_simulation(cfg, sfc_alb=0.05, sza=45)
    # run_simulation(cfg, sfc_alb=0.025, sza=45)

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
    
    # run_simulation(cfg, sfc_alb=0.25, sza=60)

    # run_simulation(cfg, sfc_alb=0.4, sza=30)
    
    # run_simulation(cfg, sfc_alb=0.4, sza=60)

    # run_simulation(cfg, sfc_alb=0.025, sza=30)
    
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







    



