#!/bin/env python
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=OCO2_test

import os
from pathlib import Path
import sys
import platform
import h5py
import numpy as np
from datetime import datetime
from scipy import stats
from er3t.pre.abs import abs_oco_h5
from er3t.pre.sfc import sfc_sat
from er3t.pre.pha import pha_mie_wc
from er3t.util.oco2 import oco2_std
from er3t.rtm.mca import mca_atm_1d, mca_atm_3d, mca_sfc_2d
from er3t.rtm.mca import mcarats_ng
from er3t.rtm.mca import mca_out_ng
from er3t.rtm.mca import mca_sca

from util.create_atm import create_oco_atm
from util.sat_download import satellite_download
from util.oco_cfg import grab_cfg, save_h5_info, check_h5_info
from util.abs_coeff import oco_abs
from util.oco_raw_collect import cdata_sat_raw
from util.oco_cloud import cdata_cld_ipa
from util.post_process import cdata_all
from util.oco_modis_650 import cal_mca_rad_650, modis_650_simulation_plot
from util.oco_util import path_dir, sat_tmp, timing, plot_mca_simulation
from util.oco_atm_atmmod import atm_atmmod
from util.oco_cld_sat import cld_sat


def cal_mca_rad_oco2(date, tag, sat, zpt_file, wavelength, cfg_info,
                     fname_atm_abs=None, 
                     scale_factor=1.0, 
                     fdir='tmp-data', solver='3D',
                     sfc_alb_abs=None, sza_abs=None, 
                     cld_manual=False, cot=None, cer=None, cth=None, aod550=None,
                     overwrite=True):

    """
    Calculate OCO2 radiance using cloud (MODIS level 1b) and surface properties (MOD09A1) from MODIS
    """

    Nphotons = float(cfg_info['oco_N_photons']) if sza_abs is None else 1e8


    # atm object
    # =================================================================================
    with h5py.File(zpt_file, 'r') as oco_zpt:
        levels = oco_zpt['h_edge'][...]
    fname_atm = '%s/atm.pk' % fdir
    atm0      = atm_atmmod(zpt_file=zpt_file, fname=fname_atm, overwrite=overwrite)
    # =================================================================================

    # abs object from OCO ABSCO, MET, CO2 prior files
    # =================================================================================
    fname_abs = '%s/abs.pk' % fdir
    abs0      = abs_oco_h5(wavelength=wavelength, fname=fname_abs, fname_h5=fname_atm_abs, atm_obj=atm0, overwrite=overwrite)
    # =================================================================================

    # mca_sfc object
    # =================================================================================
    with h5py.File(f'{sat.fdir_pre_data}/pre-data.h5', 'r') as f_pre_data:
        data = {'alb_2d': {'data': f_pre_data[f'oco/sfc/alb_{tag}_2d'][...], 'name': 'Surface albedo (lambertian)', 'units': 'N/A'},
                'lon_2d': {'data': f_pre_data['mod/sfc/lon'][...], 'name': 'Longitude', 'units': 'degrees'},
                'lat_2d': {'data': f_pre_data['mod/sfc/lat'][...], 'name': 'Latitude', 'units': 'degrees'}}
    
    if sfc_alb_abs is not None:
        print('sfc_alb_abs is not None')
        print('Simulated uniform sfc albedo: ', sfc_alb_abs)
        simulated_sfc_alb = sfc_alb_abs
        data['alb_2d']['data'][...] = simulated_sfc_alb
    else:
        print('sfc_alb_abs is None')
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

    # 3D-cld and 1D-AOD object
    # =================================================================================
    with h5py.File(f'{sat.fdir_pre_data}/pre-data.h5', 'r') as f_pre_data:
        data = {}
        data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=f_pre_data['lon'][...])
        data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=f_pre_data['lat'][...])
        data['rad_2d'] = dict(name='Gridded radiance'                , units='km'         , data=f_pre_data[f'mod/rad/rad_650'][...])
        if solver.lower() == 'ipa':
            suffix = 'ipa0'     # with wind correction only
        elif solver.lower() == '3d':
            suffix = 'ipa'      # with parallex and wind correction
        data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=f_pre_data[f'mod/cld/cot_{suffix}'][...])
        data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=f_pre_data[f'mod/cld/cer_{suffix}'][...])
        data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=f_pre_data[f'mod/cld/cth_{suffix}'][...])
        # =================================================================================

        # aod object
        # =================================================================================
        AOD_550_land_mean = f_pre_data['mod/aod/AOD_550_land_mean'][...]
        Angstrom_Exponent_land_mean = f_pre_data['mod/aod/Angstrom_Exponent_land_mean'][...]
        SSA_land_mean = f_pre_data['mod/aod/SSA_660_land_mean'][...]
        
        if aod550 is not None:
            AOD_550_land_mean = aod550
        # =================================================================================

        modl1b    =  sat_tmp(data)
        if cld_manual:
            modl1b.data['cth_2d']['data'][modl1b.data['cth_2d']['data']>0] = cth
            modl1b.data['cot_2d']['data'][modl1b.data['cot_2d']['data']>0] = cot
            modl1b.data['cer_2d']['data'][modl1b.data['cer_2d']['data']>0] = cer
        
        fname_cld = '%s/cld.pk' % fdir
        cth0 = modl1b.data['cth_2d']['data']
        cth0[cth0>10.0] = 10.0
        cgt0 = np.zeros_like(cth0)
        cgt0[cth0>0.0] = 1.0                  # all clouds have geometrical thickness of 1 km
        cgt0[cth0>4.0] = cth0[cth0>4.0]-3.0   # high clouds (cth>4km) has cloud base at 3 km
        cld0 = cld_sat(zpt_file=zpt_file, fname_atm=fname_atm, sat_info=sat,
                       sat_obj=modl1b, fname=fname_cld, cth=cth0, cgt=cgt0, dz=0.5,#np.unique(atm0.lay['thickness']['data'])[0],
                       overwrite=overwrite)
    # =================================================================================

    # mca_atm object
    # =================================================================================
    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir) # newly modified for phase function
    atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)

    if cfg_info['_aerosol'] == 'TRUE' and AOD_550_land_mean>0:
        # add homogeneous 1d mcarats "atmosphere", aerosol layer
        aod = AOD_550_land_mean*((wavelength/550)**(Angstrom_Exponent_land_mean*-1)) 
        ssa = SSA_land_mean # aerosol single scattering albedo
        cth_mode = stats.mode(cth0[np.logical_and(cth0>0, cth0<4)])
        print(f'aod {wavelength:.2f} nm mean:', aod)
        print('cth mode:', cth_mode.mode[0])
        asy    = float(cfg_info['asy']) # aerosol asymmetry parameter
        z_bot  = np.min(levels)         # altitude of layer bottom in km
        z_top  = cth_mode.mode[0]       # altitude of layer top in km
        aer_ext = aod / (z_top-z_bot) / 1000

        atm1d0.add_mca_1d_atm(ext1d=aer_ext, omg1d=ssa, apf1d=asy, z_bottom=z_bot, z_top=z_top)

    else:
        aod = 0

    atm_1ds = [atm1d0]
    atm_3ds = [atm3d0]

    # =================================================================================

    with h5py.File(f'{sat.fdir_pre_data}/pre-data.h5', 'r') as f_pre_data:
        sza = f_pre_data['oco/geo/sza'][...].mean()
        saa = f_pre_data['oco/geo/saa'][...].mean()
        vza = f_pre_data['oco/geo/vza'][...].mean()
        vaa = f_pre_data['oco/geo/vaa'][...].mean()

    if sza_abs is not None:
        sza = sza_abs
    
    print('sza:', np.nanmean(sza), 'avg sfc alb:', np.nanmean(simulated_sfc_alb), )
    # cpu number used
    if platform.system() in ['Windows', 'Darwin']:
        Ncpu=os.cpu_count()-1
    else:
        Ncpu=int(cfg_info['Ncpu'])

    
    if solver.lower()=='3d':
        # output filename
        output_file = f'{fdir}/mca-out-rad-oco2-{solver.lower()}_{wavelength:.4f}nm.h5'

        if (not os.path.isfile(output_file)) or (overwrite==True):
            # run mcarats
            temp_dir = f'{fdir}/{wavelength:.4f}nm/oco2/rad_{solver.lower()}'
            run = False if os.path.isdir(temp_dir) and overwrite==False else True
            mca0 = mcarats_ng(date=date,
                            atm_1ds=atm_1ds,
                            atm_3ds=atm_3ds,
                            surface_albedo=sfc_2d,
                            sca=sca, # newly added for phase function
                            Ng=int(abs0.Ng),
                            target='radiance',
                            solar_zenith_angle   = sza,
                            solar_azimuth_angle  = saa,
                            sensor_zenith_angle  = vza,
                            sensor_azimuth_angle = vaa,
                            fdir=temp_dir,
                            Nrun=3,
                            weights=abs0.coef['weight']['data'],
                            photons=Nphotons,
                            solver=solver,
                            Ncpu=Ncpu,
                            mp_mode='py',
                            overwrite=run)
            
            # mcarats output
            out0 = mca_out_ng(fname=output_file, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)
            oco_std0 = oco2_std(fnames=sat.fnames['oco_std'], extent=sat.extent)

            # plot
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            plot_mca_simulation(sat, modl1b, out0, oco_std0,
                                solver, fdir, cth, scale_factor, wavelength)
            # ------------------------------------------------------------------------------------------------------


    if solver.lower() == 'ipa':
        sfc0      = sfc_sat(sat_obj=mod43, fname=fname_sfc, extent=sat.extent, verbose=True, overwrite=False)
        sfc_2d    = mca_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/mca_sfc_2d.bin' % fdir, overwrite=overwrite)
        cld0    = cld_sat(fname=fname_cld, overwrite=False)
        cld0.lay['extinction']['data'][...] = 0.0
        atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % fdir)
        atm_3ds = [atm3d0]

        # output
        output_file = '%s/mca-out-rad-oco2-%s0_%.4fnm.h5' % (fdir, solver.lower(), wavelength)
        if (not os.path.isfile(output_file)) or (overwrite==True):
            # run mcarats
            temp_dir = f'{fdir}/{wavelength:.4f}nm/oco2/rad_{solver.lower()}0'
            run = False if os.path.isdir(temp_dir) and overwrite==False else True
            mca0 = mcarats_ng(date=date,
                              atm_1ds=atm_1ds,
                              atm_3ds=atm_3ds,
                              surface_albedo=sfc_2d,
                              Ng=abs0.Ng,
                              target='radiance',
                              solar_zenith_angle   = sza,
                              solar_azimuth_angle  = saa,
                              sensor_zenith_angle  = vza,
                              sensor_azimuth_angle = vaa,
                              fdir=temp_dir,
                              Nrun=3,
                              weights=abs0.coef['weight']['data'],
                              photons=Nphotons,
                              solver=solver,
                              Ncpu=Ncpu,
                              mp_mode='py',
                              overwrite=run,
                              )

            # mcarats output
            out0 = mca_out_ng(fname=output_file, mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)
            oco_std0 = oco2_std(fnames=sat.fnames['oco_std'], extent=sat.extent)

            # plot
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            plot_mca_simulation(sat, modl1b, out0, oco_std0,
                                'ipa0', fdir, cth, scale_factor, wavelength)
            # ------------------------------------------------------------------------------------------------------
    return simulated_sfc_alb, sza, aod

@timing
def preprocess(cfg_info):
    # define date and region to study
    # ===============================================================
    date = datetime(int(cfg_info['date'][:4]),    # year
                    int(cfg_info['date'][4:6]),   # month
                    int(cfg_info['date'][6:]))    # day
    extent = [float(loc) + offset for loc, offset in zip(cfg_info['subdomain'], [-0.15, 0.15, -0.15, 0.15])]
    extent_analysis = [float(loc) for loc in cfg_info['subdomain']]
    name_tag = f"{cfg_info['cfg_name']}_{date.strftime('%Y%m%d')}"
    # ===============================================================

    # create data/name_tag directory if it does not exist
    # ===============================================================
    fdir_data = path_dir('/'.join(['data', name_tag]))
    # ===============================================================

    # download satellite data based on given date and region
    # ===============================================================
    fname_sat = '/'.join([fdir_data, 'sat.pk'])
    sat0 = satellite_download(date=date, 
                              fdir_out=cfg_info['path_sat_data'], 
                              fdir_pre_data=fdir_data,
                              extent=extent,
                              extent_analysis=extent_analysis,
                              fname=fname_sat, overwrite=False)
    # ===============================================================
    if not ('l2' in cfg_info.keys()):
        oco_data_dict = {'l2': 'oco_std',
                         'met': 'oco_met',
                         'l1b': 'oco_l1b',
                         'lt': 'oco_lite',
                         'dia': 'oco_dia',
                         'imap': 'oco_imap',
                         'co2prior': 'oco_co2prior'}
        for key, value in oco_data_dict.items():
            save_h5_info(cfg_info['cfg_path'], key, sat0.fnames[value][0].split('/')[-1])
    save_h5_info(cfg_info['cfg_path'], 'png', sat0.fnames['mod_rgb'][0].split('/')[-1])
    # create tmp-data/{name_tag} directory if it does not exist
    # ===============================================================
    fdir_cot_tmp = path_dir('tmp-data/%s/cot' % (name_tag))
    # ===============================================================

    # create atmosphere based on OCO-Met and CO2_prior
    # ===============================================================
    zpt_file = os.path.abspath('/'.join([fdir_data, 'zpt.h5']))
    if not os.path.isfile(zpt_file):
        create_oco_atm(sat=sat0, o2mix=float(cfg_info['o2mix']), output=zpt_file)
    # ===============================================================

    # read out wavelength information from absorption file
    # ===============================================================
    nx = int(cfg_info['nx'])
    for iband, band_tag in enumerate(['o2a', 'wco2', 'sco2']):
        fname_abs = f'{fdir_data}/atm_abs_{band_tag}_{(nx+1):d}.h5'
        if not os.path.isfile(fname_abs):
            oco_abs(cfg, sat0, 
                    zpt_file=zpt_file, iband=iband, 
                    nx=nx, 
                    Trn_min=float(cfg_info['Trn_min']), 
                    pathout=fdir_data,
                    reextract=True, plot=True)
    
    if not os.path.isfile(f'{fdir_data}/pre-data.h5') :
        cdata_sat_raw(sat0=sat０, dx=250, dy=250, overwrite=True, plot=True)
    cdata_cld_ipa(sat０, fdir_cot_tmp, zpt_file, cfg_info, plot=True)
    # ===============================================================
    return date, extent, name_tag, fdir_data, sat0, zpt_file

@timing
def run_case_modis_650(cfg_info, preprocess_info):
    # Get information from cfg_info
    # ======================================================================
    name_tag = preprocess_info[2]
    sat0     = preprocess_info[4]
    zpt_file = preprocess_info[5]
    # ======================================================================

    # run calculations for 650 nm
    # ======================================================================
    fdir_tmp_650 = path_dir(f'tmp-data/{name_tag}/modis_650')
    for solver in ['IPA', '3D']:
        cal_mca_rad_650(sat0, zpt_file, 650, cfg_info, fdir=fdir_tmp_650, solver=solver,
                        overwrite=True, case_name_tag=name_tag)
        modis_650_simulation_plot(sat0, cfg_info, case_name_tag=name_tag, fdir=fdir_tmp_650,
                                   solver=solver, wvl=650, plot=True)
    # ======================================================================

@timing
def run_case_oco(band_tag, cfg_info, preprocess_info, sfc_alb=None, sza=None,
                 cld_manual=False, cot=None, cer=None, cth=None, aod550=None):
    # Get information from cfg_info
    # ======================================================================
    date      = preprocess_info[0]
    name_tag  = preprocess_info[2]
    fdir_data = preprocess_info[3]
    sat0      = preprocess_info[4]
    zpt_file  = preprocess_info[5]
    zpt_file = '../glint/data/20151201_ocean_1_20151201/zpt.h5'
    # ======================================================================
    if sfc_alb != None and cld_manual == False:
        fdir_tmp = path_dir(f'tmp-data/{name_tag}_alb_{sfc_alb:.3f}_sza_{sza:.1f}/{band_tag}')
    elif sfc_alb != None and cld_manual == True:
        fdir_tmp = path_dir(f'tmp-data/{name_tag}_alb_{sfc_alb:.3f}_sza_{sza:.1f}_cth_{cth:.1f}_cot_{cot:.0f}_cer_{cer:.0f}/{band_tag}')
    else:
        fdir_tmp = path_dir(f'tmp-data/{name_tag}/{band_tag}')
    
    # read out wavelength information from absorption file
    # ===============================================================
    fname_abs = f'{fdir_data}/atm_abs_{band_tag}_{(int(cfg_info["nx"])+1):d}.h5'
    with h5py.File(fname_abs, 'r') as f:
        wvls = f['lamx'][...]*1000.0 # micron to nm
    # ===============================================================
    
    # run calculations for each wavelength
    # ===============================================================
    Nphotons = float(cfg_info['oco_N_photons']) if sza is None else 1e8
    for wavelength in wvls:
        for solver in ['IPA', '3D']:
            alb_sim, sza_sim, aod_sim = cal_mca_rad_oco2(date, band_tag, sat0, zpt_file, wavelength, 
                                            cfg_info=cfg_info,
                                            fname_atm_abs=fname_abs, scale_factor=1.0, 
                                            fdir=fdir_tmp, solver=solver, 
                                            sfc_alb_abs=sfc_alb, sza_abs=sza,
                                            cld_manual=cld_manual, cot=cot, cer=cer, cth=cth, aod550=aod550,
                                            overwrite=True)
    # ===============================================================

    # post-processing - combine the all the calculations into one dataset
    # ===============================================================
    collect_data = cdata_all(date, band_tag, fdir_tmp, fname_abs, sat0, sfc_alb=alb_sim, sza=sza_sim, fdir_out=fdir_data, 
                             aod_550=aod_sim,
                             cld_manual=cld_manual, cot=cot, cer=cer,)
    # ===============================================================
    
    return collect_data
    

def run_simulation(cfg, sfc_alb=None, sza=None, cld_manual=False, cot=None, cer=None, cth=None, aod550=None):
    cfg_info = grab_cfg(cfg)
    preprocess_info = preprocess(cfg_info)
    # sys.exit()
    # run_case_modis_650(cfg_info, preprocess_info)

    if 1:#not check_h5_info(cfg, 'o2'): 
        o2_h5 = run_case_oco('o2a', cfg_info, preprocess_info, sfc_alb=sfc_alb, sza=sza,
                             cld_manual=cld_manual, cot=cot, cer=cer, cth=cth, aod550=aod550)
        save_h5_info(cfg, 'o2', o2_h5)

    if 1:#not check_h5_info(cfg, 'wco2'):
        wco2_h5 = run_case_oco('wco2', cfg_info, preprocess_info, sfc_alb=sfc_alb, sza=sza,
                             cld_manual=cld_manual, cot=cot, cer=cer, cth=cth, aod550=aod550)
        save_h5_info(cfg, 'wco2', wco2_h5)

    if 1:#not check_h5_info(cfg, 'sco2'):
        sco2_h5 = run_case_oco('sco2', cfg_info, preprocess_info, sfc_alb=sfc_alb, sza=sza,
                             cld_manual=cld_manual, cot=cot, cer=cer, cth=cth, aod550=aod550)
        save_h5_info(cfg, 'sco2', sco2_h5)

if __name__ == '__main__':
    
    cfg = 'cfg/20181018_central_asia_zpt_test.csv'
    # cfg = 'cfg/20151219_north_italy_470cloud_test.csv'
    #cfg = 'cfg/20190621_australia-2-470cloud_aod.csv'
    #cfg = 'cfg/20161023_north_france_test.csv'
    # cfg = 'cfg/20190209_dryden_470cloud.csv'
    # cfg = 'cfg/20170605_amazon_2.csv'
    # cfg = 'cfg/20190621_australia_2.csv'
    # cfg = 'cfg/20170721_australia_1.csv'
    # cfg = 'cfg/20150622_amazon.csv'
    print(cfg)
    # run_simulation(cfg, sfc_alb=0.5, sza=45, cld_manual=True, cot=5, cer=25, cth=5, aod550=0) #done
    
    # run_simulation(cfg, sfc_alb=0.05, sza=45, cld_manual=True, cot=5, cer=25, cth=5, aod550=0) #done
    # run_simulation(cfg, sfc_alb=0.1, sza=45, cld_manual=True, cot=5, cer=25, cth=5, aod550=0) #done
    # run_simulation(cfg, sfc_alb=0.3, sza=45, cld_manual=True, cot=5, cer=25, cth=5, aod550=0) #donerun_simulation(cfg, sfc_alb=0.5, sza=45, cld_manual=True, cot=5, cer=25, cth=5, aod550=0) #done

    # run_simulation(cfg, sfc_alb=0.3, sza=15, cld_manual=True, cot=5, cer=25, cth=5, aod550=0) #donerun_simulation(cfg, sfc_alb=0.5, sza=45, cld_manual=True, cot=5, cer=25, cth=5, aod550=0) #done
    # run_simulation(cfg, sfc_alb=0.3, sza=30, cld_manual=True, cot=5, cer=25, cth=5, aod550=0) #donerun_simulation(cfg, sfc_alb=0.5, sza=45, cld_manual=True, cot=5, cer=25, cth=5, aod550=0) #done
    # run_simulation(cfg, sfc_alb=0.3, sza=60, cld_manual=True, cot=5, cer=25, cth=5, aod550=0) #donerun_simulation(cfg, sfc_alb=0.5, sza=45, cld_manual=True, cot=5, cer=25, cth=5, aod550=0) #done
    # run_simulation(cfg, sfc_alb=0.3, sza=75, cld_manual=True, cot=5, cer=25, cth=5, aod550=0) #donerun_simulation(cfg, sfc_alb=0.5, sza=45, cld_manual=True, cot=5, cer=25, cth=5, aod550=0) #done

    run_simulation(cfg, sfc_alb=0.3, sza=45, cld_manual=True, cot=5, cer=25, cth=6, aod550=0) 
    run_simulation(cfg, sfc_alb=0.3, sza=45, cld_manual=True, cot=5, cer=25, cth=8, aod550=0) 
    run_simulation(cfg, sfc_alb=0.3, sza=45, cld_manual=True, cot=5, cer=25, cth=10, aod550=0) 

    run_simulation(cfg, sfc_alb=0.3, sza=45, cld_manual=True, cot=10, cer=25, cth=5, aod550=0) 
    run_simulation(cfg, sfc_alb=0.3, sza=45, cld_manual=True, cot=15, cer=25, cth=5, aod550=0) 
    run_simulation(cfg, sfc_alb=0.3, sza=45, cld_manual=True, cot=20, cer=25, cth=5, aod550=0) 


    # run_simulation(cfg, sfc_alb=0.5, sza=45)








    



