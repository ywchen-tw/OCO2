import sys
sys.path.append('/Users/yuch8913/miniforge3/envs/er3t_env/lib/python3.8/site-packages')
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
from glob import glob
import numpy as np
from sys import exit as ext
import copy
from bisect import bisect_left
from oco_post_class_ywc import OCOSIM
from matplotlib import cm
from scipy.interpolate import interpn
from scipy import stats
from scipy.ndimage import uniform_filter
import geopy.distance
import xarray as xr
import seaborn as sns
from tool_code import *
import os, pickle 

from matplotlib import font_manager

font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

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
    return result

def output_h5_info(cfg, index):
    """
    Check whether the output h5 name is saved in cfg file
    """
    try: 
        cfg_file = grab_cfg(cfg)
    except OSError as err:
        print('{} not exists!'.format(cfg))
        return False
    if index in cfg_file.keys():
        if cfg_file[index][-2:] == 'h5':
            return cfg_file[index]
    else:
        return False

def near_rad_calc(OCO_class):
    rad_mca_ipa0 = np.zeros((OCO_class.lat.shape[0], OCO_class.lat.shape[1], OCO_class.lam.size), dtype=np.float64)
    rad_mca_ipa  = np.zeros((OCO_class.lat.shape[0], OCO_class.lat.shape[1], OCO_class.lam.size), dtype=np.float64)
    rad_mca_3d   = np.zeros((OCO_class.lat.shape[0], OCO_class.lat.shape[1], OCO_class.lam.size), dtype=np.float64)

    rad_mca_ipa0_std = np.zeros((OCO_class.lat.shape[0], OCO_class.lat.shape[1], OCO_class.lam.size), dtype=np.float64)
    rad_mca_ipa_std  = np.zeros((OCO_class.lat.shape[0], OCO_class.lat.shape[1], OCO_class.lam.size), dtype=np.float64)
    rad_mca_3d_std   = np.zeros((OCO_class.lat.shape[0], OCO_class.lat.shape[1], OCO_class.lam.size), dtype=np.float64)

    rad_mca_ipa0[...] = np.nan
    rad_mca_ipa[...] = np.nan
    rad_mca_3d[...] = np.nan

    rad_mca_ipa0_std[...] = np.nan
    rad_mca_ipa_std[...] = np.nan
    rad_mca_3d_std[...] = np.nan

    rad_mca_ipa0_25 = rad_mca_ipa0.copy()
    rad_mca_ipa_25 = rad_mca_ipa0.copy()
    rad_mca_3d_25 = rad_mca_ipa0.copy()

    rad_mca_ipa0_std_25 = rad_mca_ipa0.copy()
    rad_mca_ipa_std_25 = rad_mca_ipa0.copy()
    rad_mca_3d_std_25 = rad_mca_ipa0.copy()
    count = 0
    for i in range(OCO_class.lat.shape[0]):
        for j in range(OCO_class.lat.shape[1]):
            if 1:#~ np.isnan(OCO_class.co2[i, j]):
                lon0 = OCO_class.lon[i, j]
                lat0 = OCO_class.lat[i, j]      
                index_lon = np.argmin(np.abs(OCO_class.lon2d[:, 0]-lon0))
                index_lat = np.argmin(np.abs(OCO_class.lat2d[0, :]-lat0))

                center = (lat0, lon0)
                tmp_clr, tmp_c1d, tmp_c3d = [], [], []
                tmp_clrs, tmp_c1ds, tmp_c3ds = [], [], []
                test_range = (np.arange(-4, 4+1))
                for x in test_range:
                    for y in test_range:
                        try:
                            interest_loc = (OCO_class.lat2d[0, index_lat+y], OCO_class.lon2d[index_lon+x, 0])
                            if geopy.distance.distance(center, interest_loc).km <= 0.5:

                                tmp_clr.append(OCO_class.rad_clr[index_lon+x, index_lat+y])
                                tmp_c1d.append(OCO_class.rad_c1d[index_lon+x, index_lat+y])
                                tmp_c3d.append(OCO_class.rad_c3d[index_lon+x, index_lat+y])
                                tmp_clrs.append(OCO_class.rad_clrs[index_lon+x, index_lat+y])
                                tmp_c1ds.append(OCO_class.rad_c1ds[index_lon+x, index_lat+y])
                                tmp_c3ds.append(OCO_class.rad_c3ds[index_lon+x, index_lat+y])
                                count += 1
                        except:
                            None
                
                rad_mca_ipa0[i, j, :] = np.nanmean(np.array(tmp_clr), axis=0)
                rad_mca_ipa[i, j, :]  = np.nanmean(np.array(tmp_c1d), axis=0)
                rad_mca_3d[i, j, :]   = np.nanmean(np.array(tmp_c3d), axis=0)

                rad_mca_ipa0_std[i, j, :] = (np.nanstd(np.array(tmp_clr), axis=0))
                rad_mca_ipa_std[i, j, :]  = (np.nanstd(np.array(tmp_c1d), axis=0))
                rad_mca_3d_std[i, j, :]   = (np.nanstd(np.array(tmp_c3d), axis=0))


                rad_mca_ipa0_25[i, j, :] = np.nanmean(OCO_class.rad_clr[index_lon-2:index_lon+3, 
                                                                 index_lat-2:index_lat+3], 
                                                      axis=(0, 1))
                rad_mca_ipa_25[i, j, :]  = np.nanmean(OCO_class.rad_c1d[index_lon-2:index_lon+3, 
                                                                 index_lat-2:index_lat+3], 
                                                      axis=(0, 1))
                rad_mca_3d_25[i, j, :]   = np.nanmean(OCO_class.rad_c3d[index_lon-2:index_lon+3, 
                                                                 index_lat-2:index_lat+3], 
                                                      axis=(0, 1))
                
                rad_mca_ipa0_std_25[i, j, :] = (np.nanstd(OCO_class.rad_clr[index_lon-2:index_lon+3,
                                                                      index_lat-2:index_lat+3],
                                                          axis=(0, 1)))
                rad_mca_ipa_std_25[i, j, :]  = (np.nanstd(OCO_class.rad_c1d[index_lon-2:index_lon+3,
                                                                      index_lat-2:index_lat+3],
                                                          axis=(0, 1)))
                rad_mca_3d_std_25[i, j, :]   = (np.nanstd(OCO_class.rad_c3d[index_lon-2:index_lon+3,
                                                                      index_lat-2:index_lat+3],
                                                          axis=(0, 1)))
    print(count)
    OCO_class.rad_1km_clr = rad_mca_ipa0
    OCO_class.rad_1km_c1d = rad_mca_ipa
    OCO_class.rad_1km_c3d = rad_mca_3d

    OCO_class.rad_1km_clrs = rad_mca_ipa0_std
    OCO_class.rad_1km_c1ds = rad_mca_ipa_std
    OCO_class.rad_1km_c3ds = rad_mca_3d_std

    OCO_class.rad_25p_clr = rad_mca_ipa0_25
    OCO_class.rad_25p_c1d = rad_mca_ipa_25
    OCO_class.rad_25p_c3d = rad_mca_3d_25

    OCO_class.rad_25p_clrs = rad_mca_ipa0_std_25
    OCO_class.rad_25p_c1ds = rad_mca_ipa_std_25
    OCO_class.rad_25p_c3ds = rad_mca_3d_std_25
    
    OCO_class.sl_1km  = (OCO_class.rad_1km_c3d-OCO_class.rad_1km_clr) / OCO_class.rad_1km_clr        # S_lamda
    OCO_class.sls_1km = (OCO_class.rad_1km_c3ds/OCO_class.rad_1km_clr + OCO_class.rad_1km_clrs/OCO_class.rad_1km_clr)
    OCO_class.sl_25p  = (OCO_class.rad_25p_c3d-OCO_class.rad_25p_clr) / OCO_class.rad_25p_clr        # S_lamda
    OCO_class.sls_25p = (OCO_class.rad_25p_c3ds/OCO_class.rad_25p_clr + OCO_class.rad_25p_clrs/OCO_class.rad_25p_clr)


    

def near_rad_calc_all(OCO_class):
    rad_mca_ipa0_5 = np.zeros((OCO_class.lat2d.shape[0], OCO_class.lat2d.shape[1], OCO_class.lam.size), dtype=np.float64)
    rad_mca_ipa_5  = np.zeros((OCO_class.lat2d.shape[0], OCO_class.lat2d.shape[1], OCO_class.lam.size), dtype=np.float64)
    rad_mca_3d_5   = np.zeros((OCO_class.lat2d.shape[0], OCO_class.lat2d.shape[1], OCO_class.lam.size), dtype=np.float64)

    rad_mca_ipa0_5_std = np.zeros((OCO_class.lat2d.shape[0], OCO_class.lat2d.shape[1], OCO_class.lam.size), dtype=np.float64)
    rad_mca_ipa_5_std  = np.zeros((OCO_class.lat2d.shape[0], OCO_class.lat2d.shape[1], OCO_class.lam.size), dtype=np.float64)
    rad_mca_3d_5_std   = np.zeros((OCO_class.lat2d.shape[0], OCO_class.lat2d.shape[1], OCO_class.lam.size), dtype=np.float64)

    print(rad_mca_ipa0_5.shape)
    rad_mca_ipa0_5[...] = np.nan
    rad_mca_ipa_5[...] = np.nan
    rad_mca_3d_5[...] = np.nan

    rad_mca_ipa0_5_std[...] = np.nan
    rad_mca_ipa_5_std[...] = np.nan
    rad_mca_3d_5_std[...] = np.nan

    rad_mca_ipa0_9 = rad_mca_ipa0_5.copy()
    rad_mca_ipa_9 = rad_mca_ipa0_5.copy()
    rad_mca_3d_9 = rad_mca_ipa0_5.copy()
    rad_mca_ipa0_9_std = rad_mca_ipa0_5.copy()
    rad_mca_ipa_9_std = rad_mca_ipa0_5.copy()
    rad_mca_3d_9_std = rad_mca_ipa0_5.copy()

    rad_mca_ipa0_13 = rad_mca_ipa0_5.copy()
    rad_mca_ipa_13 = rad_mca_ipa0_5.copy()
    rad_mca_3d_13 = rad_mca_ipa0_5.copy()
    rad_mca_ipa0_13_std = rad_mca_ipa0_5.copy()
    rad_mca_ipa_13_std = rad_mca_ipa0_5.copy()
    rad_mca_3d_13_std = rad_mca_ipa0_5.copy()

    rad_mca_ipa0_41 = rad_mca_ipa0_5.copy()
    rad_mca_ipa_41 = rad_mca_ipa0_5.copy()
    rad_mca_3d_41 = rad_mca_ipa0_5.copy()
    rad_mca_ipa0_41_std = rad_mca_ipa0_5.copy()
    rad_mca_ipa_41_std = rad_mca_ipa0_5.copy()
    rad_mca_3d_41_std = rad_mca_ipa0_5.copy()



    rad_mca_ipa0_5, rad_mca_ipa_5, rad_mca_3d_5,\
    rad_mca_ipa0_5_std, rad_mca_ipa_5_std, rad_mca_3d_5_std = coarsening(OCO_class, size=5)
    
    rad_mca_ipa0_9, rad_mca_ipa_9, rad_mca_3d_9,\
    rad_mca_ipa0_9_std, rad_mca_ipa_9_std, rad_mca_3d_9_std, = coarsening(OCO_class, size=9)
    
    rad_mca_ipa0_13, rad_mca_ipa_13, rad_mca_3d_13,\
    rad_mca_ipa0_13_std, rad_mca_ipa_13_std, rad_mca_3d_13_std = coarsening(OCO_class, size=13)
    
    rad_mca_ipa0_41, rad_mca_ipa_41, rad_mca_3d_41,\
    rad_mca_ipa0_41_std, rad_mca_ipa_41_std, rad_mca_3d_41_std = coarsening(OCO_class, size=41)

    OCO_class.rad_clr_5 = rad_mca_ipa0_5
    OCO_class.rad_c1d_5 = rad_mca_ipa_5
    OCO_class.rad_c3d_5 = rad_mca_3d_5
    OCO_class.rad_clrs_5 = rad_mca_ipa0_5_std
    OCO_class.ra_c1ds_5 = rad_mca_ipa_5_std
    OCO_class.rad_c3ds_5 = rad_mca_3d_5_std

    OCO_class.rad_clr_9 = rad_mca_ipa0_9
    OCO_class.rad_c1d_9 = rad_mca_ipa_9
    OCO_class.rad_c3d_9 = rad_mca_3d_9
    OCO_class.rad_clrs_9 = rad_mca_ipa0_9_std
    OCO_class.ra_c1ds_9 = rad_mca_ipa_9_std
    OCO_class.rad_c3ds_9 = rad_mca_3d_9_std

    OCO_class.rad_clr_13 = rad_mca_ipa0_13
    OCO_class.rad_c1d_13 = rad_mca_ipa_13
    OCO_class.rad_c3d_13 = rad_mca_3d_13
    OCO_class.rad_clrs_13 = rad_mca_ipa0_13_std
    OCO_class.ra_c1ds_13 = rad_mca_ipa_13_std
    OCO_class.rad_c3ds_13 = rad_mca_3d_13_std

    OCO_class.rad_clr_41 = rad_mca_ipa0_41
    OCO_class.rad_c1d_41 = rad_mca_ipa_41
    OCO_class.rad_c3d_41 = rad_mca_3d_41
    OCO_class.rad_clrs_41 = rad_mca_ipa0_41_std
    OCO_class.ra_c1ds_41 = rad_mca_ipa_41_std
    OCO_class.rad_c3ds_41 = rad_mca_3d_41_std
    
    OCO_class.sl_5  = (OCO_class.rad_c3d_5-OCO_class.rad_clr_5) / OCO_class.rad_clr_5        # S_lamda
    OCO_class.sls_5 = (OCO_class.rad_c3ds_5/OCO_class.rad_clr_5 + OCO_class.rad_clrs_5/OCO_class.rad_clr_5)

    OCO_class.sl_9  = (OCO_class.rad_c3d_9-OCO_class.rad_clr_9) / OCO_class.rad_clr_9        # S_lamda
    OCO_class.sls_9 = (OCO_class.rad_c3ds_9/OCO_class.rad_clr_9 + OCO_class.rad_clrs_9/OCO_class.rad_clr_9)

    OCO_class.sl_13  = (OCO_class.rad_c3d_13-OCO_class.rad_clr_13) / OCO_class.rad_clr_13        # S_lamda
    OCO_class.sls_13 = (OCO_class.rad_c3ds_13/OCO_class.rad_clr_13 + OCO_class.rad_clrs_13/OCO_class.rad_clr_13)

    OCO_class.sl_41  = (OCO_class.rad_c3d_41-OCO_class.rad_clr_41) / OCO_class.rad_clr_41        # S_lamda
    OCO_class.sls_41 = (OCO_class.rad_c3ds_41/OCO_class.rad_clr_41 + OCO_class.rad_clrs_41/OCO_class.rad_clr_41)


def coarsening(OCO_class, size=3):
    ipa0 = coarsening_subfunction(OCO_class.rad_clr, size)
    ipa  = coarsening_subfunction(OCO_class.rad_c1d, size)
    c3d  = coarsening_subfunction(OCO_class.rad_c3d, size)
    ipa0_std = coarsening_subfunction(OCO_class.rad_clrs, size)
    ipa_std  = coarsening_subfunction(OCO_class.rad_c1ds, size)
    c3d_std   = coarsening_subfunction(OCO_class.rad_c3ds, size)

    return ipa0, ipa, c3d, ipa0_std, ipa_std, c3d_std, 


def coarsening_subfunction(rad_mca, size):
    lams = rad_mca.shape[-1]
    tmp = np.zeros_like(rad_mca)
    for i in range(lams):
        tmp[:,:,i] = uniform_filter(rad_mca[:,:,i], size=size, mode='constant', cval=-999999)
    tmp[tmp<0] = np.nan
    return tmp

def get_slope_np(toa, sl_np, sls_np, c3d_np, clr_np, fp, z, points=11, mode='unperturb'):
     
    nwl=sls_np[z,fp,:].shape[0]
    flt=np.where(sls_np[z,fp,:]>1e-6)
    #flt=np.where(~np.isnan(sls[:]))
    use=len(flt[0])
    if use==nwl:
        w=1./sls_np[z,fp,:]    
        if mode=='unperturb':
            x=c3d_np[z,fp,:]/toa[:]*np.pi
        else:
            x=clr_np[z,fp,:]/toa[:]*np.pi
        x_len = len(x)
        mask = np.argsort(x)[x_len-points:]
        res=np.polyfit(x[mask], sl_np[z,fp,:][mask], 1, w=w[mask], cov=True) # now get covariance as well!
        slope,intercept=res[0]
        slopestd=np.sqrt(res[1][0][0])
        interceptstd=np.sqrt(res[1][1][1])
    else:
        slope=np.nan; slopestd=np.nan; intercept=np.nan; interceptstd=np.nan
    return(slope,slopestd,intercept,interceptstd)

def get_slope_1km(OCO_class,fp,z, points=11, mode='unperturb'):
    nwl=OCO_class.sls_1km[z,fp,:].shape[0]
    flt=np.where(OCO_class.sls_1km[z,fp,:]>1e-6)
    #flt=np.where(~np.isnan(sls[:]))
    use=len(flt[0])
    if use==nwl:
        w=1./OCO_class.sls_1km[z,fp,:]    
        if mode=='unperturb':
            x=OCO_class.rad_1km_c3d[z,fp,:]/OCO_class.toa[:]*np.pi
        else:
            x=OCO_class.rad_1km_clr[z,fp,:]/OCO_class.toa[:]*np.pi
        x_len = len(x)
        mask = np.argsort(x)[x_len-points:]
        mask = np.argsort(x)[5:]
        res=np.polyfit(x[mask],OCO_class.sl_1km[z,fp,:][mask],1,w=w[mask],cov=True) # now get covariance as well!
        slope,intercept=res[0]
        slopestd=np.sqrt(res[1][0][0])
        interceptstd=np.sqrt(res[1][1][1])
    else:
        slope=np.nan; slopestd=np.nan; intercept=np.nan; interceptstd=np.nan
    return(slope,slopestd,intercept,interceptstd)

def get_slope_25p(OCO_class, fp, z, points=11, mode='unperturb'):
    nwl=OCO_class.sls_25p[z,fp,:].shape[0]
    flt=np.where(OCO_class.sls_25p[z,fp,:]>1e-6)
    #flt=np.where(~np.isnan(sls[:]))
    use=len(flt[0])
    if use==nwl:
        w=1./OCO_class.sls_25p[z,fp,:]
        if mode=='unperturb':
            x=OCO_class.rad_25p_c3d[z,fp,:]/OCO_class.toa[:]*np.pi
        else:
            x=OCO_class.rad_25p_clr[z,fp,:]/OCO_class.toa[:]*np.pi   
        x_len = len(x)
        mask = np.argsort(x)[x_len-points:]
        res=np.polyfit(x[mask],OCO_class.sl_25p[z,fp,:][mask],1,w=w[mask],cov=True) # now get covariance as well!
        slope,intercept=res[0]
        slopestd=np.sqrt(res[1][0][0])
        interceptstd=np.sqrt(res[1][1][1])
    else:
        slope=np.nan; slopestd=np.nan; intercept=np.nan; interceptstd=np.nan
    return(slope,slopestd,intercept,interceptstd)

def slopes_propagation(OCO_class, mode='unperturb'): # goes through entire line for a given footprint fp
    # OCO_class.slope_1km = np.zeros([OCO_class.nz,OCO_class.nf,2])
    # OCO_class.inter_1km = np.zeros([OCO_class.nz,OCO_class.nf,2])
    # OCO_class.slope_25p = np.zeros([OCO_class.nz,OCO_class.nf,2])
    # OCO_class.inter_25p = np.zeros([OCO_class.nz,OCO_class.nf,2])
    OCO_class.slope_5avg = np.zeros([OCO_class.rad_clr_5.shape[0],OCO_class.rad_clr_5.shape[1], 2])
    OCO_class.inter_5avg = np.zeros([OCO_class.rad_clr_5.shape[0],OCO_class.rad_clr_5.shape[1], 2])
    OCO_class.slope_9avg = np.zeros([OCO_class.rad_clr_5.shape[0],OCO_class.rad_clr_5.shape[1], 2])
    OCO_class.inter_9avg = np.zeros([OCO_class.rad_clr_5.shape[0],OCO_class.rad_clr_5.shape[1], 2])
    OCO_class.slope_13avg = np.zeros([OCO_class.rad_clr_5.shape[0],OCO_class.rad_clr_5.shape[1], 2])
    OCO_class.inter_13avg = np.zeros([OCO_class.rad_clr_5.shape[0],OCO_class.rad_clr_5.shape[1], 2])
    OCO_class.slope_41avg = np.zeros([OCO_class.rad_clr_5.shape[0],OCO_class.rad_clr_5.shape[1], 2])
    OCO_class.inter_41avg = np.zeros([OCO_class.rad_clr_5.shape[0],OCO_class.rad_clr_5.shape[1], 2])
    # for z in range(OCO_class.nz):
    #     for fp in range(OCO_class.nf):
    #         if 1:#~ np.isnan(OCO_class.co2[z,fp,]):
    #             slope,slopestd,inter,interstd=OCO_class.get_slope(fp,z,mode='unperturb')
    #             OCO_class.slope[z,fp,:]=[slope,slopestd]
    #             OCO_class.inter[z,fp,:]=[inter,interstd]
    #             slope,slopestd,inter,interstd=get_slope_1km(OCO_class, fp,z,mode='unperturb')
    #             OCO_class.slope_1km[z,fp,:]=[slope,slopestd]
    #             OCO_class.inter_1km[z,fp,:]=[inter,interstd]  
    #             slope,slopestd,inter,interstd=get_slope_25p(OCO_class, fp,z,mode='unperturb')
    #             OCO_class.slope_25p[z,fp,:]=[slope,slopestd]
    #             OCO_class.inter_25p[z,fp,:]=[inter,interstd]  

    for z in range(OCO_class.rad_clr_5.shape[0]):
        for fp in range(OCO_class.rad_clr_5.shape[1]):   
            slope,slopestd,inter,interstd=get_slope_np(OCO_class.toa, OCO_class.sl_5, OCO_class.sls_5, OCO_class.rad_c3d_5, OCO_class.rad_clr_5, fp, z, points=11, mode='unperturb')
            OCO_class.slope_5avg[z,fp,:]=[slope,slopestd]
            OCO_class.inter_5avg[z,fp,:]=[inter,interstd]

            slope,slopestd,inter,interstd=get_slope_np(OCO_class.toa, OCO_class.sl_9, OCO_class.sls_9, OCO_class.rad_c3d_9, OCO_class.rad_clr_9, fp, z, points=11, mode='unperturb')
            OCO_class.slope_9avg[z,fp,:]=[slope,slopestd]
            OCO_class.inter_9avg[z,fp,:]=[inter,interstd]

            slope,slopestd,inter,interstd=get_slope_np(OCO_class.toa, OCO_class.sl_13, OCO_class.sls_13, OCO_class.rad_c3d_13, OCO_class.rad_clr_13, fp, z, points=11, mode='unperturb')
            OCO_class.slope_13avg[z,fp,:]=[slope,slopestd]
            OCO_class.inter_13avg[z,fp,:]=[inter,interstd]

            slope,slopestd,inter,interstd=get_slope_np(OCO_class.toa, OCO_class.sl_41, OCO_class.sls_41, OCO_class.rad_c3d_41, OCO_class.rad_clr_41, fp, z, points=11, mode='unperturb')
            OCO_class.slope_41avg[z,fp,:]=[slope,slopestd]
            OCO_class.inter_41avg[z,fp,:]=[inter,interstd]

            

def main(cfg_name='20151219_north_italy_470cloud_test.csv'):

    cfg_dir = '../simulation/cfg'


    cfg_info = grab_cfg(f'{cfg_dir}/{cfg_name}')
    if 'o2' in cfg_info.keys():
        id_num = output_h5_info(f'{cfg_dir}/{cfg_name}', 'o2')[-12:-3]
        boundary = [[float(i) for i in cfg_info['subdomain']], 'r']
    else:
        boundary = [[float(i) for i in cfg_info['subdomain']], 'orange']
    subdomain = cfg_info['subdomain']


    alb_list, sza_list = [], []
    o2a_slope_a_list = []
    o2a_slope_b_list = []
    o2a_inter_a_list = []
    o2a_inter_b_list = []
    
    wco2_slope_a_list = []
    wco2_slope_b_list = []
    wco2_inter_a_list = []
    wco2_inter_b_list = []

    sco2_slope_a_list = []
    sco2_slope_b_list = []
    sco2_inter_a_list = []
    sco2_inter_b_list = []

    compare_num = 9
    rad_c3d_compare = f'rad_c3d_{compare_num}'
    rad_clr_compare = f'rad_clr_{compare_num}'
    slope_compare = f'slope_{compare_num}avg'
    inter_compare = f'inter_{compare_num}avg'
    if not os.path.isfile(f'o2a_para_{compare_num}_italy.csv'):
        if not os.path.isfile(f'20151219_north_italy_470cloud_test_o2a.pkl'):

            filename = '../simulation/data_all_20151219_{}_{}.h5'
            o2a_file  = filename.format('o2a', id_num)
            o1 = OCOSIM(o2a_file)

            wco2_file  = filename.format('wco2', id_num)
            o2 = OCOSIM(wco2_file)

            sco2_file  = filename.format('sco2', id_num)
            o3 = OCOSIM(sco2_file)

            wco2_file = filename.format('wco2', id_num)
            sco2_file = filename.format('sco2', id_num)
            for var in [o1, o2, o3]:#, ('o2', wco2_file), ('o3', sco2_file)]:
                # for j in range(8):
                #     var.slopes(j)
                # near_rad_calc(var)
                near_rad_calc_all(var)
                slopes_propagation(var)

            file_o1 = open(f'20151219_north_italy_470cloud_test_o2a.pkl', 'wb') 
            pickle.dump(o1, file_o1)

            file_o2 = open(f'20151219_north_italy_470cloud_test_wco2.pkl', 'wb')
            pickle.dump(o2, file_o2)

            file_o3 = open(f'20151219_north_italy_470cloud_test_sco2.pkl', 'wb') 
            pickle.dump(o3, file_o3)

            file_o1.close()
            file_o2.close()
            file_o3.close()
        else:
            with open(f'20151219_north_italy_470cloud_test_o2a.pkl', 'rb') as f:
                o1 = pickle.load(f)
            with open(f'20151219_north_italy_470cloud_test_wco2.pkl', 'rb') as f:
                o2 = pickle.load(f)
            with open(f'20151219_north_italy_470cloud_test_sco2.pkl', 'rb') as f:
                o3 = pickle.load(f)


        if not os.path.isfile(f'{cfg_name[:-4]}_cld_distance.pkl'):
            cld_dist_calc(cfg_name, o2, slope_compare)
        cld_data = pd.read_pickle(f'{cfg_name[:-4]}_cld_distance.pkl')
        cld_dist = cld_data['cld_dis']

        # print(getattr(o1, rad_c3d_compare)[:,:, -1])
        # plt.contourf(o1.lon2d, o1.lat2d, getattr(o1, rad_c3d_compare)[:,:, -1])
        
        # plt.show()

        o2_slope_a, o2_slope_b, o2_inter_a, o2_inter_b = fitting(cld_dist, getattr(o1, rad_c3d_compare)[:,:, -1].flatten(), getattr(o1, rad_clr_compare)[:,:, -1].flatten(), getattr(o1, slope_compare)[:,:,0].flatten(), getattr(o1, inter_compare)[:,:,0].flatten(),
                                                        band=f'O_2-A_{compare_num}',  plot=True)
        
        # o2a_slope_a_list.append(o2_slope_a)
        # o2a_slope_b_list.append(o2_slope_b)
        # o2a_inter_a_list.append(o2_inter_a)
        # o2a_inter_b_list.append(o2_inter_b)

        wco2_slope_a, wco2_slope_b, wco2_inter_a, wco2_inter_b = fitting(cld_dist, getattr(o2, rad_c3d_compare)[:,:, -1].flatten(), getattr(o2, rad_clr_compare)[:,:, -1].flatten(), getattr(o2, slope_compare)[:,:,0].flatten(), getattr(o2, inter_compare)[:,:,0].flatten(),
                                                        band=f'WCO_2_{compare_num}', plot=True)
        # wco2_slope_a_list.append(wco2_slope_a)
        # wco2_slope_b_list.append(wco2_slope_b)
        # wco2_inter_a_list.append(wco2_inter_a)
        # wco2_inter_b_list.append(wco2_inter_b)

        sco2_slope_a, sco2_slope_b, sco2_inter_a, sco2_inter_b = fitting(cld_dist, getattr(o3, rad_c3d_compare)[:,:, -1].flatten(), getattr(o3, rad_clr_compare)[:,:, -1].flatten(), getattr(o3, slope_compare)[:,:,0].flatten(), getattr(o3, inter_compare)[:,:,0].flatten(),
                                                        band=f'SCO_2_{compare_num}', plot=True)
        # sco2_slope_a_list.append(sco2_slope_a)
        # sco2_slope_b_list.append(sco2_slope_b)
        # sco2_inter_a_list.append(sco2_inter_a)
        # sco2_inter_b_list.append(sco2_inter_b)

    
        # out_o2a = pd.DataFrame(np.array([alb_list, sza_list, o2a_slope_a_list, o2a_slope_b_list, o2a_inter_a_list, o2a_inter_b_list]).T,
        #                 columns=['albedo', 'sza', 'o2a_slope_a', 'o2a_slope_b', 'o2a_inter_a', 'o2a_inter_b'])
        # out_o2a.to_csv(f'o2a_para_{compare_num}_italy.csv')

        # out_wco2 = pd.DataFrame(np.array([alb_list, sza_list, wco2_slope_a_list, wco2_slope_b_list, wco2_inter_a_list, wco2_inter_b_list]).T,
        #                 columns=['albedo', 'sza', 'wco2_slope_a', 'wco2_slope_b', 'wco2_inter_a', 'wco2_inter_b'])
        # out_wco2.to_csv(f'wco2_para_{compare_num}_italy.csv')

        # out_sco2 = pd.DataFrame(np.array([alb_list, sza_list, sco2_slope_a_list, sco2_slope_b_list, sco2_inter_a_list, sco2_inter_b_list]).T,
        #                 columns=['albedo', 'sza', 'sco2_slope_a', 'sco2_slope_b', 'sco2_inter_a', 'sco2_inter_b'])
        # out_sco2.to_csv(f'sco2_para_{compare_num}_italy.csv')
        # print(out_wco2)
    else:
        None
        # out_o2a = pd.read_csv(f'o2a_para_{compare_num}_italy.csv')
        # out_wco2 = pd.read_csv(f'wco2_para_{compare_num}_italy.csv')
        # out_sco2 = pd.read_csv(f'sco2_para_{compare_num}_italy.csv')

    # for col in ['o2a_slope_a', 'o2a_slope_b', 'o2a_inter_a', 'o2a_inter_b']:
    #     out_o2a[col][out_o2a[col]<1e-4] = np.nan
    # sfc_alb = out_o2a['albedo']
    # sza = out_o2a['sza']
    # inter_a_list = out_o2a['o2a_inter_a']
    # slope_a_list = out_o2a['o2a_slope_a']
    # inter_e_list = 1/out_o2a['o2a_inter_b']
    # slope_e_list = 1/out_o2a['o2a_slope_b']
    # plot_alb_sza_relationship(sfc_alb, sza, inter_a_list, slope_a_list, inter_e_list, slope_e_list, 'o2a', compare_num)

    # for col in ['wco2_slope_a', 'wco2_slope_b', 'wco2_inter_a', 'wco2_inter_b']:
    #     out_wco2[col][out_wco2[col]<1e-4] = np.nan
    # sfc_alb = out_wco2['albedo']
    # sza = out_wco2['sza']
    # inter_a_list = out_wco2['wco2_inter_a']
    # slope_a_list = out_wco2['wco2_slope_a']
    # inter_e_list = 1/out_wco2['wco2_inter_b']
    # slope_e_list = 1/out_wco2['wco2_slope_b']
    # plot_alb_sza_relationship(sfc_alb, sza, inter_a_list, slope_a_list, inter_e_list, slope_e_list, 'wco2', compare_num)

    # for col in ['sco2_slope_a', 'sco2_slope_b', 'sco2_inter_a', 'sco2_inter_b']:
    #     out_sco2[col][out_sco2[col]<1e-4] = np.nan
    # sfc_alb = out_sco2['albedo']
    # sza = out_sco2['sza']
    # inter_a_list = out_sco2['sco2_inter_a']
    # slope_a_list = out_sco2['sco2_slope_a']
    # inter_e_list = 1/out_sco2['sco2_inter_b']
    # slope_e_list = 1/out_sco2['sco2_slope_b']
    
    # plot_alb_sza_relationship(sfc_alb, sza, inter_a_list, slope_a_list, inter_e_list, slope_e_list, 'sco2', compare_num)

    # plot_all_band_alb_sza_relationship(sfc_alb, sza, out_o2a['o2a_slope_a'], out_wco2['wco2_slope_a'], out_sco2['sco2_slope_a'], compare_num)

def plot_all_band_alb_sza_relationship(sfc_alb, sza, o2a_slope_a_list, wco2_slope_a_list, sco2_slope_a_list, point_avg):
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 5), sharex=False)
    fig.tight_layout(pad=5.0)
    label_size = 16
    tick_size = 12

    o2a_slope_a = np.array([o2a_slope_a_list[sza==30], o2a_slope_a_list[sza==45], o2a_slope_a_list[sza==60]]).mean(axis=0)
    wco2_slope_a = np.array([wco2_slope_a_list[sza==30], wco2_slope_a_list[sza==45], wco2_slope_a_list[sza==60]]).mean(axis=0)
    sco2_slope_a = np.array([sco2_slope_a_list[sza==30], sco2_slope_a_list[sza==45], sco2_slope_a_list[sza==60]]).mean(axis=0)

    ax1.scatter(sfc_alb[sza==30], o2a_slope_a_list[sza==45], s=50, label='o2a, sza=45', marker='X', alpha=0.65)
    ax1.scatter(sfc_alb[sza==45], wco2_slope_a_list[sza==45], s=50, label='wco2, sza=45', marker='p', alpha=0.65)
    ax1.scatter(sfc_alb[sza==60], sco2_slope_a_list[sza==45], s=50, label='sco2, sza=45', alpha=0.65)

    xx = sfc_alb[sza==30]
    yy = o2a_slope_a
    mask = ~(np.isnan(xx) | np.isnan(yy) | np.isinf(xx) | np.isinf(yy))
    xx_o2a, yy_o2a = xx[mask], yy[mask]
    
    popt, pcov = curve_fit(func_with_intercept, xx_o2a, yy_o2a, bounds=([-5, 0., 0], [5, 50, 0.3]),
                        #p0=(0.1, 0.7),
                        maxfev=3000,
                        #sigma=value_std[val_mask], 
                        #absolute_sigma=True,
                        )
    residuals = yy_o2a - func_with_intercept(xx_o2a, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy_o2a-np.mean(yy_o2a))**2)
    r_squared = 1 - (ss_res / ss_tot)

    plot_xx = np.arange(0.05, xx.max()+0.05, 0.01)
    ax1.plot(plot_xx, func_with_intercept(plot_xx, *popt), '--', color='tab:blue', 
              label='o2a\nfit: a=%5.3f\n     b=%5.1f\n     c=%5.3f' % tuple(popt), linewidth=1.5, alpha=0.5)
    
    yy = wco2_slope_a
    mask = ~(np.isnan(xx) | np.isnan(yy) | np.isinf(xx) | np.isinf(yy))
    xx_wco2, yy_wco2 = xx[mask], yy[mask]

    popt, pcov = curve_fit(func_with_intercept, xx_wco2, yy_wco2, bounds=([-5, 0., 0], [5, 50, 0.3]),
                        #p0=(0.1, 0.7),
                        maxfev=3000,
                        #sigma=value_std[val_mask], 
                        #absolute_sigma=True,
                        )
    residuals = yy_wco2 - func_with_intercept(xx_wco2, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy_wco2-np.mean(yy_wco2))**2)
    r_squared = 1 - (ss_res / ss_tot)

    plot_xx = np.arange(0.05, xx.max()+0.05, 0.01)
    ax1.plot(plot_xx, func_with_intercept(plot_xx, *popt), '--', color='tab:orange', 
              label='wco2\nfit: a=%5.3f\n     b=%5.1f\n     c=%5.3f' % tuple(popt), linewidth=1.5, alpha=0.5)
    
    yy = sco2_slope_a
    mask = ~(np.isnan(xx) | np.isnan(yy) | np.isinf(xx) | np.isinf(yy))
    xx_sco2, yy_sco2 = xx[mask], yy[mask]
    
    popt, pcov = curve_fit(func_with_intercept, xx_sco2, yy_sco2, bounds=([-5, 0., 0], [5, 50, 0.3]),
                        #p0=(0.1, 0.7),
                        maxfev=3000,
                        #sigma=value_std[val_mask], 
                        #absolute_sigma=True,
                        )
    residuals = yy_sco2 - func_with_intercept(xx_sco2, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy_sco2-np.mean(yy_sco2))**2)
    r_squared = 1 - (ss_res / ss_tot)

    plot_xx = np.arange(0.05, xx.max()+0.05, 0.01)
    ax1.plot(plot_xx, func_with_intercept(plot_xx, *popt), '--', color='tab:green', 
              label='sco2\nfit: a=%5.3f\n     b=%5.1f\n     c=%5.3f' % tuple(popt), linewidth=1.5, alpha=0.5)

    



    # ax1.scatter(sfc_alb[sza==30], slope_a_list[sza==30], s=50, label='sza=30', marker='X', alpha=0.65)
    # ax1.scatter(sfc_alb[sza==45], slope_a_list[sza==45], s=50, label='sza=45', marker='p', alpha=0.65)
    # ax1.scatter(sfc_alb[sza==60], slope_a_list[sza==60], s=50, label='sza=60', alpha=0.65)
    

    """val_mask = ~(np.isnan(value_avg) | np.isnan(value_std) | np.isinf(value_avg) | np.isinf(value_std))
    #print(value_avg[val_mask])
    #print(value_std[val_mask])
    temp_r2 = 0
    for cld_max in np.arange(3, 15, 0.5):
        cld_val = cld_list[val_mask]
        xx = cld_val[cld_val<=cld_max]
        yy = value_avg[val_mask][cld_val<=cld_max]
        popt, pcov = curve_fit(func, xx, yy, bounds=([-2, 0.], [2, 10,]),
                            p0=(0.1, 0.7),
                            maxfev=3000,
                            #sigma=value_std[val_mask], 
                            #absolute_sigma=True,
                            )
        residuals = yy - func(xx, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((yy-np.mean(yy))**2)
        r_squared = 1 - (ss_res / ss_tot)

        if r_squared > temp_r2:
            temp_r2 = r_squared
        else:
            break

    plot_xx = np.arange(0, cld_list.max()+0.75, 0.5)
    ax.plot(plot_xx, func(plot_xx, *popt), '--', color='limegreen', 
            label='fit: a=%5.3f\n     b=%5.3f' % tuple(popt), linewidth=3.5)
    print('-'*15)
    print(f'E-folding dis: {1/popt[1]}')"""
    #ax.plot(cld_list, func(cld_list, 1, 2), '--', color='green',)
    #ax.plot(cld_list, func(cld_list, 0.2, 1), '--', color='cyan',)
    

    
    
    
    
    
    ax1.legend()
    

    ax1.set_ylabel('a', fontsize=label_size)
    
    ax1.set_xlabel('surface albedo', fontsize=label_size)
    
    ax1.set_title('coefficient a for slope', fontsize=label_size)
    
    fig.tight_layout()
    fig.savefig(f'all_bands_slope_alb_{point_avg}avg.png', dpi=150)
    plt.show()

def plot_alb_sza_relationship(sfc_alb, sza, inter_a_list, slope_a_list, inter_e_list, slope_e_list, band_tag, point_avg):
    fig, ((ax1, ax2),
        (ax21, ax22)) = plt.subplots(2, 2, figsize=(12, 10), sharex=False)
    fig.tight_layout(pad=5.0)
    label_size = 16
    tick_size = 12


    ax1.scatter(sfc_alb[sza==30], slope_a_list[sza==30], s=50, label='sza=30', marker='X', alpha=0.65)
    ax1.scatter(sfc_alb[sza==45], slope_a_list[sza==45], s=50, label='sza=45', marker='p', alpha=0.65)
    ax1.scatter(sfc_alb[sza==60], slope_a_list[sza==60], s=50, label='sza=60', alpha=0.65)
    #ax.hist2d(plot_x, plot_y, bins=150, norm=LogNorm(), cmap=light_jet)
    ax2.scatter(sfc_alb[sza==30], inter_a_list[sza==30], s=50, label='sza=30', marker='X', alpha=0.65)
    ax2.scatter(sfc_alb[sza==45], inter_a_list[sza==45], s=50, label='sza=45', marker='p', alpha=0.65)
    ax2.scatter(sfc_alb[sza==60], inter_a_list[sza==60], s=50, label='sza=60', alpha=0.65)


    ax21.scatter(sfc_alb[sza==30], slope_e_list[sza==30], s=50, label='sza=30', marker='X', alpha=0.65)
    ax21.scatter(sfc_alb[sza==45], slope_e_list[sza==45], s=50, label='sza=45', marker='p', alpha=0.65)
    ax21.scatter(sfc_alb[sza==60], slope_e_list[sza==60], s=50, label='sza=60', alpha=0.65)
    #ax.hist2d(plot_x, plot_y, bins=150, norm=LogNorm(), cmap=light_jet)
    ax22.scatter(sfc_alb[sza==30], inter_e_list[sza==30], s=50, label='sza=30', marker='X', alpha=0.65)
    ax22.scatter(sfc_alb[sza==45], inter_e_list[sza==45], s=50, label='sza=45', marker='p', alpha=0.65)
    ax22.scatter(sfc_alb[sza==60], inter_e_list[sza==60], s=50, label='sza=60', alpha=0.65)

    """val_mask = ~(np.isnan(value_avg) | np.isnan(value_std) | np.isinf(value_avg) | np.isinf(value_std))
    #print(value_avg[val_mask])
    #print(value_std[val_mask])
    temp_r2 = 0
    for cld_max in np.arange(3, 15, 0.5):
        cld_val = cld_list[val_mask]
        xx = cld_val[cld_val<=cld_max]
        yy = value_avg[val_mask][cld_val<=cld_max]
        popt, pcov = curve_fit(func, xx, yy, bounds=([-2, 0.], [2, 10,]),
                            p0=(0.1, 0.7),
                            maxfev=3000,
                            #sigma=value_std[val_mask], 
                            #absolute_sigma=True,
                            )
        residuals = yy - func(xx, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((yy-np.mean(yy))**2)
        r_squared = 1 - (ss_res / ss_tot)

        if r_squared > temp_r2:
            temp_r2 = r_squared
        else:
            break

    plot_xx = np.arange(0, cld_list.max()+0.75, 0.5)
    ax.plot(plot_xx, func(plot_xx, *popt), '--', color='limegreen', 
            label='fit: a=%5.3f\n     b=%5.3f' % tuple(popt), linewidth=3.5)
    print('-'*15)
    print(f'E-folding dis: {1/popt[1]}')"""
    #ax.plot(cld_list, func(cld_list, 1, 2), '--', color='green',)
    #ax.plot(cld_list, func(cld_list, 0.2, 1), '--', color='cyan',)
    

    
    xx = sfc_alb[sza==30]
    yy = np.nanmean(np.array([slope_a_list[sza==30], slope_a_list[sza==45], slope_a_list[sza==60]]), axis=0)
    mask = ~(np.isnan(xx) | np.isnan(yy) | np.isinf(xx) | np.isinf(yy))
    xx, yy = xx[mask], yy[mask]


    popt, pcov = curve_fit(func_with_intercept, xx, yy, bounds=([-5, 0., 0], [5, 50, 0.3]),
                        #p0=(0.1, 0.7),
                        maxfev=3000,
                        #sigma=value_std[val_mask], 
                        #absolute_sigma=True,
                        )
    residuals = yy - func_with_intercept(xx, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy-np.mean(yy))**2)
    r_squared = 1 - (ss_res / ss_tot)

    plot_xx = np.arange(0.05, xx.max()+0.05, 0.01)
    ax1.plot(plot_xx, func_with_intercept(plot_xx, *popt), '--', color='r', 
              label='fit: a=%5.3f\n     b=%5.1f\n     c=%5.3f' % tuple(popt), linewidth=1.5, alpha=0.5)
    
    
    xx = sfc_alb[sza==30]
    yy = np.array([inter_a_list[sza==30], inter_a_list[sza==45], inter_a_list[sza==60]]).mean(axis=0)
    popt, pcov = curve_fit(func_with_intercept, xx, yy, bounds=([-5, 0., 0], [5, 50, 0.3]),
                        #p0=(0.1, 0.7),
                        maxfev=3000,
                        #sigma=value_std[val_mask], 
                        #absolute_sigma=True,
                        )
    residuals = yy - func_with_intercept(xx, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy-np.mean(yy))**2)
    r_squared = 1 - (ss_res / ss_tot)
    plot_xx = np.arange(0.05, xx.max()+0.05, 0.01)
    # ax2.plot(plot_xx, func_with_intercept(plot_xx, *popt), '--', color='tab:blue', 
    #           label='fit: a=%5.3f\n     b=%5.1f\n     c=%5.3f' % tuple(popt), linewidth=1.5, alpha=0.5)
    b_inter_a = popt[1]
    

    xx = sfc_alb[sza==30]
    yy = inter_a_list[sza==30]
    popt, pcov = curve_fit(func_with_intercept, xx, yy, bounds=([-5, b_inter_a, 0], [5, b_inter_a*1.00001, 0.3]),
                        #p0=(0.1, 0.7),
                        maxfev=3000,
                        #sigma=value_std[val_mask], 
                        #absolute_sigma=True,
                        )
    residuals = yy - func_with_intercept(xx, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy-np.mean(yy))**2)
    r_squared = 1 - (ss_res / ss_tot)
    plot_xx = np.arange(0.05, xx.max()+0.05, 0.01)
    ax2.plot(plot_xx, func_with_intercept(plot_xx, *popt), '--', color='tab:blue', 
              label='fit: a=%5.3f\n     b=%5.1f\n     c=%5.3f' % tuple(popt), linewidth=1.5, alpha=0.5)
    
    yy = inter_a_list[sza==45]
    popt, pcov = curve_fit(func_with_intercept, xx, yy, bounds=([-5, b_inter_a, 0], [5, b_inter_a*1.00001, 0.3]),
                        #p0=(0.1, 0.7),
                        maxfev=3000,
                        #sigma=value_std[val_mask], 
                        #absolute_sigma=True,
                        )
    residuals = yy - func_with_intercept(xx, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy-np.mean(yy))**2)
    r_squared = 1 - (ss_res / ss_tot)
    plot_xx = np.arange(0.05, xx.max()+0.05, 0.01)
    ax2.plot(plot_xx, func_with_intercept(plot_xx, *popt), '--', color='tab:orange', 
              label='fit: a=%5.3f\n     b=%5.1f\n     c=%5.3f' % tuple(popt), linewidth=1.5, alpha=0.5)
    
    yy = inter_a_list[sza==60]
    popt, pcov = curve_fit(func_with_intercept, xx, yy, bounds=([-5, b_inter_a, 0], [5, b_inter_a*1.00001, 0.3]),
                        #p0=(0.1, 0.7),
                        maxfev=3000,
                        #sigma=value_std[val_mask], 
                        #absolute_sigma=True,
                        )
    residuals = yy - func_with_intercept(xx, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy-np.mean(yy))**2)
    r_squared = 1 - (ss_res / ss_tot)
    plot_xx = np.arange(0.05, xx.max()+0.05, 0.01)
    ax2.plot(plot_xx, func_with_intercept(plot_xx, *popt), '--', color='tab:green', 
              label='fit: a=%5.3f\n     b=%5.1f\n     c=%5.3f' % tuple(popt), linewidth=1.5, alpha=0.5)

    yy = np.array([slope_e_list[sza==30], slope_e_list[sza==45], slope_e_list[sza==60]])
    yy = yy[yy<10].mean()
    ax21.hlines(yy, plot_xx[0], plot_xx[-1], 'r', alpha=0.5, label=f'{yy:.3f}')

    yy = inter_e_list[sza==30]
    yy = yy[yy<10].mean()
    ax22.hlines(yy, plot_xx[0], plot_xx[-1], 'tab:blue', alpha=0.5, label=f'{yy:.3f}')
    yy = inter_e_list[sza==45]
    yy = yy[yy<10].mean()
    ax22.hlines(yy, plot_xx[0], plot_xx[-1], 'tab:orange', alpha=0.5, label=f'{yy:.3f}')
    yy = inter_e_list[sza==60]
    yy = yy[yy<10].mean()
    ax22.hlines(yy, plot_xx[0], plot_xx[-1], 'tab:green', alpha=0.5, label=f'{yy:.3f}')
    
    ax1.legend()
    ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    ax21.legend()
    ax22.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))



    ax21.set_ylim(0, 10)

    ax1.set_ylabel('a', fontsize=label_size)
    ax2.set_ylabel('a', fontsize=label_size)

    ax21.set_ylabel('e-folding distance (km)', fontsize=label_size)
    ax22.set_ylabel('e-folding distance (km)', fontsize=label_size)

    ax1.set_xlabel('surface albedo', fontsize=label_size)
    ax2.set_xlabel('surface albedo', fontsize=label_size)
    ax21.set_xlabel('surface albedo', fontsize=label_size)
    ax22.set_xlabel('surface albedo', fontsize=label_size)

    ax1.set_title('coefficient a for slope', fontsize=label_size)
    ax2.set_title('coefficient a for intercept', fontsize=label_size)
    ax21.set_title('e-folding distance for slope', fontsize=label_size)
    ax22.set_title('e-folding distance for intercept', fontsize=label_size)
    fig.tight_layout()
    fig.savefig(f'{band_tag}_slope_inter_cld_distance_{point_avg}avg.png', dpi=150)
    plt.show()
    
class sat_tmp:

        def __init__(self, data):

            self.data = data

def cld_dist_calc(cfg_name, o1, slope_compare):


    cldfile = f'../simulation/data/{cfg_name[:-4]}_{cfg_name[:8]}/pre-data.h5'
    data = {}
    f = h5py.File(cldfile, 'r')
    data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=f['lon'][...])
    data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=f['lat'][...])
    data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=f[f'mod/cld/cot_ipa'][...])
    data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=f[f'mod/cld/cer_ipa'][...])
    data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=f[f'mod/cld/cth_ipa'][...])
    f.close()


    modl1b    =  sat_tmp(data)

    lon_2d, lat_2d = o1.lon2d, o1.lat2d
    lon_cld, lat_cld = modl1b.data['lon_2d']['data'], modl1b.data['lat_2d']['data']
    cld_list = modl1b.data['cth_2d']['data']>0
    cld_X, cld_Y = np.where(cld_list==1)[0], np.where(cld_list==1)[1]
    cld_position = []
    for i in range(len(cld_X)):
        cld_position.append(np.array([cld_X[i], cld_Y[i]]))
    cld_position = np.array(cld_position)

    cloud_dist = np.zeros_like(getattr(o1, slope_compare)[:,:,0])
    for j in range(cloud_dist.shape[1]):
        for i in range(cloud_dist.shape[0]):
            if cld_list[i, j] == 1:
                cloud_dist[i, j] = 0
            else:
                min_ind = np.argmin(np.sqrt(np.sum((cld_position-np.array([i, j]))**2, axis=1)))
                #print(min_ind)
                #print(cld_position[min_ind])
                cld_x, cld_y = cld_position[min_ind][0], cld_position[min_ind][1]
                dist = geopy.distance.distance((o1.lat2d[cld_x, cld_y], o1.lon2d[cld_x, cld_y]), (o1.lat2d[i, j], o1.lon2d[i, j])).km
                #print(dist)
                cloud_dist[i, j] = dist
    
    output = np.array([o1.lon2d, o1.lat2d, cloud_dist, ])
    cld_slope_inter = pd.DataFrame(output.reshape(output.shape[0], output.shape[1]*output.shape[2]).T,
                                columns=['lon', 'lat', 'cld_dis', ])

    cld_slope_inter.to_pickle(f'{cfg_name[:-4]}_cld_distance.pkl')

    
    



def heatmap_xy_3(x, y, ax):
    light_jet = cmap_map(lambda x: x/3*2 + 0.33, cm.jet)
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x, y = x[mask], y[mask]
    interval = 1/2
    start = 1
    

    # # Calculate the point density
    # data , x_e, y_e = np.histogram2d(x, y, bins=15)#, density=True)
    # z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)
    # z[np.where(np.isnan(z))] = 0.0
    # z[np.where(np.isinf(z))] = np.nanmax(z)
    # z[np.where(z<0)] = 0.0
    # # Sort the points by density, so that the densest points are plotted last
    # idx = z.argsort()
    # plot_x, plot_y, z = np.array(x)[idx], np.array(y)[idx], z[idx]
    # #ax.scatter(plot_x, plot_y, c=z, s=15*z/np.nanmax(z), cmap=light_jet)
    
    
    #
    #ax.scatter(x[x<start], y[x<start], s=1, color='lightgrey')
    ax.scatter(x[x>=start], y[x>=start], s=1, color='k')

    sns.kdeplot(x=x, y=y, cmap='hot_r', n_levels=20, fill=True, ax=ax, alpha=0.65)
    

    cld_levels = np.arange(start, 18, interval)
    value_avg, value_std = np.zeros(len(cld_levels)-1), np.zeros(len(cld_levels)-1)
    for i in range(len(cld_levels)-1):
        select = np.logical_and(x>=cld_levels[i], x < cld_levels[i+1])
        if select.sum()>0:
            #value_avg[i] = np.nanmean(y[select])
            #value_std[i] = np.nanstd(y[select])
            value_avg[i] = np.percentile(y[select], 50)
            value_std[i] = np.percentile(y[select], 75)-np.percentile(y[select], 25)
        else:
            value_avg[i] = np.nan
            value_std[i] = np.nan
    cld_list = (cld_levels[:-1] + cld_levels[1:])/2
    
    ax.errorbar(cld_list, value_avg, yerr=value_std, 
                marker='s', color='r', linewidth=2, linestyle='', ecolor='skyblue')#light_jet)
    
    val_mask = ~(np.isnan(value_avg) | np.isnan(value_std) | np.isinf(value_avg) | np.isinf(value_std))
    #print(value_avg[val_mask])
    #print(value_std[val_mask])
    temp_r2 = 0
    for cld_min in [1, 1.25]:
        for cld_max in np.arange(3, 15, 0.5):
            cld_val = cld_list[val_mask]
            mask = np.logical_and(cld_val>=cld_min, cld_val<=cld_max)
            xx = cld_val[mask]
            yy = value_avg[val_mask][mask]
            popt, pcov = curve_fit(func, xx, yy, bounds=([-5, 1e-3], [5, 15,]),
                                p0=(0.1, 0.7),
                                maxfev=3000,
                                #sigma=value_std[val_mask], 
                                #absolute_sigma=True,
                                )
            residuals = yy - func(xx, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((yy-np.mean(yy))**2)
            r_squared = 1 - (ss_res / ss_tot)

            if r_squared > temp_r2:
                temp_r2 = r_squared
            else:
                break
    
    plot_xx = np.arange(0, cld_list.max()+0.75, 0.5)
    ax.plot(plot_xx, func(plot_xx, *popt), '--', color='limegreen', 
              label='fit: a=%5.3f\n     b=%5.3f' % tuple(popt), linewidth=3.5)
    print('-'*15)
    print(f'E-folding dis: {1/popt[1]}')
    #ax.plot(cld_list, func(cld_list, 1, 2), '--', color='green',)
    #ax.plot(cld_list, func(cld_list, 0.2, 1), '--', color='cyan',)
    ax.legend()
    return popt#XX, YY, heatmap



from  scipy.optimize import curve_fit

def func(x, a, b):
     return a * np.exp(-b * x)

def func_with_intercept(x, a, b, c):
     return a * np.exp(-b * x) + c


def fitting(cloud_dist, rad_3d, rad_clr, slope, inter, band, plot=False):
    # fig, ((ax11, ax12), 
    #     (ax21, ax22),
    #     (ax31, ax32)) = plt.subplots(3, 2, figsize=(12, 12), sharex=False)
    if plot:
        fig, (ax11, ax12) = plt.subplots(1, 2, figsize=(12, 4), sharex=False)
        fig.tight_layout(pad=5.0)
        label_size = 16
        tick_size = 12

        mask = np.logical_and(cloud_dist > 0, rad_3d>rad_clr)

        #"""
        o2_slope_a, o2_slope_b = heatmap_xy_3(cloud_dist[mask], slope[mask], ax11)
        o2_inter_a, o2_inter_b = heatmap_xy_3(cloud_dist[mask], inter[mask], ax12)
        # heatmap_xy_3(cloud_dist[mask], df_all.wco2_slope[mask], ax21)
        # heatmap_xy_3(cloud_dist[mask], df_all.wco2_inter[mask], ax22)
        # heatmap_xy_3(cloud_dist[mask], df_all.sco2_slope[mask], ax31)
        # heatmap_xy_3(cloud_dist[mask], df_all.sco2_inter[mask], ax32)
        #"""


        #popt, pcov = curve_fit(func, cloud_dist[mask], o1.slope_1km_all[:,:,0][mask])#, bounds=(0, [3., 1., 0.5]))

        #ax11.plot(cloud_dist[mask], func(cloud_dist[mask], *popt), 'r--',
        #          label='fit: a=%5.3f, b=%5.3ff' % tuple(popt))


        """
        ax11.scatter(cloud_dist[mask], o1.slope_1km_all[:,:,0][mask])
        ax12.scatter(cloud_dist[mask], o1.inter_1km_all[:,:,0][mask])
        ax21.scatter(cloud_dist[mask], o2.slope_1km_all[:,:,0][mask])
        ax22.scatter(cloud_dist[mask], o2.inter_1km_all[:,:,0][mask])
        ax31.scatter(cloud_dist[mask], o3.slope_1km_all[:,:,0][mask])
        ax32.scatter(cloud_dist[mask], o3.inter_1km_all[:,:,0][mask])
        #"""
        for ax in [ax11, ax12]: # ax21, ax31, ax22, ax32
            ax.set_xlabel('Cloud distance (km)', fontsize=label_size)
            ax.tick_params(axis='both', labelsize=tick_size)
            _, xmax = ax.get_xlim()
            ax.hlines(0, 0, xmax, linestyle='--', color='white')
            
        ax11.set_ylabel('$\mathrm{band}$ slope', fontsize=label_size)
        ax12.set_ylabel('$\mathrm{band}$ intercept', fontsize=label_size)
        # ax21.set_ylabel('$\mathrm{WCO_2}$ slope', fontsize=label_size)
        # ax22.set_ylabel('$\mathrm{WCO_2}$ intercept', fontsize=label_size)
        # ax31.set_ylabel('$\mathrm{SCO_2}$ slope', fontsize=label_size)
        # ax32.set_ylabel('$\mathrm{SCO_2}$ intercept', fontsize=label_size)
        cld_low, cld_max = 0, 15
        limit_1 = 1.0
        limit_2 = 0.5
        for ax in [ax11,]:# ax21, ax31]:
            ax.set_xlim(cld_low, cld_max)
            ax.set_ylim(-limit_1, limit_1)
            
        for ax in [ax12,]:# ax22, ax32]:
            ax.set_xlim(cld_low, cld_max)
            ax.set_ylim(-limit_2, limit_2)

        #ax.plot([20, 20], [0, 1.1], 'r')
        #ax.plot([400, 400], [0, 1.1], 'r')
        #ax.set_ylim(0, 1.1)
        #ax.fill_between(t[2:41]*1e9, intensity[2:41], 0, color='lightgrey', interpolate=True)
        #I0 = quad(intensity_fxn, 20e-9, 400e-9, args=(decay_const))[0]

        #ax.set_yscale('log')
        # fig.suptitle(f"sfc albedo={alb:.2f}, sza={sza:.1f}$^\circ$")
        fig.savefig(f'italy_test_{band}.png', dpi=150, bbox_inches='tight')
        #plt.show()
    else:
        mask = np.logical_and(cloud_dist > 0, rad_3d>rad_clr)
        o2_slope_a, o2_slope_b = fitting_without_plot(cloud_dist[mask], slope[mask])
        o2_inter_a, o2_inter_b = fitting_without_plot(cloud_dist[mask], inter[mask])

    return o2_slope_a, o2_slope_b, o2_inter_a, o2_inter_b


def fitting_without_plot(x, y):
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x, y = x[mask], y[mask]
    interval = 1/2
    start = 1
    cld_levels = np.arange(start, 18, interval)
    value_avg, value_std = np.zeros(len(cld_levels)-1), np.zeros(len(cld_levels)-1)
    for i in range(len(cld_levels)-1):
        select = np.logical_and(x>=cld_levels[i], x < cld_levels[i+1])
        if select.sum()>0:
            #value_avg[i] = np.nanmean(y[select])
            #value_std[i] = np.nanstd(y[select])
            value_avg[i] = np.percentile(y[select], 50)
            value_std[i] = np.percentile(y[select], 75)-np.percentile(y[select], 25)
        else:
            value_avg[i] = np.nan
            value_std[i] = np.nan
    cld_list = (cld_levels[:-1] + cld_levels[1:])/2
    val_mask = ~(np.isnan(value_avg) | np.isnan(value_std) | np.isinf(value_avg) | np.isinf(value_std))
    #print(value_avg[val_mask])
    #print(value_std[val_mask])
    
    temp_r2 = 0
    for cld_min in [1, 1.25]:
        for cld_max in np.arange(3, 15, 0.5):
            cld_val = cld_list[val_mask]
            mask = np.logical_and(cld_val>=cld_min, cld_val<=cld_max)
            xx = cld_val[mask]
            yy = value_avg[val_mask][mask]
            popt, pcov = curve_fit(func, xx, yy, bounds=([-2, 0.], [2, 10,]),
                                p0=(0.1, 0.7),
                                maxfev=3000,
                                #sigma=value_std[val_mask], 
                                #absolute_sigma=True,
                                )
            residuals = yy - func(xx, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((yy-np.mean(yy))**2)
            r_squared = 1 - (ss_res / ss_tot)

            if r_squared > temp_r2:
                temp_r2 = r_squared
            else:
                break
    return popt

if __name__ == "__main__":
    now = time.time()
    
    main()

    print(f'{(time.time()-now)/60:.3f} min')