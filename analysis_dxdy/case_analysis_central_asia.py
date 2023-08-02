import sys
sys.path.append('/Users/yuch8913/miniforge3/envs/er3t_env/lib/python3.8/site-packages')
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
from glob import glob
import numpy as np
import copy
from bisect import bisect_left
from oco_post_class_ywc import OCOSIM
from matplotlib import cm
from scipy import interpolate
from scipy import stats
from scipy.ndimage import uniform_filter
from  scipy.optimize import curve_fit
import geopy.distance
import xarray as xr
import seaborn as sns
from tool_code import *
import os, pickle 
from matplotlib import font_manager
from oco_satellite import satellite_download
import matplotlib.image as mpl_img
from haversine import haversine, Unit, haversine_vector
from matplotlib import cm, colors
import uncertainties.unumpy as unp
import uncertainties as unc

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
    
def nan_array(shape, dtype):
    tmp = np.zeros(shape, dtype=dtype)
    tmp.fill(np.nan)
    return tmp

def near_rad_calc(OCO_class):
    array_size = (OCO_class.lat2d.shape[0], OCO_class.lat2d.shape[1], OCO_class.lam.size)
    rad_mca_ipa0_5 = nan_array(array_size, dtype=np.float64)
    rad_mca_3d_5   = nan_array(array_size, dtype=np.float64)
    rad_mca_ipa0_5_std =  nan_array(array_size, dtype=np.float64)
    rad_mca_3d_5_std   =  nan_array(array_size, dtype=np.float64)

    rad_mca_ipa0_9 = nan_array(array_size, dtype=np.float64)
    rad_mca_3d_9 = nan_array(array_size, dtype=np.float64)
    rad_mca_ipa0_9_std = nan_array(array_size, dtype=np.float64)
    rad_mca_3d_9_std = nan_array(array_size, dtype=np.float64)

    rad_mca_ipa0_13 = nan_array(array_size, dtype=np.float64)
    rad_mca_3d_13 = nan_array(array_size, dtype=np.float64)
    rad_mca_ipa0_13_std = nan_array(array_size, dtype=np.float64)
    rad_mca_3d_13_std = nan_array(array_size, dtype=np.float64)

    rad_mca_ipa0_41 = nan_array(array_size, dtype=np.float64)
    rad_mca_3d_41 = nan_array(array_size, dtype=np.float64)
    rad_mca_ipa0_41_std = nan_array(array_size, dtype=np.float64)
    rad_mca_3d_41_std = nan_array(array_size, dtype=np.float64)

    rad_mca_ipa0_5, rad_mca_3d_5,\
    rad_mca_ipa0_5_std, rad_mca_3d_5_std  = coarsening(OCO_class, size=5)
    
    rad_mca_ipa0_9, rad_mca_3d_9,\
    rad_mca_ipa0_9_std, rad_mca_3d_9_std = coarsening(OCO_class, size=9)
    
    rad_mca_ipa0_13, rad_mca_3d_13,\
    rad_mca_ipa0_13_std, rad_mca_3d_13_std = coarsening(OCO_class, size=13)
    
    rad_mca_ipa0_41, rad_mca_3d_41,\
    rad_mca_ipa0_41_std, rad_mca_3d_41_std = coarsening(OCO_class, size=41)


    OCO_class.rad_clr_5 = rad_mca_ipa0_5
    OCO_class.rad_c3d_5 = rad_mca_3d_5
    OCO_class.rad_clrs_5 = rad_mca_ipa0_5_std
    OCO_class.rad_c3ds_5 = rad_mca_3d_5_std

    OCO_class.rad_clr_9 = rad_mca_ipa0_9
    OCO_class.rad_c3d_9 = rad_mca_3d_9
    OCO_class.rad_clrs_9 = rad_mca_ipa0_9_std
    OCO_class.rad_c3ds_9 = rad_mca_3d_9_std

    OCO_class.rad_clr_13 = rad_mca_ipa0_13
    OCO_class.rad_c3d_13 = rad_mca_3d_13
    OCO_class.rad_clrs_13 = rad_mca_ipa0_13_std
    OCO_class.rad_c3ds_13 = rad_mca_3d_13_std

    OCO_class.rad_clr_41 = rad_mca_ipa0_41
    OCO_class.rad_c3d_41 = rad_mca_3d_41
    OCO_class.rad_clrs_41 = rad_mca_ipa0_41_std
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
    ipa0 = coarsening_subfunction(OCO_class.rad_clr, OCO_class.cld_location, size)
    c3d  = coarsening_subfunction(OCO_class.rad_c3d, OCO_class.cld_location, size)
    ipa0_std = coarsening_subfunction(OCO_class.rad_clrs, OCO_class.cld_location, size)
    c3d_std   = coarsening_subfunction(OCO_class.rad_c3ds, OCO_class.cld_location, size)
    
    # return ipa0, c3d, ipa0_std, c3d_std
    return ipa0, c3d, ipa0_std, c3d_std


def coarsening_subfunction(rad_mca, cld_position, size):
    """
    Parameters:
    -----------
    rad_mca: 3D array, radiances of various wavelengths
    cld_position: 2D array
    size: int, size of the filter
    """
    lams = rad_mca.shape[-1]
    tmp = np.zeros_like(rad_mca)
    rad_mca_mask_cld = rad_mca.copy()
    rad_mca_mask_cld[cld_position] = -999999
    for i in range(lams):
        tmp[:,:,i] = uniform_filter(rad_mca_mask_cld[:,:,i], size=size, mode='constant', cval=-999999)
    tmp[tmp<0] = np.nan
    
    return tmp

def get_slope_np(toa, sl_np, sls_np, c3d_np, clr_np, fp, z, points=11, mode='unperturb'):
     
    nwl=sls_np[z,fp,:].shape[0]
    flt=np.where(sls_np[z,fp,:]>1e-6)
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

def slopes_propagation(OCO_class, mode='unperturb'): # goes through entire line for a given footprint fp
    array_size = [OCO_class.rad_clr_5.shape[0],OCO_class.rad_clr_5.shape[1], 2]
    OCO_class.slope_5avg, OCO_class.inter_5avg = (nan_array(array_size, dtype=np.float64),)*2
    OCO_class.slope_9avg, OCO_class.inter_9avg = (nan_array(array_size, dtype=np.float64),)*2
    OCO_class.slope_13avg, OCO_class.inter_13avg = (nan_array(array_size, dtype=np.float64),)*2
    OCO_class.slope_41avg, OCO_class.inter_41avg = (nan_array(array_size, dtype=np.float64),)*2

    for z in range(OCO_class.rad_clr_5.shape[0]):
        for fp in range(OCO_class.rad_clr_5.shape[1]):   
            slope, slopestd, inter, interstd = get_slope_np(OCO_class.toa, OCO_class.sl_5, OCO_class.sls_5, OCO_class.rad_c3d_5, OCO_class.rad_clr_5, fp, z, points=11, mode='unperturb')
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

def cld_rad_slope_calc(band_tag, id_num, filename, pkl_filename, cld_location, ):
    h5_file  = filename.format(band_tag, id_num)
    OCO_class = OCOSIM(h5_file)
    OCO_class.cld_location = cld_location
    near_rad_calc(OCO_class)
    slopes_propagation(OCO_class)
    with open(pkl_filename.format(band_tag), 'wb') as pkl_file:
        pickle.dump(OCO_class, pkl_file)
    return OCO_class


def main(cfg_name='20181018_central_asia_2_test4.csv'):

    cfg_dir = '../simulation_dxdy/cfg'

    cfg_info = grab_cfg(f'{cfg_dir}/{cfg_name}')
    print(cfg_info.keys())
    if 'o2' in cfg_info.keys():
        id_num = output_h5_info(f'{cfg_dir}/{cfg_name}', 'o2')[22:31]
        boundary = [[float(i) for i in cfg_info['subdomain']], 'r']
    else:
        boundary = [[float(i) for i in cfg_info['subdomain']], 'orange']

    compare_num = 5
    rad_c3d_compare = f'rad_c3d_{compare_num}'
    rad_clr_compare = f'rad_clr_{compare_num}'
    slope_compare = f'slope_{compare_num}avg'
    inter_compare = f'inter_{compare_num}avg'

    if 1:#not os.path.isfile(f'o2a_para_{compare_num}_central_asia_2.csv'):
        filename = '../simulation_dxdy/data_all_20181018_{}_{}_lbl_3.h5'

        # filename = '../simulation_dxdy/data_all_20181018_{}_{}_test_3.h5'
        # filename = '../simulation_dxdy/data_all_20181018_{}_{}_photon_5e8_no_aod.h5'
        #filename = '../simulation_dxdy/data_all_20181018_{}_{}_photon_1e9_with_aod.h5'
        
        pkl_filename = '20181018_central_asia_2_test4_{}_lbl_3.pkl'
        if 1:#not os.path.isfile(pkl_filename.format('o2a')):
            _, _, cld_location = cld_position(cfg_name)
            o1 = cld_rad_slope_calc('o2a', id_num, filename, pkl_filename, cld_location)
            o2 = cld_rad_slope_calc('wco2', id_num, filename, pkl_filename, cld_location)
            o3 = cld_rad_slope_calc('sco2', id_num, filename, pkl_filename, cld_location)
        else:
            with open(pkl_filename.format('o2a'), 'rb') as f:
                o1 = pickle.load(f)
            with open(pkl_filename.format('wco2'), 'rb') as f:
                o2 = pickle.load(f)
            with open(pkl_filename.format('sco2'), 'rb') as f:
                o3 = pickle.load(f)

        if not os.path.isfile(f'{cfg_name[:-4]}_cld_distance.pkl'):
            cld_dist_calc(cfg_name, o2, slope_compare)
        cld_data = pd.read_pickle(f'{cfg_name[:-4]}_cld_distance.pkl')
        cld_dist = cld_data['cld_dis']


        # weighted_cld_dist_calc
        #--------------------------------------
        """
        if not os.path.isfile(f'{cfg_name[:-4]}_weighted_cld_distance.pkl'):
            weighted_cld_dist_calc(cfg_name, o2, slope_compare)
        cld_data = pd.read_pickle(f'{cfg_name[:-4]}_weighted_cld_distance.pkl')
        cld_dist = cld_data['cld_dis']
        plt.scatter(cld_data['lon'], cld_data['lat'], c=cld_dist)
        plt.colorbar()
        plt.show()
        print(cld_dist)
        print(cld_dist[cld_dist>0].min(), cld_dist.max())
        #"""
        #--------------------------------------
        
        xco2 = o1.co2
        psur = o1.psur
        snd = o1.snd
        xco2_valid = xco2>0

        extent = [float(loc) for loc in cfg_info['subdomain']]
        mask_fp = np.logical_and(np.logical_and(o1.lon[xco2_valid] >= extent[0], o1.lon[xco2_valid] <= extent[1]),
                              np.logical_and(o1.lat[xco2_valid] >= extent[2], o1.lat[xco2_valid] <= extent[3]))


        f_cld_distance = interpolate.RegularGridInterpolator((np.array(cld_data['lon']).reshape(o1.lon2d.shape)[:, 0], 
                                                              np.array(cld_data['lat']).reshape(o1.lon2d.shape)[0, :]),
                                                             np.array(cld_data['cld_dis']).reshape(o1.lon2d.shape), method='linear')
        
        points_footprint = np.column_stack((o1.lon[xco2_valid][mask_fp].flatten(), o1.lat[xco2_valid][mask_fp].flatten()))
        oco_footprint_cld_distance = f_cld_distance(points_footprint)

        #""" 
        extent = [float(loc) for loc in cfg_info['subdomain']]
        mask = np.logical_and(np.logical_and(o1.lon2d >= extent[0], o1.lon2d <= extent[1]),
                              np.logical_and(o1.lat2d >= extent[2], o1.lat2d <= extent[3]))
        mask = mask.flatten()
        parameters_cld_distance_list = fitting_3bands(cld_dist, o1, o2, o3, rad_c3d_compare, rad_clr_compare, slope_compare, inter_compare, mask)
        print(parameters_cld_distance_list)

        
        # fitting_3bands(cld_dist, o1, o2, o3, rad_c3d_compare, rad_clr_compare, slope_compare, inter_compare, mask, weighted=True)

        
       # slope_a, slope_b, inter_a, inter_b
        print(f'oco_footprint_cld_distance shape: {oco_footprint_cld_distance.shape}')
        o2a_inter = exp_decay_func(oco_footprint_cld_distance, parameters_cld_distance_list[0][2], parameters_cld_distance_list[0][3])
        o2a_slope = exp_decay_func(oco_footprint_cld_distance, parameters_cld_distance_list[0][0], parameters_cld_distance_list[0][1])

        wco2_inter = exp_decay_func(oco_footprint_cld_distance, parameters_cld_distance_list[1][2], parameters_cld_distance_list[1][3])
        wco2_slope = exp_decay_func(oco_footprint_cld_distance, parameters_cld_distance_list[1][0], parameters_cld_distance_list[1][1])

        sco2_inter = exp_decay_func(oco_footprint_cld_distance, parameters_cld_distance_list[2][2], parameters_cld_distance_list[2][3])
        sco2_slope = exp_decay_func(oco_footprint_cld_distance, parameters_cld_distance_list[2][0], parameters_cld_distance_list[2][1])

        output_csv = pd.DataFrame({'SND': snd[xco2_valid][mask_fp].flatten(),
                                   'LON': o1.lon[xco2_valid][mask_fp].flatten(),
                                   'LAT': o1.lat[xco2_valid][mask_fp].flatten(),
                                   'L2XCO2[ppm]': xco2[xco2_valid][mask_fp].flatten()*1e6,
                                   'L2PSUR[kPa]': psur[xco2_valid][mask_fp].flatten()/1000,
                                   #'cld_distance': oco_footprint_cld_distance.flatten(),
                                   'i1': o2a_inter,
                                   's1': o2a_slope,
                                   'i2': wco2_inter,
                                   's2': wco2_slope,
                                   'i3': sco2_inter,
                                   's3': sco2_slope
                                   },)
        output_csv['SND'] = output_csv['SND']#.apply(lambda x: f'SND{x:.0f}')
        output_csv.to_csv(f'central_asia_2_footprint_cld_distance.csv', index=False)


        #sys.exit()


        output_csv['SND'] = output_csv['SND'].apply(lambda x: f'SND{x:.0f}')
        output_csv.to_csv(f'central_asia_2_footprint_cld_distance.csv', index=False)

                
        # o2_slope_a, o2_slope_b, o2_inter_a, o2_inter_b = fitting(cld_dist[mask], getattr(o1, rad_c3d_compare)[:,:, -1].flatten()[mask], getattr(o1, rad_clr_compare)[:,:, -1].flatten()[mask], getattr(o1, slope_compare)[:,:,0].flatten()[mask], getattr(o1, inter_compare)[:,:,0].flatten()[mask],
        #                                                 band=f'O_2-A_{compare_num}',  plot=True)
        
        # o2a_slope_a_list.append(o2_slope_a)
        # o2a_slope_b_list.append(o2_slope_b)
        # o2a_inter_a_list.append(o2_inter_a)
        # o2a_inter_b_list.append(o2_inter_b)

        # wco2_slope_a, wco2_slope_b, wco2_inter_a, wco2_inter_b = fitting(cld_dist[mask], getattr(o2, rad_c3d_compare)[:,:, -1].flatten()[mask], getattr(o2, rad_clr_compare)[:,:, -1].flatten()[mask], getattr(o2, slope_compare)[:,:,0].flatten()[mask], getattr(o2, inter_compare)[:,:,0].flatten()[mask],
        #                                                 band=f'WCO_2_{compare_num}', plot=True)
        # wco2_slope_a_list.append(wco2_slope_a)
        # wco2_slope_b_list.append(wco2_slope_b)
        # wco2_inter_a_list.append(wco2_inter_a)
        # wco2_inter_b_list.append(wco2_inter_b)

        # sco2_slope_a, sco2_slope_b, sco2_inter_a, sco2_inter_b = fitting(cld_dist[mask], getattr(o3, rad_c3d_compare)[:,:, -1].flatten()[mask], getattr(o3, rad_clr_compare)[:,:, -1].flatten()[mask], getattr(o3, slope_compare)[:,:,0].flatten()[mask], getattr(o3, inter_compare)[:,:,0].flatten()[mask],
        #                                                 band=f'SCO_2_{compare_num}', plot=True)

        #"""
        # sco2_slope_a_list.append(sco2_slope_a)
        # sco2_slope_b_list.append(sco2_slope_b)
        # sco2_inter_a_list.append(sco2_inter_a)
        # sco2_inter_b_list.append(sco2_inter_b)

    
        # out_o2a = pd.DataFrame(np.array([alb_list, sza_list, o2a_slope_a_list, o2a_slope_b_list, o2a_inter_a_list, o2a_inter_b_list]).T,
        #                 columns=['albedo', 'sza', 'o2a_slope_a', 'o2a_slope_b', 'o2a_inter_a', 'o2a_inter_b'])
        # out_o2a.to_csv(f'o2a_para_{compare_num}_central_asia_lbl.csv')

        # out_wco2 = pd.DataFrame(np.array([alb_list, sza_list, wco2_slope_a_list, wco2_slope_b_list, wco2_inter_a_list, wco2_inter_b_list]).T,
        #                 columns=['albedo', 'sza', 'wco2_slope_a', 'wco2_slope_b', 'wco2_inter_a', 'wco2_inter_b'])
        # out_wco2.to_csv(f'wco2_para_{compare_num}_central_asia_lbl.csv')

        # out_sco2 = pd.DataFrame(np.array([alb_list, sza_list, sco2_slope_a_list, sco2_slope_b_list, sco2_inter_a_list, sco2_inter_b_list]).T,
        #                 columns=['albedo', 'sza', 'sco2_slope_a', 'sco2_slope_b', 'sco2_inter_a', 'sco2_inter_b'])
        # out_sco2.to_csv(f'sco2_para_{compare_num}_central_asia_lbl.csv')
        # print(out_wco2)
    else:
        None
        # out_o2a = pd.read_csv(f'o2a_para_{compare_num}_central_asia_lbl.csv')
        # out_wco2 = pd.read_csv(f'wco2_para_{compare_num}_central_asia_lbl.csv')
        # out_sco2 = pd.read_csv(f'sco2_para_{compare_num}_central_asia_lbl.csv')

    cld_lon, cld_lat, cld_location = cld_position(cfg_name)
    date   = datetime.datetime(int(cfg_info['date'][:4]),    # year
                               int(cfg_info['date'][4:6]),   # month
                               int(cfg_info['date'][6:])     # day
                              )
    
    cfg_name = cfg_info['cfg_name']
    case_name_tag = '%s_%s' % (cfg_info['cfg_name'], date.strftime('%Y%m%d'))
    extent_png = [float(loc) + offset for loc, offset in zip(cfg_info['subdomain'], [-0.15, 0.15, -0.15, 0.15])]
    extent_analysis = [float(loc) for loc in cfg_info['subdomain']]

    data = {}
    with h5py.File(f'../simulation_dxdy/data/{case_name_tag}/pre-data.h5', 'r') as f:
        lon_2d = f['lon'][...]
        lat_2d = f['lat'][...]
        sfh_2d = f['mod/geo/sfh'][...]
        rad650_2d = f[f'mod/rad/rad_650'][...]
        cth0 = f['mod/cld/cth_ipa'][...]
    

    title_size = 16
    label_size = 14
    legend_size = 14
    tick_size = 12
    extent = [float(loc) for loc in cfg_info['subdomain']]
    mask = np.logical_and(np.logical_and(lon_2d >= extent[0], lon_2d <= extent[1]),
                          np.logical_and(lat_2d >= extent[2], lat_2d <= extent[3]))
    print(f'average sfh: {np.mean(sfh_2d[mask])}')


    img_file = '../simulation_dxdy/data/20181018_central_asia_2_470cloud_test2_20181018/aqua_rgb_2018-10-18_55.00-55.60-33.70-34.45.png'
    wesn= extent_png
    img = mpimg.imread(img_file)
    lon_dom = extent_analysis[:2]
    lat_dom = extent_analysis[2:]


    sfc_alt_plt(cfg_name,
                img, wesn, lon_dom, lat_dom, 
                lon_2d, lat_2d, sfh_2d)
    
    cld_dist_plot(o1, cfg_name,
                  img, wesn, lon_dom, lat_dom, 
                  lon_2d, lat_2d, cth0, cld_dist)
    
    slope_intercept_compare_plot(o1, 'O_2-A', 'o2a', cfg_name,
                                img, wesn, lon_dom, lat_dom, 
                                lon_2d, lat_2d, cth0, 
                                slope_compare, inter_compare)
    
    slope_intercept_compare_plot(o3, 'SCO_2', 'sco2', cfg_name,
                                img, wesn, lon_dom, lat_dom, 
                                lon_2d, lat_2d, cth0, 
                                slope_compare, inter_compare)
    
    o2a_conti_plot(o1, rad_c3d_compare, cfg_name,
                   img, wesn, lon_dom, lat_dom, label_size=14)
    
    continuum_fp_compare_plot(o1, o2, o3, cfg_name,
                              img, wesn, lon_dom, lat_dom, 
                              lon_2d, lat_2d, cth0, )
    
    


def ax_lon_lat_label(ax, label_size=14, tick_size=12):
    ax.set_xlabel('Longitude ($^\circ$E)', fontsize=label_size)
    ax.set_ylabel('Latitude ($^\circ$N)', fontsize=label_size)
    ax.tick_params(axis='both', labelsize=tick_size)

def sfc_alt_plt(cfg_name,
                img, wesn, lon_dom, lat_dom, 
                lon_2d, lat_2d, sfh_2d, label_size=14):
    f, ax=plt.subplots(figsize=(8, 8))
    ax.imshow(img, extent=wesn)
    ax.vlines(lon_dom, ymin=lat_dom[0], ymax=lat_dom[1], color='k', linewidth=1)
    ax.hlines(lat_dom, xmin=lon_dom[0], xmax=lon_dom[1], color='k', linewidth=1)
    c = ax.contourf(lon_2d,lat_2d, sfh_2d*1000,
                   cmap='terrain', levels=201, vmin=0, vmax=2000)
    cbar = f.colorbar(c, ax=ax, extend='both')
    cbar.set_label('Surface altitude (m)', fontsize=label_size)
    ax_lon_lat_label(ax, label_size=14, tick_size=12)
    f.tight_layout()
    f.savefig(f'{cfg_name}_surface_altitude.png', dpi=300)


def cld_dist_plot(o1, cfg_name,
                  img, wesn, lon_dom, lat_dom, 
                  lon_2d, lat_2d, cth0, cld_dist, label_size=14):
    f, ax=plt.subplots(figsize=(8, 8))
    ax.imshow(img, extent=wesn)
    ax.vlines(lon_dom, ymin=lat_dom[0], ymax=lat_dom[1], color='k', linewidth=1)
    ax.hlines(lat_dom, xmin=lon_dom[0], xmax=lon_dom[1], color='k', linewidth=1)
    c = ax.scatter(o1.lon2d, o1.lat2d, 
                   c=cld_dist, s=5,
                   cmap='Reds', vmin=0, vmax=20)
    ax.scatter(lon_2d[cth0>0], lat_2d[cth0>0], s=15, color='b')
    cbar = f.colorbar(c, ax=ax, extend='both')
    cbar.set_label('Cloud distance (km)', fontsize=label_size)
    ax_lon_lat_label(ax, label_size=14, tick_size=12)
    f.tight_layout()
    f.savefig(f'{cfg_name}_cloud_distance.png', dpi=300)


def o2a_conti_plot(o1, rad_c3d_compare, cfg_name,
                   img, wesn, lon_dom, lat_dom, label_size=14):
    f, ax=plt.subplots(figsize=(8, 8))
    ax.imshow(img, extent=wesn)
    ax.vlines(lon_dom, ymin=lat_dom[0], ymax=lat_dom[1], color='k', linewidth=1)
    ax.hlines(lat_dom, xmin=lon_dom[0], xmax=lon_dom[1], color='k', linewidth=1)
    mask = np.isnan(getattr(o1, rad_c3d_compare)[:,:,-1])
    print(mask.sum())
    c = ax.scatter(o1.lon2d, o1.lat2d, 
                   c=getattr(o1, rad_c3d_compare)[:,:,-1], s=5, cmap='Reds')
    ax.scatter(o1.lon2d[mask], o1.lat2d[mask], 
                   c='grey', s=5, cmap='Reds')
    cbar = f.colorbar(c, ax=ax, extend='both')
    cbar.set_label('$\mathrm{O_2-A}$ continuum (mW m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=label_size)
    ax_lon_lat_label(ax, label_size=14, tick_size=12)
    f.tight_layout()
    f.savefig(f'{cfg_name}_o2a_conti_{rad_c3d_compare}.png', dpi=300)

def slope_intercept_compare_plot(OCO_class, label_tag, file_tag, cfg_name,
                                img, wesn, lon_dom, lat_dom, 
                                lon_2d, lat_2d, cth0, 
                                slope_compare, inter_compare, label_size=14):
    
    f, (ax1, ax2) =plt.subplots(1, 2, figsize=(16, 9))
    for ax in [ax1, ax2]:
        ax.imshow(img, extent=wesn)
        ax.set_xlim(np.min(lon_dom), np.max(lon_dom))
        ax.set_ylim(np.min(lat_dom), np.max(lat_dom))
        ax.scatter(lon_2d[cth0>0], lat_2d[cth0>0], s=15, color='r')
        ax_lon_lat_label(ax, label_size=14, tick_size=12)
    mask = ~(cth0>0)
    c1 = ax1.scatter(OCO_class.lon2d[mask], OCO_class.lat2d[mask], 
                   c=getattr(OCO_class, slope_compare)[:,:,0][mask], s=10,
                   cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    cbar1 = f.colorbar(c1, ax=ax1, extend='both')
    cbar1.set_label('$\mathrm{%s}$ slope' %(label_tag), fontsize=label_size)

    c2 = ax2.scatter(OCO_class.lon2d[mask], OCO_class.lat2d[mask], 
                   c=getattr(OCO_class, inter_compare)[:,:,0][mask], s=10,
                   cmap='RdBu_r', vmin=-0.15, vmax=0.15)
    cbar2 = f.colorbar(c2, ax=ax2, extend='both')
    cbar2.set_label('$\mathrm{%s}$ intercept' %(label_tag), fontsize=label_size)
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    ax1.text(xmin+0.0*(xmax-xmin), ymin+1.015*(ymax-ymin), '(a)', fontsize=label_size+4, color='k')

    xmin, xmax = ax2.get_xlim()
    ymin, ymax = ax2.get_ylim()
    ax2.text(xmin+0.0*(xmax-xmin), ymin+1.015*(ymax-ymin), '(b)', fontsize=label_size+4, color='k')
    
    f.tight_layout()
    f.savefig(f'{cfg_name}_{file_tag}_{slope_compare}.png', dpi=300)


def continuum_fp_compare_plot(o1, o2, o3, cfg_name,
                              img, wesn, lon_dom, lat_dom, 
                              lon_2d, lat_2d, cth0, 
                              tick_size=12, label_size=14):
    f, (ax1, ax2, ax3)=plt.subplots(1, 3, figsize=(24, 7.5))
    for ax in [ax1, ax2, ax3]:
        ax.imshow(img, extent=wesn)
        ax.set_xlim(np.min(lon_dom), np.max(lon_dom))
        ax.set_ylim(np.min(lat_dom), np.max(lat_dom))
        ax.scatter(lon_2d[cth0>0], lat_2d[cth0>0], s=15, color='r')
        ax_lon_lat_label(ax, label_size=14, tick_size=12)

    # ax1
    vmin, vmax = 0.03, 0.13
    lev = np.arange(vmin, vmax+1e-7, 0.001)
    rad_to_plot = o1.rad_c3d[:,:,10].copy()
    rad_to_plot[rad_to_plot>vmax] = vmax

    l1b_lon, l1b_lat, l1b_continuum = [], [], []
    for i in range(o1.lon.shape[0]):
        for j in range(o1.lon.shape[1]):
            l1b_lon.append(o1.lon[i, j])
            l1b_lat.append(o1.lat[i, j])
            l1b_continuum.append(o1.l1b[i, j, np.argmin(np.abs(o1.wvl[i, j, :]-o1.lam[10]))])
    cc1 = ax1.contourf(o1.lon2d, o1.lat2d, rad_to_plot, lev, cmap='jet',
                    vmin=vmin, vmax=vmax, extend='both', alpha=1)
    scatter_arg = {'s': 60, 'cmap': 'jet', 'marker': 'o', 'edgecolors': 'k'}
    mask = (o1.lat.flatten()*1e6)>0
    ax1.scatter(np.array(l1b_lon)[mask], np.array(l1b_lat)[mask],
                c=np.array(l1b_continuum)[mask],
                vmin=vmin, vmax=vmax,
                **scatter_arg)

    cbar1 = f.colorbar(cc1, ax=ax1)#, extend='both')
    cbar1.set_label('3D radiance', fontsize=16)
    ax1.set_title(f'{o1.lam[10]:.3f}nm')

    # ax2
    vmin, vmax = 0.0, 0.04
    lev = np.arange(vmin, vmax+1e-7, 0.0005)
    rad_to_plot = o2.rad_c3d[:,:,10].copy()
    rad_to_plot[rad_to_plot>vmax] = vmax

    l1b_lon, l1b_lat, l1b_continuum = [], [], []
    for i in range(o2.lon.shape[0]):
        for j in range(o2.lon.shape[1]):
            l1b_lon.append(o2.lon[i, j])
            l1b_lat.append(o2.lat[i, j])
            l1b_continuum.append(o2.l1b[i, j, np.argmin(np.abs(o2.wvl[i, j, :]-o2.lam[10]))])
    cc2 = ax2.contourf(o2.lon2d, o2.lat2d, rad_to_plot, lev, cmap='jet',
                    vmin=vmin, vmax=vmax, extend='both', alpha=1)

    mask = (o2.lat.flatten()*1e6)>0
    ax2.scatter(np.array(l1b_lon)[mask], np.array(l1b_lat)[mask],
                c=np.array(l1b_continuum)[mask],
                vmin=vmin, vmax=vmax,
                **scatter_arg)
    cbar2 = f.colorbar(cc2, ax=ax2)#, extend='both')
    cbar2.set_label('3D radiance', fontsize=16)
    ax2.set_title(f'{o2.lam[10]:.3f}nm')

    # ax3
    vmin, vmax = 0.00, 0.01
    lev = np.arange(vmin, vmax+1e-7, 0.0001)
    rad_to_plot = o3.rad_c3d[:,:,10].copy()
    rad_to_plot[rad_to_plot>vmax] = vmax

    l1b_lon, l1b_lat, l1b_continuum = [], [], []
    for i in range(o1.lon.shape[0]):
        for j in range(o3.lon.shape[1]):
            l1b_lon.append(o3.lon[i, j])
            l1b_lat.append(o3.lat[i, j])
            l1b_continuum.append(o3.l1b[i, j, np.argmin(np.abs(o3.wvl[i, j, :]-o3.lam[10]))])
    cc3 = ax3.contourf(o3.lon2d, o3.lat2d, rad_to_plot, lev, cmap='jet',
                    vmin=vmin, vmax=vmax, extend='both', alpha=1)

    mask = (o3.lat.flatten()*1e6)>0
    ax3.scatter(np.array(l1b_lon)[mask], np.array(l1b_lat)[mask],
                c=np.array(l1b_continuum)[mask],
                vmin=vmin, vmax=vmax,
                **scatter_arg)
    cbar3 = f.colorbar(cc3, ax=ax3)#, extend='both')
    cbar3.set_label('3D radiance', fontsize=16)
    ax3.set_title(f'{o3.lam[10]:.3f}nm')
    f.tight_layout()
    f.savefig(f'{cfg_name,}_continuum_fp_compare.png', dpi=300)

    # plt.show()


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
    
    popt, pcov = curve_fit(exp_decay_func_with_intercept, xx_o2a, yy_o2a, bounds=([-5, 0., 0], [5, 50, 0.3]),
                        #p0=(0.1, 0.7),
                        maxfev=3000,
                        #sigma=value_std[val_mask], 
                        #absolute_sigma=True,
                        )
    residuals = yy_o2a - exp_decay_func_with_intercept(xx_o2a, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy_o2a-np.mean(yy_o2a))**2)
    r_squared = 1 - (ss_res / ss_tot)

    plot_xx = np.arange(0.05, xx.max()+0.05, 0.01)
    ax1.plot(plot_xx, exp_decay_func_with_intercept(plot_xx, *popt), '--', color='tab:blue', 
              label='o2a\nfit: a=%5.3f\n     b=%5.1f\n     c=%5.3f' % tuple(popt), linewidth=1.5, alpha=0.5)
    
    yy = wco2_slope_a
    mask = ~(np.isnan(xx) | np.isnan(yy) | np.isinf(xx) | np.isinf(yy))
    xx_wco2, yy_wco2 = xx[mask], yy[mask]

    popt, pcov = curve_fit(exp_decay_func_with_intercept, xx_wco2, yy_wco2, bounds=([-5, 0., 0], [5, 50, 0.3]),
                        #p0=(0.1, 0.7),
                        maxfev=3000,
                        #sigma=value_std[val_mask], 
                        #absolute_sigma=True,
                        )
    residuals = yy_wco2 - exp_decay_func_with_intercept(xx_wco2, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy_wco2-np.mean(yy_wco2))**2)
    r_squared = 1 - (ss_res / ss_tot)

    plot_xx = np.arange(0.05, xx.max()+0.05, 0.01)
    ax1.plot(plot_xx, exp_decay_func_with_intercept(plot_xx, *popt), '--', color='tab:orange', 
              label='wco2\nfit: a=%5.3f\n     b=%5.1f\n     c=%5.3f' % tuple(popt), linewidth=1.5, alpha=0.5)
    
    yy = sco2_slope_a
    mask = ~(np.isnan(xx) | np.isnan(yy) | np.isinf(xx) | np.isinf(yy))
    xx_sco2, yy_sco2 = xx[mask], yy[mask]
    
    popt, pcov = curve_fit(exp_decay_func_with_intercept, xx_sco2, yy_sco2, bounds=([-5, 0., 0], [5, 50, 0.3]),
                        #p0=(0.1, 0.7),
                        maxfev=3000,
                        #sigma=value_std[val_mask], 
                        #absolute_sigma=True,
                        )
    residuals = yy_sco2 - exp_decay_func_with_intercept(xx_sco2, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy_sco2-np.mean(yy_sco2))**2)
    r_squared = 1 - (ss_res / ss_tot)

    plot_xx = np.arange(0.05, xx.max()+0.05, 0.01)
    ax1.plot(plot_xx, exp_decay_func_with_intercept(plot_xx, *popt), '--', color='tab:green', 
              label='sco2\nfit: a=%5.3f\n     b=%5.1f\n     c=%5.3f' % tuple(popt), linewidth=1.5, alpha=0.5)

    ax1.legend()
    ax1.set_ylabel('a', fontsize=label_size)
    ax1.set_xlabel('surface albedo', fontsize=label_size)
    ax1.set_title('coefficient a for slope', fontsize=label_size)
    
    fig.tight_layout()
    fig.savefig(f'all_bands_slope_alb_{point_avg}avg.png', dpi=150)

    
def plot_alb_sza_relationship(sfc_alb, sza, inter_a_list, slope_a_list, inter_e_list, slope_e_list, band_tag, point_avg):
    fig, ((ax1, ax2),
        (ax21, ax22)) = plt.subplots(2, 2, figsize=(12, 10), sharex=False)
    fig.tight_layout(pad=5.0)
    label_size = 16

    for sza_val, marker in zip([30, 45, 60], ['X', 'p', 'o']):
        ax1.scatter(sfc_alb[sza==sza_val], slope_a_list[sza==sza_val], s=50, label=f'sza={sza_val}', marker=marker, alpha=0.65)
        ax2.scatter(sfc_alb[sza==sza_val], inter_a_list[sza==sza_val], s=50, label=f'sza={sza_val}', marker=marker, alpha=0.65)
        ax21.scatter(sfc_alb[sza==sza_val], slope_e_list[sza==sza_val], s=50, label=f'sza={sza_val}', marker=marker, alpha=0.65)
        ax22.scatter(sfc_alb[sza==sza_val], inter_e_list[sza==sza_val], s=50, label=f'sza={sza_val}', marker=marker, alpha=0.65)
   
    xx = sfc_alb[sza==30]
    yy = np.nanmean(np.array([slope_a_list[sza==30], slope_a_list[sza==45], slope_a_list[sza==60]]), axis=0)
    mask = ~(np.isnan(xx) | np.isnan(yy) | np.isinf(xx) | np.isinf(yy))
    xx, yy = xx[mask], yy[mask]

    popt, pcov = curve_fit(exp_decay_func_with_intercept, xx, yy, bounds=([-5, 0., 0], [5, 50, 0.3]),
                        #p0=(0.1, 0.7),
                        maxfev=3000,
                        #sigma=value_std[val_mask], 
                        #absolute_sigma=True,
                        )
    residuals = yy - exp_decay_func_with_intercept(xx, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy-np.mean(yy))**2)
    r_squared = 1 - (ss_res / ss_tot)

    plot_xx = np.arange(0.05, xx.max()+0.05, 0.01)
    ax1.plot(plot_xx, exp_decay_func_with_intercept(plot_xx, *popt), '--', color='r', 
              label='fit: a=%5.3f\n     b=%5.1f\n     c=%5.3f' % tuple(popt), linewidth=1.5, alpha=0.5)
    
    xx = sfc_alb[sza==30]
    yy = np.array([inter_a_list[sza==30], inter_a_list[sza==45], inter_a_list[sza==60]]).mean(axis=0)
    popt, pcov = curve_fit(exp_decay_func_with_intercept, xx, yy, bounds=([-5, 0., 0], [5, 50, 0.3]),
                        #p0=(0.1, 0.7),
                        maxfev=3000,)
    residuals = yy - exp_decay_func_with_intercept(xx, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy-np.mean(yy))**2)
    r_squared = 1 - (ss_res / ss_tot)
    plot_xx = np.arange(0.05, xx.max()+0.05, 0.01)
   
    b_inter_a = popt[1]
    

    xx = sfc_alb[sza==30]
    yy = inter_a_list[sza==30]
    popt, pcov = curve_fit(exp_decay_func_with_intercept, xx, yy, bounds=([-5, b_inter_a, 0], [5, b_inter_a*1.00001, 0.3]),
                        #p0=(0.1, 0.7),
                        maxfev=3000,
                        #sigma=value_std[val_mask], 
                        #absolute_sigma=True,
                        )
    residuals = yy - exp_decay_func_with_intercept(xx, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy-np.mean(yy))**2)
    r_squared = 1 - (ss_res / ss_tot)
    plot_xx = np.arange(0.05, xx.max()+0.05, 0.01)
    ax2.plot(plot_xx, exp_decay_func_with_intercept(plot_xx, *popt), '--', color='tab:blue', 
              label='fit: a=%5.3f\n     b=%5.1f\n     c=%5.3f' % tuple(popt), linewidth=1.5, alpha=0.5)
    
    yy = inter_a_list[sza==45]
    popt, pcov = curve_fit(exp_decay_func_with_intercept, xx, yy, bounds=([-5, b_inter_a, 0], [5, b_inter_a*1.00001, 0.3]),
                        #p0=(0.1, 0.7),
                        maxfev=3000,
                        #sigma=value_std[val_mask], 
                        #absolute_sigma=True,
                        )
    residuals = yy - exp_decay_func_with_intercept(xx, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy-np.mean(yy))**2)
    r_squared = 1 - (ss_res / ss_tot)
    plot_xx = np.arange(0.05, xx.max()+0.05, 0.01)
    ax2.plot(plot_xx, exp_decay_func_with_intercept(plot_xx, *popt), '--', color='tab:orange', 
              label='fit: a=%5.3f\n     b=%5.1f\n     c=%5.3f' % tuple(popt), linewidth=1.5, alpha=0.5)
    
    yy = inter_a_list[sza==60]
    popt, pcov = curve_fit(exp_decay_func_with_intercept, xx, yy, bounds=([-5, b_inter_a, 0], [5, b_inter_a*1.00001, 0.3]),
                        #p0=(0.1, 0.7),
                        maxfev=3000,
                        #sigma=value_std[val_mask], 
                        #absolute_sigma=True,
                        )
    residuals = yy - exp_decay_func_with_intercept(xx, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yy-np.mean(yy))**2)
    r_squared = 1 - (ss_res / ss_tot)
    plot_xx = np.arange(0.05, xx.max()+0.05, 0.01)
    ax2.plot(plot_xx, exp_decay_func_with_intercept(plot_xx, *popt), '--', color='tab:green', 
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

    for ax in [ax1, ax2, ax21, ax22]:
        ax.set_xlabel('surface albedo', fontsize=label_size)
    
    ax1.set_title('coefficient a for slope', fontsize=label_size)
    ax2.set_title('coefficient a for intercept', fontsize=label_size)
    ax21.set_title('e-folding distance for slope', fontsize=label_size)
    ax22.set_title('e-folding distance for intercept', fontsize=label_size)
    fig.tight_layout()
    fig.savefig(f'{band_tag}_slope_inter_cld_distance_{point_avg}avg.png', dpi=150)
    plt.show()
    

def cld_position(cfg_name):
    cldfile = f'../simulation_dxdy/data/{cfg_name[:-4]}_{cfg_name[:8]}/pre-data.h5'
    with h5py.File(cldfile, 'r') as f:
        lon_cld = f['lon'][...]
        lat_cld = f['lat'][...]
        cth = f[f'mod/cld/cth_ipa'][...]
        cld_list = cth>0
    return lon_cld, lat_cld, cld_list


def cld_dist_calc(cfg_name, o1, slope_compare):

    cldfile = f'../simulation_dxdy/data/{cfg_name[:-4]}_{cfg_name[:8]}/pre-data.h5'
    with h5py.File(cldfile, 'r') as f:
        cth = f[f'mod/cld/cth_ipa'][...]

    cld_list = cth>0
    cld_X, cld_Y = np.where(cld_list==1)[0], np.where(cld_list==1)[1]
    # cld_position = np.array(zip(cld_X, cld_Y))
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
                cld_x, cld_y = cld_position[min_ind]
                if cld_x==cloud_dist.shape[0] or cld_y==cloud_dist.shape[1]:
                    print(cld_x, cld_y)
                dist = geopy.distance.distance((o1.lat2d[cld_x, cld_y], o1.lon2d[cld_x, cld_y]), (o1.lat2d[i, j], o1.lon2d[i, j])).km
                cloud_dist[i, j] = dist
    
    output = np.array([o1.lon2d, o1.lat2d, cloud_dist, ])
    cld_slope_inter = pd.DataFrame(output.reshape(output.shape[0], output.shape[1]*output.shape[2]).T,
                                columns=['lon', 'lat', 'cld_dis', ])
    cld_slope_inter.to_pickle(f'{cfg_name[:-4]}_cld_distance.pkl')

    
def weighted_cld_dist_calc(cfg_name, o1, slope_compare):


    cldfile = f'../simulation_dxdy/data/{cfg_name[:-4]}_{cfg_name[:8]}/pre-data.h5'
    with h5py.File(cldfile, 'r') as f:
        lon_cld = f['lon'][...]
        lat_cld = f['lat'][...]
        cth = f[f'mod/cld/cth_ipa'][...]

    cld_list = cth>0
    cld_X, cld_Y = np.where(cld_list==1)[0], np.where(cld_list==1)[1]
    # cld_position = zip(cld_X, cld_Y)
    cld_position = []
    cld_latlon = []
    for i in range(len(cld_X)):
        cld_position.append(np.array([cld_X[i], cld_Y[i]]))
        cld_latlon.append([lat_cld[cld_X[i], cld_Y[i]], lon_cld[cld_X[i], cld_Y[i]]])
    cld_position = np.array(cld_position)
    cld_latlon = np.array(cld_latlon)

    cloud_dist = np.zeros_like(getattr(o1, slope_compare)[:,:,0])
    for j in range(cloud_dist.shape[1]):
        for i in range(cloud_dist.shape[0]):
            if cld_list[i, j] == 1:
                cloud_dist[i, j] = 0
            else:
                point = np.array([o1.lat2d[i, j], o1.lon2d[i, j]])
                if i==0 and j==0:
                    print(point)
                    print(cld_latlon[:5])
                # distances = np.array([haversine(point, p, unit=Unit.KILOMETERS) for p in cld_latlon])
                distances = haversine_vector(point, cld_latlon, unit=Unit.KILOMETERS, comb=True)
                # Calculate the inverse distance weights
                
                weights = 1 / distances**3 #np.exp(-distances)
                
                # Calculate the weighted average distance
                weighted_avg_distance = np.sum(distances * weights) / np.sum(weights)
                
                cloud_dist[i, j] = weighted_avg_distance
    
    output = np.array([o1.lon2d, o1.lat2d, cloud_dist, ])
    cld_slope_inter = pd.DataFrame(output.reshape(output.shape[0], output.shape[1]*output.shape[2]).T,
                                columns=['lon', 'lat', 'cld_dis', ])

    cld_slope_inter.to_pickle(f'{cfg_name[:-4]}_weighted_cld_distance.pkl')   



def heatmap_xy_3(x, y, ax):
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x, y = x[mask], y[mask]
    
    # cloud distance
    interval = 1/2
    start = 1
    
    ax.scatter(x[x>=start], y[x>=start], s=1, color='k')
    sns.kdeplot(x=x, y=y, cmap='hot_r', n_levels=20, fill=True, ax=ax, alpha=0.65)
    
    cld_levels = np.arange(start, 18, interval)
    value_avg, value_std = np.zeros(len(cld_levels)-1), np.zeros(len(cld_levels)-1)
    for i in range(len(cld_levels)-1):
        select = np.logical_and(x>=cld_levels[i], x < cld_levels[i+1])
        if select.sum()>0:
            value_avg[i] = np.percentile(y[select], 50)
            value_std[i] = np.percentile(y[select], 75)-np.percentile(y[select], 25)
        else:
            value_avg[i] = np.nan
            value_std[i] = np.nan
    cld_list = (cld_levels[:-1] + cld_levels[1:])/2
    
    ax.errorbar(cld_list, value_avg, yerr=value_std, 
                marker='s', color='r', linewidth=2, linestyle='', ecolor='skyblue')
    
    val_mask = ~(np.isnan(value_avg) | np.isnan(value_std) | np.isinf(value_avg) | np.isinf(value_std))
    temp_r2 = 0
    cld_val = cld_list[val_mask]
    cld_min_list = [1, 1.25, 1.5, 1.75] if cld_val.min()<=2 else [cld_val.min().round(0)-0.25, cld_val.min().round(0)-0.5, cld_val.min().round(0), cld_val.min().round(0)+0.25, cld_val.min().round(0)+0.5] 
    for cld_min in cld_min_list:
        for cld_max in np.arange(10, 18, 0.5):
            mask = np.logical_and(cld_val>=cld_min, cld_val<=cld_max)
            xx = cld_val[mask]
            yy = value_avg[val_mask][mask]
            popt, pcov = curve_fit(exp_decay_func, xx, yy, bounds=([-5, 1e-3], [5, 15,]),
                                p0=(0.1, 0.7),
                                maxfev=3000,
                                #sigma=value_std[val_mask], 
                                #absolute_sigma=True,
                                )
            residuals = yy - exp_decay_func(xx, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((yy-np.mean(yy))**2)
            r_squared = 1 - (ss_res / ss_tot)

            if r_squared > temp_r2:
                temp_r2 = r_squared
            else:
                break
    
    plot_xx = np.arange(0, cld_list.max()+0.75, 0.5)
    ax.plot(plot_xx, exp_decay_func(plot_xx, *popt), '--', color='limegreen', 
            label=f'fit: amplitude     = {popt[0]:.3f}\n     e-folding dis = {1/popt[1]:.2f}', linewidth=3.5)
    print('-'*15)
    print(f'E-folding dis: {1/popt[1]}')
    ax.legend()
    return popt


def exp_decay_func(x, a, b):
     return a * np.exp(-b * x)

def exp_decay_func_with_intercept(x, a, b, c):
     return a * np.exp(-b * x) + c


def fitting(cloud_dist, rad_3d, rad_clr, slope, inter, band, plot=False):

    if plot:
        fig, (ax11, ax12) = plt.subplots(1, 2, figsize=(12, 4), sharex=False)
        fig.tight_layout(pad=5.0)
        label_size = 16
        tick_size = 12

        mask = np.logical_and(cloud_dist > 0, rad_3d>rad_clr)

        slope_a, slope_b = heatmap_xy_3(cloud_dist[mask], slope[mask], ax11)
        inter_a, inter_b = heatmap_xy_3(cloud_dist[mask], inter[mask], ax12)

        for ax in [ax11, ax12]: 
            ax.set_xlabel('Cloud distance (km)', fontsize=label_size)
            ax.tick_params(axis='both', labelsize=tick_size)
            _, xmax = ax.get_xlim()
            ax.hlines(0, 0, xmax, linestyle='--', color='white')
            
        ax11.set_ylabel('$\mathrm{%s}$ slope' %(band), fontsize=label_size)
        ax12.set_ylabel('$\mathrm{%s}$ intercept' %(band), fontsize=label_size)
        
        cld_low, cld_max = 0, 15
        limit_1 = 0.3
        limit_2 = 0.15
        for ax in [ax11, ax12]:
            ax.set_xlim(cld_low, cld_max)
        ax11.set_ylim(-limit_1, limit_1)
        ax12.set_ylim(-limit_2, limit_2)
        fig.savefig(f'central_asia_test2_{band}.png', dpi=150, bbox_inches='tight')

    else:
        mask = np.logical_and(cloud_dist > 0, rad_3d>rad_clr)
        slope_a, slope_b = fitting_without_plot(cloud_dist[mask], slope[mask])
        inter_a, inter_b = fitting_without_plot(cloud_dist[mask], inter[mask])

    return slope_a, slope_b, inter_a, inter_b


def fitting_3bands(cloud_dist, o1, o2, o3, rad_3d_compare, rad_clr_compare, slope_compare, inter_compare, region_mask):

    return_list = []
    fig, ((ax11, ax12), 
            (ax21, ax22),
            (ax31, ax32)) = plt.subplots(3, 2, figsize=(12, 12), sharex=False)
    fig.tight_layout(pad=5.0)
    label_size = 16
    tick_size = 12

    ax_list = [(ax11, ax12), 
                (ax21, ax22),
                (ax31, ax32)]
    for i in range(3):
        oco_band = [o1, o2, o3][i]
        rad_3d = getattr(oco_band, rad_3d_compare)[:,:, -1].flatten()
        rad_clr = getattr(oco_band, rad_clr_compare)[:,:, -1].flatten()
        mask = np.logical_and(np.logical_and(cloud_dist > 0, rad_3d>rad_clr), region_mask)

        slope = getattr(oco_band, slope_compare)[:,:,0].flatten()
        inter = getattr(oco_band, inter_compare)[:,:,0].flatten()

        ax1, ax2 = ax_list[i]
        slope_a, slope_b = heatmap_xy_3(cloud_dist[mask], slope[mask], ax1)
        inter_a, inter_b = heatmap_xy_3(cloud_dist[mask], inter[mask], ax2)
        return_list.append((slope_a, slope_b, inter_a, inter_b))


    cld_low, cld_max = 0, 20
    limit_1 = 0.2
    limit_2 = 0.08
    for ax_l, ax_r in zip([ax11, ax21, ax31], [ax12, ax22, ax32]):
        ax_l.set_xlim(cld_low, cld_max)
        ax_l.set_ylim(-limit_1, limit_1)
        ax_r.set_xlim(cld_low, cld_max)
        ax_r.set_ylim(-limit_2, limit_2)

    ax11.set_ylim(-0.3, 0.3)
    ax12.set_ylim(-0.12, 0.12)

    label_list = ['a', 'b', 'c', 'd', 'e', 'f']
    ax_list = [ax11, ax12, ax21, ax31, ax22, ax32]
    for i in range(6):
        ax = ax_list[i]
        label_text = f'({label_list[i]})'
        ax.set_xlabel('Cloud distance (km)', fontsize=label_size)
        ax.tick_params(axis='both', labelsize=tick_size)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.hlines(0, 0, xmax, linestyle='--', color='white')
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.05*(ymax-ymin), label_text, fontsize=label_size, color='k')
    
    for ax_l, ax_r, band_tag in zip([ax11, ax21, ax31], [ax12, ax22, ax32], ['O_2-A', 'WCO_2', 'SCO_2']):
        ax_l.set_ylabel('$\mathrm{%s}$ slope' %(band_tag), fontsize=label_size)
        ax_r.set_ylabel('$\mathrm{%s}$ intercept' %(band_tag), fontsize=label_size)
    fig.savefig(f'central_asia_test2_all_band_{slope_compare.split("_")[-1]}.png', dpi=150, bbox_inches='tight')

    return return_list


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
            value_avg[i] = np.percentile(y[select], 50)
            value_std[i] = np.percentile(y[select], 75)-np.percentile(y[select], 25)
        else:
            value_avg[i] = np.nan
            value_std[i] = np.nan
    cld_list = (cld_levels[:-1] + cld_levels[1:])/2
    val_mask = ~(np.isnan(value_avg) | np.isnan(value_std) | np.isinf(value_avg) | np.isinf(value_std))
    
    temp_r2 = 0
    for cld_min in [1, 1.25, 1.5]:
        for cld_max in np.arange(3, 15, 0.5):
            cld_val = cld_list[val_mask]
            mask = np.logical_and(cld_val>=cld_min, cld_val<=cld_max)
            xx = cld_val[mask]
            yy = value_avg[val_mask][mask]
            popt, pcov = curve_fit(exp_decay_func, xx, yy, bounds=([-2, 0.], [2, 10,]),
                                p0=(0.1, 0.7),
                                maxfev=3000,
                                #sigma=value_std[val_mask], 
                                #absolute_sigma=True,
                                )
            residuals = yy - exp_decay_func(xx, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((yy-np.mean(yy))**2)
            r_squared = 1 - (ss_res / ss_tot)

            if r_squared > temp_r2:
                temp_r2 = r_squared
            else:
                break
    return popt



def o2a_wvl_select_slope_derivation(cfg_info, o1):
    cfg_name = cfg_info['cfg_name']
    date   = datetime.datetime(int(cfg_info['date'][:4]),    # year
                               int(cfg_info['date'][4:6]),   # month
                               int(cfg_info['date'][6:])     # day
                              )
    case_name_tag = '%s_%s' % (cfg_info['cfg_name'], date.strftime('%Y%m%d'))
    nx = int(cfg_info['nx'])
    with h5py.File(f'../simulation_dxdy/data/{case_name_tag}/atm_abs_o2a_{nx+1}.h5', 'r') as file:
        wvl = file['wl_oco'][...]
        trnsx = file['trns_oco'][...]
        oco_lam = file['lamx'][...]
        oco_tx = file['tx'][...]

    f, (ax1, ax2) =plt.subplots(1, 2, figsize=(14, 5))
    f.tight_layout(pad=3.0)

    title_size = 18
    label_size = 16
    legend_size = 16
    tick_size = 14

    refl = o1.sfc_alb
    # first fig
    x = np.arange(1016)
    sx = np.argsort(trnsx)
    y = trnsx[sx]*refl
    ax1.scatter(x, y, color='k', s=3)
    ax1.tick_params(axis='both', labelsize=tick_size)
    ax1.set_xlabel('Wavelength index', fontsize=label_size)
    ax1.set_ylabel('Transmittance', fontsize=label_size)

    # plot setting
    norm = colors.Normalize(vmin=0.0, vmax=255.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.rainbow)
    for i in range(nx+1):
        wli0 = np.where(sx==np.argmin(np.abs(y-oco_tx[i])))[0][0]
        ax1.plot([0,1016],[oco_tx[i],oco_tx[i]],color='orange',linestyle='dotted')
        ax1.plot([sx[wli0], sx[wli0]], [0,oco_tx[i]], linestyle='dashed', 
                 color=mapper.to_rgba(30*(i+1)), linewidth=2)

    # second fig
    toa = o1.toa
    mu = o1.mu
    sl_np = o1.sl_5
    sls_np = o1.sls_5/np.sqrt(3)
    c3d_np = o1.rad_c3d_5
    clr_np = o1.rad_clr_5
    fp, z = 160, 130
    points = nx+1

    w = 1./sls_np[z,fp,:] 
    x = c3d_np[z,fp,:]/(toa[:]*mu)*np.pi
    x_len = len(x)
    mask = np.argsort(x)[x_len-points:]
    res = np.polyfit(x[mask], sl_np[z,fp,:][mask], 1, w=w[mask], cov=True) # now get covariance as well!
    slope, intercept = res[0]
    slopes = np.sqrt(res[1][0][0])
    intercepts = np.sqrt(res[1][1][1])

    ax2.errorbar(x[mask], sl_np[z,fp,:][mask]*100, yerr=sls_np[z,fp,:]*100, color='k',
                ecolor='k',
                elinewidth=1,
                capsize=5,
                linewidth=0,
                marker='o', ms=5)
    yy=intercept+slope*x
    y1=intercept+intercepts+(slope+slopes)*x
    y2=intercept-intercepts+(slope-slopes)*x
    # ax.plot(x,yy*100,'r-',linewidth=2)  
    # ax.plot(x,y1*100,'r:',linewidth=1)  
    # ax.plot(x,y2*100,'r:',linewidth=1) 
    # ax.plot(x, (x*slope+intercept)*100, 'purple', label='prediction')


    ax2.tick_params(axis='both', labelsize=tick_size)
    ymin, ymax = ax2.get_ylim()
    xmin, xmax = ax2.get_xlim()
    xmin, xmax = 0., xmax*1.1
    ax2.set_xlim(0, xmax)

    ax2.set_xlabel('Reflectance', fontsize=label_size)
    ax2.set_ylabel('Perturbation (%)', fontsize=label_size)


    popt, pcov = res
    # calculate parameter confidence interval
    a, b = unc.correlated_values(popt, pcov)
    px = np.linspace(0, xmax, num=50, endpoint=True) 
    py = a*px+b
    nom = unp.nominal_values(py)*100
    std = unp.std_devs(py)*100
    ax2.plot(px, nom, c='r') # plot the regression line and uncertainty band (95% confidence)
    ax2.fill_between(px, nom - 1.96 * std, nom + 1.96 * std, color='orange', alpha=0.2)

    ax_index_label(ax1, '(a)', label_size+2)
    ax_index_label(ax2, '(b)', label_size+2)
    
    f.savefig('wavelength_select_and_o2a_slope_inter_derive.png', dpi=300)

def ax_index_label(ax, label, label_size):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text(xmin+0.0*(xmax-xmin), ymin+1.025*(ymax-ymin), label, fontsize=label_size, color='k')


if __name__ == "__main__":
    now = time.time()
    
    main()

    print(f'{(time.time()-now)/60:.3f} min')