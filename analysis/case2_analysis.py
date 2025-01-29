import sys
sys.path.append('/Users/yuch8913/miniforge3/envs/er3t_env/lib/python3.8/site-packages')
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import font_manager, cm, colors
from matplotlib.colors import LogNorm
import numpy as np
import copy
from oco_post_class_ywc import OCOSIM
from scipy import interpolate
from scipy.ndimage import uniform_filter
from  scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import geopy.distance
import seaborn as sns
from tool_code import *
import os, pickle 
from haversine import Unit, haversine_vector
import uncertainties.unumpy as unp
import uncertainties as unc
from util.oco_cfg import grab_cfg, output_h5_info, nan_array, ax_lon_lat_label

font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()


def near_rad_calc(OCO_class):
    OCO_class.rad_clr_5, OCO_class.rad_c3d_5,\
    OCO_class.rad_clrs_5, OCO_class.rad_c3ds_5  = coarsening(OCO_class, size=5)
    
    OCO_class.rad_clr_9, OCO_class.rad_c3d_9,\
    OCO_class.rad_clrs_9, OCO_class.rad_c3ds_9 = coarsening(OCO_class, size=9)
    
    OCO_class.rad_clr_13, OCO_class.rad_c3d_13,\
    OCO_class.rad_clrs_13, OCO_class.rad_c3ds_13 = coarsening(OCO_class, size=13)
    
    OCO_class.sl_5  = (OCO_class.rad_c3d_5-OCO_class.rad_clr_5) / OCO_class.rad_clr_5 
    OCO_class.sls_5 = (OCO_class.rad_c3ds_5/OCO_class.rad_clr_5 + OCO_class.rad_clrs_5/OCO_class.rad_clr_5)

    OCO_class.sl_9  = (OCO_class.rad_c3d_9-OCO_class.rad_clr_9) / OCO_class.rad_clr_9
    OCO_class.sls_9 = (OCO_class.rad_c3ds_9/OCO_class.rad_clr_9 + OCO_class.rad_clrs_9/OCO_class.rad_clr_9)

    OCO_class.sl_13  = (OCO_class.rad_c3d_13-OCO_class.rad_clr_13) / OCO_class.rad_clr_13
    OCO_class.sls_13 = (OCO_class.rad_c3ds_13/OCO_class.rad_clr_13 + OCO_class.rad_clrs_13/OCO_class.rad_clr_13)

def coarsening(OCO_class, size=3):
    ipa0 = coarsening_subfunction(OCO_class.rad_clr, OCO_class.cld_location, size, option='no_cloud')
    c3d  = coarsening_subfunction(OCO_class.rad_c3d, OCO_class.cld_location, size, option='no_cloud')
    ipa0_std = coarsening_subfunction(OCO_class.rad_clrs, OCO_class.cld_location, size, option='no_cloud')
    c3d_std   = coarsening_subfunction(OCO_class.rad_c3ds, OCO_class.cld_location, size, option='no_cloud')
    
    return ipa0, c3d, ipa0_std, c3d_std


def coarsening_subfunction(rad_mca, cld_position, size, option='no_cloud'):
    """
    Apply filter to a 3D array to process cloud data.

    Parameters:
    rad_mca: 3D array, radiances of various wavelengths
    cld_position: 2D array
    size: int, size of the filter
    option: str, filtering option

    Returns:
    Filtered array based on the specified option.
    
    option: str, 'no cloud' or 'cloud edge' or 'all'
    ###### 'no cloud': filter out the average areas containing cloud
    ###### 'cloud edge': only present the average areas containing cloud
    ###### 'all': no filter
    ****** Note that the box edge is not included in the filter
    """
    lams = rad_mca.shape[-1]
    tmp = np.zeros_like(rad_mca)
    rad_mca_mask_cld = rad_mca.copy()
    rad_mca_mask_cld[cld_position] = -999999
    for i in range(lams):
        tmp[:,:,i] = uniform_filter(rad_mca_mask_cld[:,:,i], size=size, mode='constant', cval=-999999)
    tmp[tmp<0] = np.nan

    if option == 'no_cloud':
        return tmp

    elif option == 'cloud_edge':
        tmp2 = np.zeros_like(rad_mca)
        ad_mca_mask_cld = rad_mca.copy()
        for i in range(lams):
            tmp2[:,:,i] = uniform_filter(rad_mca_mask_cld[:,:,i], size=size, mode='constant', cval=-999999)
        tmp2[tmp2<0] = np.nan
        tmp2[~np.isnan(tmp)] = np.nan
        tmp2[cld_position] = np.nan
        return tmp2

    elif option == 'all':
        tmp3 = np.zeros_like(rad_mca)
        rad_mca_mask_cld = rad_mca.copy()
        for i in range(lams):
            tmp3[:,:,i] = uniform_filter(rad_mca_mask_cld[:,:,i], size=size, mode='constant', cval=-999999)
        tmp3[tmp3<0] = np.nan
        return tmp3

    elif option == 'all_exclude_cloud':
        tmp4 = np.zeros_like(rad_mca)
        rad_mca_mask_cld = rad_mca.copy()
        for i in range(lams):
            tmp4[:,:,i] = uniform_filter(rad_mca_mask_cld[:,:,i], size=size, mode='constant', cval=-999999)
        tmp4[tmp4<0] = np.nan
        tmp4[cld_position] = np.nan
        return tmp4

    else:
        raise OSError('Invalid option!')

def get_slope_np(toa, sl_np, sls_np, c3d_np, clr_np, fp, z, sza, points=11, mode='unperturb'):
    """
    Calculate slope using numpy functions.

    Parameters:
    toa: Top of atmosphere data
    sl_np: Slope data
    sls_np: Slope statistics data
    c3d_np: Cloud data
    clr_np: Clear data
    fp: Footprint index
    z: Index
    sza: Solar zenith angle
    points: Number of data points
    mode: Calculation mode

    Returns:
    Tuple of slope, slope standard deviation, intercept, and intercept standard deviation.
    """
    nwl = sls_np[z,fp,:].shape[0]
    flt = np.where(sls_np[z,fp,:] > 1e-6)
    use = len(flt[0])
    mu = np.mean(sza)/180*np.pi
    if use == nwl:
        w = 1./sls_np[z,fp,:]    
        if mode == 'unperturb':
            x = c3d_np[z,fp,:]/(toa[:]*np.cos(mu))*np.pi
        else:
            x = clr_np[z,fp,:]/(toa[:]*np.cos(mu))*np.pi
        x_len = len(x)
        mask = np.argsort(x)[x_len-points:]
        res = np.polyfit(x[mask], sl_np[z,fp,:][mask], 1, w=w[mask], cov=True) # now get covariance as well!
        slope, intercept = res[0]
        slopestd = np.sqrt(res[1][0][0])
        interceptstd = np.sqrt(res[1][1][1])
    else:
        slope = np.nan
        slopestd = np.nan
        intercept = np.nan
        interceptstd = np.nan
    return(slope, slopestd, intercept, interceptstd)

def slopes_propagation(OCO_class, mode='unperturb'): 
    # goes through entire line for a given footprint fp
    array_size = [OCO_class.rad_clr_5.shape[0], OCO_class.rad_clr_5.shape[1], 2]
    OCO_class.slope_5avg, OCO_class.inter_5avg = (nan_array(array_size, dtype=np.float64) for _ in range(2))
    OCO_class.slope_9avg, OCO_class.inter_9avg = (nan_array(array_size, dtype=np.float64) for _ in range(2)) 
    OCO_class.slope_13avg, OCO_class.inter_13avg = (nan_array(array_size, dtype=np.float64) for _ in range(2)) 
    sza = np.mean(OCO_class.sza_avg)
    for z in range(OCO_class.rad_clr_5.shape[0]):
        for fp in range(OCO_class.rad_clr_5.shape[1]):   
            slope, slopestd, inter, interstd = get_slope_np(OCO_class.toa, OCO_class.sl_5, OCO_class.sls_5, OCO_class.rad_c3d_5, OCO_class.rad_clr_5, fp, z, sza, points=11, mode='unperturb')
            OCO_class.slope_5avg[z,fp,:]=[slope,slopestd]
            OCO_class.inter_5avg[z,fp,:]=[inter,interstd]

            slope,slopestd,inter,interstd=get_slope_np(OCO_class.toa, OCO_class.sl_9, OCO_class.sls_9, OCO_class.rad_c3d_9, OCO_class.rad_clr_9, fp, z, sza, points=11, mode='unperturb')
            OCO_class.slope_9avg[z,fp,:]=[slope,slopestd]
            OCO_class.inter_9avg[z,fp,:]=[inter,interstd]

            slope,slopestd,inter,interstd=get_slope_np(OCO_class.toa, OCO_class.sl_13, OCO_class.sls_13, OCO_class.rad_c3d_13, OCO_class.rad_clr_13, fp, z, sza, points=11, mode='unperturb')
            OCO_class.slope_13avg[z,fp,:]=[slope,slopestd]
            OCO_class.inter_13avg[z,fp,:]=[inter,interstd]

def cld_rad_slope_calc(band_tag, id_num, filename, pkl_filename, cld_location):
    h5_file  = filename.format(band_tag, id_num)
    OCO_class = OCOSIM(h5_file)
    OCO_class.cld_location = cld_location
    near_rad_calc(OCO_class)
    slopes_propagation(OCO_class)
    with open(pkl_filename.format(band_tag), 'wb') as pkl_file:
        pickle.dump(OCO_class, pkl_file)
    return OCO_class

def setup_plot(wesn, lon_dom, lat_dom, label_size, tick_size=12, figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(wesn[0], wesn[1])
    ax.set_ylim(wesn[2], wesn[3])
    ax.vlines(lon_dom, ymin=lat_dom[0], ymax=lat_dom[1], color='k', linewidth=1)
    ax.hlines(lat_dom, xmin=lon_dom[0], xmax=lon_dom[1], color='k', linewidth=1)
    ax_lon_lat_label(ax, label_size=label_size, tick_size=tick_size)
    return fig, ax

def save_figure(fig, file_name, img_dir, pad=None, dpi=300):
    if pad is None:
        fig.tight_layout()
    else:
        fig.tight_layout(pad=pad)
    fig.savefig(f'{img_dir}/{file_name}.png', format='png', dpi=dpi)


def sfc_alt_plt(img, wesn, lon_dom, lat_dom, 
                lon_2d, lat_2d, sfh_2d, label_size=14, img_dir='.'):
    fig, ax = setup_plot(wesn, lon_dom, lat_dom, label_size)
    ax.imshow(img, extent=wesn)
    c = ax.contourf(lon_2d,lat_2d, sfh_2d*1000,
                    cmap='terrain', levels=201, vmin=0, vmax=2000)
    cbar = fig.colorbar(c, ax=ax, extend='both')
    cbar.set_label('Surface altitude (m)', fontsize=label_size)
    save_figure(fig, 'surface_altitude', img_dir)


def cld_dist_plot(o1, img, wesn, lon_dom, lat_dom, 
                  lon_2d, lat_2d, cth0, cld_dist, label_size=14, img_dir='.'):
    fig, ax = setup_plot(wesn, lon_dom, lat_dom, label_size)
    ax.imshow(img, extent=wesn)
    c = ax.scatter(o1.lon2d, o1.lat2d, 
                   c=cld_dist, s=5,
                   cmap='Reds', vmin=0, vmax=20)
    ax.scatter(lon_2d[cth0>0], lat_2d[cth0>0], s=15, color='b')
    cbar = fig.colorbar(c, ax=ax, extend='both')
    cbar.set_label('Cloud distance (km)', fontsize=label_size)
    ax_lon_lat_label(ax, label_size=14, tick_size=12)
    save_figure(fig, 'cloud_distance', img_dir)

def weighted_cld_dist_plot(o1, img, wesn, lon_dom, lat_dom, 
                  lon_2d, lat_2d, cth0, weighted_cld_dist, label_size=14, img_dir='.'):
    fig, ax = setup_plot(wesn, lon_dom, lat_dom, label_size)
    ax.imshow(img, extent=wesn)
    c = ax.scatter(o1.lon2d, o1.lat2d, 
                   c=weighted_cld_dist, s=5,
                   cmap='Reds', vmin=0, vmax=20)
    ax.scatter(lon_2d[cth0>0], lat_2d[cth0>0], s=15, color='b')
    cbar = fig.colorbar(c, ax=ax, extend='both')
    cbar.set_label('$\mathrm{D_e}$ (km)', fontsize=label_size)
    ax_lon_lat_label(ax, label_size=14, tick_size=12)
    save_figure(fig, 'weighted_cloud_distance', img_dir)

def o2a_conti_plot(o1, rad_c3d_compare,
                   img, wesn, lon_dom, lat_dom, label_size=14, img_dir='.'):
    fig, ax = setup_plot(wesn, lon_dom, lat_dom, label_size)
    ax.imshow(img, extent=wesn)
    mask = np.isnan(getattr(o1, rad_c3d_compare)[:,:,-1])
    c = ax.scatter(o1.lon2d, o1.lat2d, 
                   c=getattr(o1, rad_c3d_compare)[:,:,-1], s=5, cmap='Reds')
    ax.scatter(o1.lon2d[mask], o1.lat2d[mask], c='grey', s=5, cmap='Reds')
    cbar = fig.colorbar(c, ax=ax, extend='both')
    cbar.set_label('$\mathrm{O_2-A}$ continuum (mW m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=label_size)
    ax_lon_lat_label(ax, label_size=14, tick_size=12)
    save_figure(fig, f'o2a_conti_{rad_c3d_compare}', img_dir)

def slope_intercept_compare_plot(OCO_class, label_tag, file_tag, pxl_by_pxl_output_csv,
                                img, wesn, lon_dom, lat_dom, 
                                lon_2d, lat_2d, cth0, 
                                slope_compare, inter_compare, label_size=14, img_dir='.'):
    
    f, (ax1, ax2) =plt.subplots(1, 2, figsize=(12, 6.75))
    for ax in [ax1, ax2]:
        ax.imshow(img, extent=wesn)
        ax.set_xlim(np.min(lon_dom), np.max(lon_dom))
        ax.set_ylim(np.min(lat_dom), np.max(lat_dom))
        ax_lon_lat_label(ax, label_size=14, tick_size=12)
    
    mask = ~(cth0>0)
    c1 = ax1.scatter(OCO_class.lon2d[mask], OCO_class.lat2d[mask], 
                     c=getattr(OCO_class, slope_compare)[:,:,0][mask], s=10,
                     cmap='RdBu_r', vmin=-0.15, vmax=0.15)
    idx = 91
    print(pxl_by_pxl_output_csv['SND'][idx])
    print('lon ind:', np.argmin(np.abs(pxl_by_pxl_output_csv['LON'][idx]-OCO_class.lon2d[:, 0])))
    print('lat ind:', np.argmin(np.abs(pxl_by_pxl_output_csv['LAT'][idx]-OCO_class.lat2d[0, :])))
    ax1.scatter(pxl_by_pxl_output_csv['LON'][idx], pxl_by_pxl_output_csv['LAT'][idx], marker='^', s=30, color='k')
    cbar1 = f.colorbar(c1, ax=ax1, extend='both')
    cbar1.set_label('$\mathit{s _{%s}}$' %(label_tag), fontsize=label_size)

    c2 = ax2.scatter(OCO_class.lon2d[mask], OCO_class.lat2d[mask], 
                     c=getattr(OCO_class, inter_compare)[:,:,0][mask], s=10,
                     cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    cbar2 = f.colorbar(c2, ax=ax2, extend='both')
    cbar2.set_label('$\mathit{i _{%s}}$' %(label_tag), fontsize=label_size)
    
    lonlat_interval = 0.1 if (lon_2d.max()-lon_2d.min())<1 else 0.2
    for ax, label in zip([ax1, ax2], ['(a)', '(b)']):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.scatter(lon_2d[cth0>0], lat_2d[cth0>0], s=15, color='r')
        ax.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 181.0, lonlat_interval)))
        ax.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 91.0, lonlat_interval)))
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.025*(ymax-ymin), label, fontsize=label_size+4, color='k')

    save_figure(f, f'{file_tag}_{slope_compare}_test', img_dir, pad=0.5)


def continuum_fp_compare_plot(o1, o2, o3, 
                              img, wesn, lon_dom, lat_dom, 
                              lon_2d, lat_2d, cth0, 
                              tick_size=12, label_size=14, img_dir='.'):
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
                vmin=vmin, vmax=vmax, **scatter_arg)
    cbar1 = f.colorbar(cc1, ax=ax1)
    ax1.set_title(f'{o1.lam[10]:.4f}nm')

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
                vmin=vmin, vmax=vmax, **scatter_arg)
    cbar2 = f.colorbar(cc2, ax=ax2)
    ax2.set_title(f'{o2.lam[10]:.4f}nm')

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
                vmin=vmin, vmax=vmax, **scatter_arg)
    cbar3 = f.colorbar(cc3, ax=ax3)
    ax3.set_title(f'{o3.lam[10]:.4f}nm')

    for cbar in [cbar1, cbar2, cbar3]:
        cbar.set_label('Radiance', fontsize=16)
    for ax, label in zip([ax1, ax2, ax3], ['(a)', '(b)', '(c)']):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.025*(ymax-ymin), label, fontsize=label_size+4, color='k')

    save_figure(f, 'continuum_fp_compare', img_dir)


def cld_position_func(cfg_name):
    cldfile = f'../simulation/data/{cfg_name}_{cfg_name[:8]}/pre-data.h5'
    with h5py.File(cldfile, 'r') as f:
        lon_cld = f['lon'][...]
        lat_cld = f['lat'][...]
        cth = f[f'mod/cld/cth_ipa'][...]
        cld_list = cth>0
    return lon_cld, lat_cld, cld_list

def cld_dist_calc(cfg_name, o1, slope_compare):
    _, _, cld_list = cld_position_func(cfg_name)
    cld_position = np.argwhere(cld_list)
    cloud_dist = np.zeros_like(getattr(o1, slope_compare)[:,:,0])
    for j in range(cloud_dist.shape[1]):
        for i in range(cloud_dist.shape[0]):
            if cld_list[i, j] == 1:
                cloud_dist[i, j] = 0
            else:
                min_ind = np.argmin(np.sqrt(np.sum((cld_position-np.array([i, j]))**2, axis=1)))
                cld_x, cld_y = cld_position[min_ind]
                cloud_dist[i, j] = geopy.distance.distance((o1.lat2d[cld_x, cld_y], o1.lon2d[cld_x, cld_y]), 
                                                           (o1.lat2d[i, j], o1.lon2d[i, j])).km
    output = np.array([o1.lon2d, o1.lat2d, cloud_dist,])
    cld_slope_inter = pd.DataFrame(output.reshape(output.shape[0], -1).T, columns=['lon', 'lat', 'cld_dis', ])
    cld_slope_inter.to_pickle(f'{cfg_name}_cld_distance.pkl')

def weighted_cld_dist_calc(cfg_name, o1, slope_compare):
    lon_cld, lat_cld, cld_list = cld_position_func(cfg_name)
    cld_position = np.argwhere(cld_list)
    cld_latlon = np.array([[lat_cld[i, j], lon_cld[i, j]] for i, j in cld_position])
    weighted_cloud_dist = np.zeros_like(getattr(o1, slope_compare)[:,:,0])
    for j in range(weighted_cloud_dist.shape[1]):
        for i in range(weighted_cloud_dist.shape[0]):
            if cld_list[i, j] == 1:
                weighted_cloud_dist[i, j] = 0
            else:
                point = np.array([o1.lat2d[i, j], o1.lon2d[i, j]])
                distances = haversine_vector(point, cld_latlon, unit=Unit.KILOMETERS, comb=True)
                # Calculate the inverse distance weights
                weights = 1 / distances**2  
                # Calculate the weighted average distance
                weighted_cloud_dist[i, j] = np.sum(distances * weights) / np.sum(weights)
    output = np.array([o1.lon2d, o1.lat2d, weighted_cloud_dist,])
    print('output shape:', output.shape)
    cld_slope_inter = pd.DataFrame(output.reshape(output.shape[0], -1).T, columns=['lon', 'lat', 'cld_dis', ])
    cld_slope_inter.to_pickle(f'{cfg_name}_weighted_cld_distance.pkl')   

def valid_data(x, y):
    # Remove invalid data points
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    return x[mask], y[mask]

def plot_errorbar(x, y_med, y_std, ax, color='r'):
    ax.errorbar(x, y_med, yerr=y_std, marker='s', markersize=3, color=color,
                linewidth=2, linestyle='', ecolor='skyblue')

def exp_decay_func(x, a, b):
     return a * np.exp(-b * x)

def heatmap_xy_3(x, y, ax, tag=''):
    # Remove invalid values
    x, y = valid_data(x, y)
    # cloud distance
    interval = 1/2
    start = 1
    # Scatter plot for valid data points
    ax.scatter(x[x>=start], y[x>=start], s=1, color='k')
    sns.kdeplot(x=x, y=y, cmap='hot_r', n_levels=20, fill=True, ax=ax, alpha=0.65)
    # Calculate value median of each cloud level
    cld_levels = np.arange(start, 50, interval)
    value_med, value_std = np.zeros(len(cld_levels)-1), np.zeros(len(cld_levels)-1)
    for i in range(len(cld_levels)-1):
        select = np.logical_and(x>=cld_levels[i], x < cld_levels[i+1])
        if select.sum()>5:
            value_med[i] = np.percentile(y[select], 50)
            value_std[i] = np.percentile(y[select], 75)-np.percentile(y[select], 25)
        else:
            value_med[i] = np.nan
            value_std[i] = np.nan
    cld_list = (cld_levels[:-1] + cld_levels[1:])/2
    
    plot_errorbar(cld_list, value_med, value_std, ax, color='r')
    
    val_mask = ~(np.isnan(value_med) | np.isnan(value_std) | np.isinf(value_med) | np.isinf(value_std))
    temp_r2 = 0
    cld_val = cld_list[val_mask]
    cld_min_list = [1+0.5*i for i in range(3)] if cld_val.min()<=2 else [cld_val.min().round(0)+0.5*(i-1) for i in range(10)] 
    if tag == 's':
        cld_max_start = 10 if cld_val.min()<=2 else  25
        cld_min_list = [1+0.5*i for i in range(3)] if cld_val.min()<=2 else [cld_val.min().round(0)+0.5*(i-1) for i in range(10)] 
    else:
        cld_max_start = 10 if cld_val.min()<=2 else  15
        cld_min_list = [1+0.5*i for i in range(3)] if cld_val.min()<=2 else [cld_val.min().round(0)+0.25*(i-1) for i in range(3)] 
    for cld_min in cld_min_list:
        for cld_max in np.arange(cld_max_start, 50, 1):
            mask = np.logical_and (np.logical_and(cld_val>=cld_min, cld_val<=cld_max), value_std[val_mask]>0)
            xx = cld_val[mask]
            yy = value_med[val_mask][mask]
            if len(yy) > 0:
                popt, pcov = curve_fit(exp_decay_func, xx, yy, bounds=([1e-2, 2e-2], [5, 1,]),
                                       p0=(0.15, 0.07),
                                       maxfev=10000,
                                       sigma=value_std[val_mask][mask], 
                                       absolute_sigma=True,)
                residuals = yy - exp_decay_func(xx, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((yy-np.mean(yy))**2)
                r_squared = 1 - (ss_res / ss_tot)
                if r_squared > temp_r2:
                    temp_r2 = r_squared
                else:
                    break
    perr = np.sqrt(np.diag(pcov))
    e_fold_dist = 1/popt[1]
    e_fold_dist_err = perr[1]/(popt[1]**2)
    amp_str = r'$\mathrm{a_{%s}}$' %(tag)
    d_ef_tag = r'$\mathrm{d_{%s}}$' %(tag)
    plot_xx = np.arange(0, cld_list.max()+0.75, 0.5)
    ax.plot(plot_xx, exp_decay_func(plot_xx, *popt), '--', color='limegreen', 
            label='fit: {} = {:.3f} $\pm$ {:.3f}\n     {} =   {:.2f} $\pm$ {:.2f}'.format(amp_str, popt[0], perr[0], d_ef_tag, e_fold_dist, e_fold_dist_err), linewidth=3.5)
    ax.legend(fontsize=13)
    return popt, perr

def scatter_xy_3(x, y, ax, tag=''):
    # Remove invalid values
    x, y = valid_data(x, y)
    # cloud distance
    interval = 1/2
    start = 1
    # Calculate the point density
    xs, ys = np.array(x[x>=start]), np.array(y[x>=start])
    xy = np.vstack([xs, ys])
    Z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = Z.argsort()
    x_den, y_den, Z_den = xs[idx], ys[idx], Z[idx]
    cs = ax.scatter(x_den, y_den, 
                    c=Z_den, s=1, 
                    #Norm=LogNorm(vmin=1, vmax=300), 
                    cmap='hot',
                        )
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('Density', fontsize=14)
    # Calculate value median of each cloud level
    cld_levels = np.arange(start, 50, interval)
    value_med, value_std = np.zeros(len(cld_levels)-1), np.zeros(len(cld_levels)-1)
    for i in range(len(cld_levels)-1):
        select = np.logical_and(x>=cld_levels[i], x < cld_levels[i+1])
        if select.sum()>5:
            value_med[i] = np.percentile(y[select], 50)
            value_std[i] = np.percentile(y[select], 75)-np.percentile(y[select], 25)
        else:
            value_med[i] = np.nan
            value_std[i] = np.nan
    cld_list = (cld_levels[:-1] + cld_levels[1:])/2
    
    plot_errorbar(cld_list, value_med, value_std, ax, color='r')
    
    val_mask = ~(np.isnan(value_med) | np.isnan(value_std) | np.isinf(value_med) | np.isinf(value_std))
    temp_r2 = 0
    cld_val = cld_list[val_mask]
    cld_min_list = [1+0.5*i for i in range(3)] if cld_val.min()<=2 else [cld_val.min().round(0)+0.25*(i-1) for i in range(8)] 
    cld_max_start = 10 if cld_val.min()<=2 else  15 
    for cld_min in cld_min_list[:1]:
        for cld_max in np.arange(cld_max_start, 50, 1):
            mask = np.logical_and(np.logical_and(cld_val>=cld_min, cld_val<=cld_max), value_std[val_mask]>0)
            xx = cld_val[mask]
            yy = value_med[val_mask][mask]
            if len(yy) > 0:
                popt, pcov = curve_fit(exp_decay_func, xx, yy, bounds=([1e-5, 1e-5], [15, 15,]),
                                       p0=(0.1, 0.7),
                                       maxfev=5000,
                                       sigma=value_std[val_mask][mask], 
                                       absolute_sigma=True,)
                residuals = yy - exp_decay_func(xx, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((yy-np.mean(yy))**2)
                r_squared = 1 - (ss_res / ss_tot)
                if r_squared > temp_r2:
                    temp_r2 = r_squared
                else:
                    break
    perr = np.sqrt(np.diag(pcov))
    e_fold_dist = 1/popt[1]
    e_fold_dist_err = perr[1]/(popt[1]**2)
    amp_str = r'$\mathrm{a_{%s}}$' %(tag)
    d_ef_tag = r'$\mathrm{d_{%s}}$' %(tag)
    plot_xx = np.arange(0, cld_list.max()+0.75, 0.5)
    ax.plot(plot_xx, exp_decay_func(plot_xx, *popt), '--', color='limegreen', 
            label='fit: {} = {:.3f} $\pm$ {:.3f}\n     {} =   {:.2f} $\pm$ {:.2f}'.format(amp_str, popt[0], perr[0], d_ef_tag, e_fold_dist, e_fold_dist_err), linewidth=3.5)
    ax.legend(fontsize=13)
    return popt, perr

def fitting_3bands(cloud_dist, o1, o2, o3, rad_3d_compare, rad_clr_compare, 
                   slope_compare, inter_compare, region_mask,
                   img_dir='.'):

    return_list = []
    fig, ((ax11, ax12), 
          (ax21, ax22),
          (ax31, ax32)) = plt.subplots(3, 2, figsize=(12, 12), sharex=False)
    label_size = 16
    tick_size = 13

    ax_list = np.array([(ax11, ax12), 
                        (ax21, ax22),
                        (ax31, ax32)])
    for i in range(3):
        oco_band = [o1, o2, o3][i]
        rad_3d = getattr(oco_band, rad_3d_compare)[:,:, -1].flatten()
        rad_clr = getattr(oco_band, rad_clr_compare)[:,:, -1].flatten()
        mask = np.logical_and(np.logical_and(cloud_dist > 0, rad_3d>rad_clr), region_mask)
        slope = getattr(oco_band, slope_compare)[:,:,0].flatten()
        inter = getattr(oco_band, inter_compare)[:,:,0].flatten()
        ax1, ax2 = ax_list[i]
        (slope_a, slope_b), (slope_a_unc, slope_b_unc) = heatmap_xy_3(cloud_dist[mask], slope[mask], ax1, tag='s')
        (inter_a, inter_b), (inter_a_unc, inter_b_unc) = heatmap_xy_3(cloud_dist[mask], inter[mask], ax2, tag='i')
        return_list.append((slope_a, slope_b, inter_a, inter_b))

    ax11.set_ylim(-0.1, 0.3)
    ax12.set_ylim(-0.05, 0.2)
    ax21.set_ylim(-0.1, 0.2)
    ax22.set_ylim(-0.05, 0.2)
    ax31.set_ylim(-0.1, 0.2)
    ax32.set_ylim(-0.05, 0.2)
    cld_low, cld_max = 0, 45

    label_list = ['a', 'b', 'c', 'd', 'e', 'f']
    for i, ax in enumerate(ax_list.flatten()):
        ax.set_xlim(cld_low, cld_max)
        label_text = f'({label_list[i]})'
        ax.set_xlabel('Cloud distance (km)', fontsize=label_size)
        ax.tick_params(axis='both', labelsize=tick_size)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.hlines(0, 0, xmax, linestyle='--', color='white')
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.05*(ymax-ymin), label_text, fontsize=label_size, color='k')
    
    for ax_l, ax_r, band_tag in zip([ax11, ax21, ax31], [ax12, ax22, ax32], ['O_2-A', 'WCO_2', 'SCO_2']):
        ax_l.set_ylabel('$\mathit{s _{%s}}$' %(band_tag), fontsize=label_size)
        ax_r.set_ylabel('$\mathit{i _{%s}}$' %(band_tag), fontsize=label_size)
    save_figure(fig, f'all_band_{slope_compare.split("_")[-1]}', img_dir,)# pad=5.0)
    return return_list

def fitting_3bands_with_weighted_dis(cloud_dist, o1, o2, o3, 
                                     rad_3d_compare, rad_clr_compare, slope_compare, inter_compare, region_mask,
                                     img_dir='.'):
    return_list, return_list_err = [], []
    fig, ((ax11, ax12), 
          (ax21, ax22),
          (ax31, ax32)) = plt.subplots(3, 2, figsize=(12, 12), sharex=False)
    label_size = 16
    tick_size = 13

    ax_list = np.array([(ax11, ax12), 
                        (ax21, ax22),
                        (ax31, ax32)])
    for i in range(3):
        oco_band = [o1, o2, o3][i]
        rad_3d = getattr(oco_band, rad_3d_compare)[:,:, -1].flatten()
        rad_clr = getattr(oco_band, rad_clr_compare)[:,:, -1].flatten()
        mask = np.logical_and(np.logical_and(cloud_dist > 0, rad_3d>rad_clr), region_mask)
        
        slope = getattr(oco_band, slope_compare)[:,:,0].flatten()
        inter = getattr(oco_band, inter_compare)[:,:,0].flatten()

        ax1, ax2 = ax_list[i]
        (slope_a, slope_b), (slope_a_unc, slope_b_unc) = heatmap_xy_3(cloud_dist[mask], slope[mask], ax1, tag='s')
        (inter_a, inter_b), (inter_a_unc, inter_b_unc) = heatmap_xy_3(cloud_dist[mask], inter[mask], ax2, tag='i')
        return_list.append((slope_a, slope_b, inter_a, inter_b))
        return_list_err.append((slope_a_unc, slope_b_unc, inter_a_unc, inter_b_unc))

    ax11.set_ylim(-0.1, 0.2)
    ax12.set_ylim(-0.05, 0.25)
    ax21.set_ylim(-0.05, 0.15)
    ax22.set_ylim(-0.05, 0.25)
    ax31.set_ylim(-0.05, 0.15)
    ax32.set_ylim(-0.05, 0.2)

    cld_low, cld_max = 0, 45
    label_list = ['a', 'b', 'c', 'd', 'e', 'f']
    for i, ax in enumerate(ax_list.flatten()):
        ax.set_xlim(cld_low, cld_max)
        label_text = f'({label_list[i]})'
        ax.set_xlabel('$\mathrm{D_e}$ (km)', fontsize=label_size)
        ax.tick_params(axis='both', labelsize=tick_size)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.hlines(0, 0, xmax, linestyle='--', color='white')
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.05*(ymax-ymin), label_text, fontsize=label_size, color='k')
    
    for ax_l, ax_r, band_tag in zip([ax11, ax21, ax31], [ax12, ax22, ax32], ['O_2-A', 'WCO_2', 'SCO_2']):
        ax_l.set_ylabel('$\mathit{s _{%s}}$' %(band_tag), fontsize=label_size)
        ax_r.set_ylabel('$\mathit{i _{%s}}$' %(band_tag), fontsize=label_size)
    save_figure(fig, f'all_band_weighted_dis_{slope_compare.split("_")[-1]}', img_dir,)# pad=5.0)
    return return_list, return_list_err

def fitting_3bands_with_weighted_dis_scatter(cloud_dist, o1, o2, o3, 
                                     rad_3d_compare, rad_clr_compare, slope_compare, inter_compare, region_mask,
                                     img_dir='.'):
    return_list, return_list_err = [], []
    fig, ((ax11, ax12), 
          (ax21, ax22),
          (ax31, ax32)) = plt.subplots(3, 2, figsize=(12, 12), sharex=False)
    label_size = 16
    tick_size = 13

    ax_list = np.array([(ax11, ax12), 
                        (ax21, ax22),
                        (ax31, ax32)])
    for i in range(3):
        oco_band = [o1, o2, o3][i]
        rad_3d = getattr(oco_band, rad_3d_compare)[:,:, -1].flatten()
        rad_clr = getattr(oco_band, rad_clr_compare)[:,:, -1].flatten()
        mask = np.logical_and(np.logical_and(cloud_dist > 0, rad_3d>rad_clr), region_mask)
        
        slope = getattr(oco_band, slope_compare)[:,:,0].flatten()
        inter = getattr(oco_band, inter_compare)[:,:,0].flatten()

        ax1, ax2 = ax_list[i]
        (slope_a, slope_b), (slope_a_unc, slope_b_unc) = scatter_xy_3(cloud_dist[mask], slope[mask], ax1, tag='s')
        (inter_a, inter_b), (inter_a_unc, inter_b_unc) = scatter_xy_3(cloud_dist[mask], inter[mask], ax2, tag='i')
        return_list.append((slope_a, slope_b, inter_a, inter_b))
        return_list_err.append((slope_a_unc, slope_b_unc, inter_a_unc, inter_b_unc))

    ax11.set_ylim(-0.1, 0.2)
    ax12.set_ylim(-0.05, 0.25)
    ax21.set_ylim(-0.05, 0.15)
    ax22.set_ylim(-0.05, 0.25)
    ax31.set_ylim(-0.05, 0.15)
    ax32.set_ylim(-0.05, 0.2)

    cld_low, cld_max = 0, 45
    label_list = ['a', 'b', 'c', 'd', 'e', 'f']
    for i, ax in enumerate(ax_list.flatten()):
        ax.set_xlim(cld_low, cld_max)
        label_text = f'({label_list[i]})'
        ax.set_xlabel('$\mathrm{D_e}$ (km)', fontsize=label_size)
        ax.tick_params(axis='both', labelsize=tick_size)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.hlines(0, 0, xmax, linestyle='--', color='white')
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.05*(ymax-ymin), label_text, fontsize=label_size, color='k')
    
    for ax_l, ax_r, band_tag in zip([ax11, ax21, ax31], [ax12, ax22, ax32], ['O_2-A', 'WCO_2', 'SCO_2']):
        ax_l.set_ylabel('$\mathit{s _{%s}}$' %(band_tag), fontsize=label_size)
        ax_r.set_ylabel('$\mathit{i _{%s}}$' %(band_tag), fontsize=label_size)
    save_figure(fig, f'all_band_weighted_dis_scatter_{slope_compare.split("_")[-1]}', img_dir,)# pad=5.0)
    return return_list, return_list_err

def main(cfg_csv='20191005_central_asia.csv'):
    #'20181018_central_asia_2_test4.csv'
    #'20191005_central_asia'
    # '20181018_central_asia_2'
    
    
    cfg_dir = '../simulation/cfg'
    cfg_info = grab_cfg(f'{cfg_dir}/{cfg_csv}')
    if 'o2' in cfg_info.keys():
        id_num = output_h5_info(f'{cfg_dir}/{cfg_csv}', 'o2')[22:31]
        print(output_h5_info(f'{cfg_dir}/{cfg_csv}', 'o2'))
        print(id_num)
    else:
        raise IOError('No output files are recorded!')
    
    date   = datetime.datetime(int(cfg_info['date'][:4]),    # year
                               int(cfg_info['date'][4:6]),   # month
                               int(cfg_info['date'][6:]))    # day
                              
    cfg_name = cfg_info['cfg_name']
    case_name_tag = '%s_%s' % (cfg_info['cfg_name'], date.strftime('%Y%m%d'))
    extent = [float(loc) + offset for loc, offset in zip(cfg_info['subdomain'], [-0.15, 0.15, -0.15, 0.15])]
    extent_analysis = [float(loc) for loc in cfg_info['subdomain']]

    if not os.path.exists('output'):
        os.makedirs('output')
    img_dir = f'output/{case_name_tag}'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    compare_num = 5
    rad_c3d_compare = f'rad_c3d_{compare_num}'
    rad_clr_compare = f'rad_clr_{compare_num}'
    slope_compare = f'slope_{compare_num}avg'
    inter_compare = f'inter_{compare_num}avg'

    #filename = '../simulation/data_all_20181018_{}_{}_with_aod.h5' 
    filename = '../simulation/data/%s/data_all_%s_{}_{}.h5' % (cfg_name+'_'+cfg_info['date'], cfg_info['date'])

    img_dir = f'output/{case_name_tag}'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    print('filename:', filename)
    #pkl_filename = '20181018_central_asia_{}_with_aod.pkl'
    pkl_filename = '%s_{}_with_aod.pkl' % cfg_name
    if not os.path.isfile(pkl_filename.format('o2a')):
        _, _, cld_location = cld_position_func(cfg_name)
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

    if not os.path.isfile(f'{cfg_name}_cld_distance.pkl'):
        cld_dist_calc(cfg_name, o2, slope_compare)
    cld_data = pd.read_pickle(f'{cfg_name}_cld_distance.pkl')
    cld_dist = cld_data['cld_dis']

    # weighted_cld_dist_calc
    #--------------------------------------
    if not os.path.isfile(f'{cfg_name}_weighted_cld_distance.pkl'):
        weighted_cld_dist_calc(cfg_name, o2, slope_compare)
    weighted_cld_data = pd.read_pickle(f'{cfg_name}_weighted_cld_distance.pkl')
    weighted_cld_dist = weighted_cld_data['cld_dis']
    #--------------------------------------

    xco2 = o1.co2
    psur = o1.psur
    snd = o1.snd
    xco2_valid = xco2>0


    mask_fp = np.logical_and(np.logical_and(o1.lon[xco2_valid] >= extent_analysis[0], o1.lon[xco2_valid] <= extent_analysis[1]),
                             np.logical_and(o1.lat[xco2_valid] >= extent_analysis[2], o1.lat[xco2_valid] <= extent_analysis[3]))
    f_cld_distance = interpolate.RegularGridInterpolator((np.array(weighted_cld_data['lon']).reshape(o1.lon2d.shape)[:, 0], 
                                                          np.array(weighted_cld_data['lat']).reshape(o1.lon2d.shape)[0, :]),
                                                          np.array(weighted_cld_data['cld_dis']).reshape(o1.lon2d.shape), method='nearest')
    
    points_footprint = np.column_stack((o1.lon[xco2_valid][mask_fp].flatten(), o1.lat[xco2_valid][mask_fp].flatten()))
    oco_footprint_cld_distance = f_cld_distance(points_footprint)
    # oco_footprint_cld_distance = np.array([i for i in np.arange(0, 56, 1)]*3+[0, 0])[::-1]
    # print('check assign footprint cloud distance length is correct:', len(points_footprint)==len(oco_footprint_cld_distance))
    #""" 

    mask = np.logical_and(np.logical_and(o1.lon2d >= extent_analysis[0], o1.lon2d <= extent_analysis[1]),
                            np.logical_and(o1.lat2d >= extent_analysis[2], o1.lat2d <= extent_analysis[3]))
    mask = mask.flatten()
    # parameters_cld_distance_list = fitting_3bands(cld_dist, o1, o2, o3, rad_c3d_compare, rad_clr_compare, slope_compare, inter_compare, mask, img_dir=img_dir)
    parameters_cld_distance_list, parameters_cld_distance_list_unc = fitting_3bands_with_weighted_dis(weighted_cld_dist, o1, o2, o3, rad_c3d_compare, rad_clr_compare, slope_compare, inter_compare, mask, img_dir=img_dir)
    _, _ = fitting_3bands_with_weighted_dis_scatter(weighted_cld_dist, o1, o2, o3, rad_c3d_compare, rad_clr_compare, slope_compare, inter_compare, mask, img_dir=img_dir)

    
    # slope_a, slope_b, inter_a, inter_b
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
                                'i1': o2a_inter,
                                's1': o2a_slope,
                                'i2': wco2_inter,
                                's2': wco2_slope,
                                'i3': sco2_inter,
                                's3': sco2_slope,
                                'weighted_cld_distance': oco_footprint_cld_distance.flatten(),
                                },)
    output_csv['SND'] = output_csv['SND'].apply(lambda x: f'SND{x:.0f}')
    # output_csv.to_csv(f'{cfg_name}_footprint_cld_distance.csv', index=False)

    output_csv.to_csv(f'{cfg_name}_footprint_idealized.csv', index=False)
    # sys.exit()

    inter_lon_lat = (np.array(weighted_cld_data['lon']).reshape(o1.lon2d.shape)[:, 0], 
                     np.array(weighted_cld_data['lat']).reshape(o1.lon2d.shape)[0, :])
    o2a_slope_5avg_func = interpolate.RegularGridInterpolator(inter_lon_lat,
                                                              getattr(o1, 'slope_5avg')[:,:,0], method='nearest')
    o2a_inter_5avg_func = interpolate.RegularGridInterpolator(inter_lon_lat,
                                                              getattr(o1, 'inter_5avg')[:,:,0], method='nearest')
    wco2_slope_5avg_func = interpolate.RegularGridInterpolator(inter_lon_lat,
                                                               getattr(o2, 'slope_5avg')[:,:,0], method='nearest')
    wco2_inter_5avg_func = interpolate.RegularGridInterpolator(inter_lon_lat,
                                                               getattr(o2, 'inter_5avg')[:,:,0], method='nearest')
    sco2_slope_5avg_func = interpolate.RegularGridInterpolator(inter_lon_lat,
                                                               getattr(o3, 'slope_5avg')[:,:,0], method='nearest')
    sco2_inter_5avg_func = interpolate.RegularGridInterpolator(inter_lon_lat,
                                                               getattr(o3, 'inter_5avg')[:,:,0], method='nearest')
    oco_footprint_o2a_slope = o2a_slope_5avg_func(points_footprint)
    oco_footprint_o2a_inter = o2a_inter_5avg_func(points_footprint)
    wco2_footprint_o2a_slope = wco2_slope_5avg_func(points_footprint)
    wco2_footprint_o2a_inter = wco2_inter_5avg_func(points_footprint)
    sco2_footprint_o2a_slope = sco2_slope_5avg_func(points_footprint)
    sco2_footprint_o2a_inter = sco2_inter_5avg_func(points_footprint)

    pxl_by_pxl_output_csv = pd.DataFrame({'SND': snd[xco2_valid][mask_fp].flatten(),
                                        'LON': o1.lon[xco2_valid][mask_fp].flatten(),
                                        'LAT': o1.lat[xco2_valid][mask_fp].flatten(),
                                        'L2XCO2[ppm]': xco2[xco2_valid][mask_fp].flatten()*1e6,
                                        'L2PSUR[kPa]': psur[xco2_valid][mask_fp].flatten()/1000,
                                        'i1': oco_footprint_o2a_inter,
                                        's1': oco_footprint_o2a_slope,
                                        'i2': wco2_footprint_o2a_inter,
                                        's2': wco2_footprint_o2a_slope,
                                        'i3': sco2_footprint_o2a_inter,
                                        's3': sco2_footprint_o2a_slope,
                                        'weighted_cld_distance': oco_footprint_cld_distance.flatten(),
                                        },)
    pxl_by_pxl_output_csv['SND'] = pxl_by_pxl_output_csv['SND'].apply(lambda x: f'SND{x:.0f}')
    pxl_by_pxl_output_csv.to_csv(f'{cfg_name}_footprint_pixel_by_pixel.csv', index=False)
    
    cld_lon, cld_lat, cld_location = cld_position_func(cfg_name)
    
    with h5py.File(f'../simulation/data/{case_name_tag}/pre-data.h5', 'r') as f:
        lon_2d = f['lon'][...]
        lat_2d = f['lat'][...]
        sfh_2d = f['mod/geo/sfh'][...]
        cth0 = f['mod/cld/logic_cld'][...]
    
    mask = np.logical_and(np.logical_and(lon_2d >= extent[0], lon_2d <= extent[1]),
                          np.logical_and(lat_2d >= extent[2], lat_2d <= extent[3]))

    img_file = f'../simulation/data/{case_name_tag}/{cfg_info["png"]}'
    wesn = extent
    img = mpimg.imread(img_file)
    lon_dom = extent_analysis[:2]
    lat_dom = extent_analysis[2:]

    
    sfc_alt_plt(img, wesn, lon_dom, lat_dom, 
                lon_2d, lat_2d, sfh_2d, img_dir=img_dir)
    
    cld_dist_plot(o1, img, wesn, lon_dom, lat_dom, 
                  lon_2d, lat_2d, cth0, cld_dist, img_dir=img_dir)
    
    weighted_cld_dist_plot(o1, img, wesn, lon_dom, lat_dom, 
                  lon_2d, lat_2d, cth0, weighted_cld_dist, img_dir=img_dir)
    
    slope_intercept_compare_plot(o1, 'O_2-A', 'o2a', pxl_by_pxl_output_csv,
                                img, wesn, lon_dom, lat_dom, 
                                lon_2d, lat_2d, cth0, 
                                slope_compare, inter_compare, img_dir=img_dir)
    
    slope_intercept_compare_plot(o2, 'WCO_2', 'wco2', pxl_by_pxl_output_csv,
                                img, wesn, lon_dom, lat_dom, 
                                lon_2d, lat_2d, cth0, 
                                slope_compare, inter_compare, img_dir=img_dir)
    
    slope_intercept_compare_plot(o3, 'SCO_2', 'sco2', pxl_by_pxl_output_csv,
                                img, wesn, lon_dom, lat_dom, 
                                lon_2d, lat_2d, cth0, 
                                slope_compare, inter_compare, img_dir=img_dir)
    
    o2a_conti_plot(o1, rad_c3d_compare,
                   img, wesn, lon_dom, lat_dom, label_size=14, img_dir=img_dir)
    
    continuum_fp_compare_plot(o1, o2, o3,
                              img, wesn, lon_dom, lat_dom, 
                              lon_2d, lat_2d, cth0, img_dir=img_dir)
    
    o2a_wvl_select_slope_derivation(cfg_info, o1, img_dir=img_dir)
    


def o2a_wvl_select_slope_derivation(cfg_info, o1, img_dir='.'):
    date   = datetime.datetime(int(cfg_info['date'][:4]),    # year
                               int(cfg_info['date'][4:6]),   # month
                               int(cfg_info['date'][6:]))    # day
    case_name_tag = '%s_%s' % (cfg_info['cfg_name'], date.strftime('%Y%m%d'))
    nx = int(cfg_info['nx'])
    with h5py.File(f'../simulation/data/{case_name_tag}/atm_abs_o2a_{nx+1}.h5', 'r') as file:
        trnsx = file['trns_oco'][...]
        oco_tx = file['tx'][...]

    f, (ax1, ax2) =plt.subplots(1, 2, figsize=(14, 5))
    f.tight_layout(pad=5.0)

    label_size = 16
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
    mu = np.mean(o1.sza_avg)/180*np.pi
    sl_np = o1.sl_5
    sls_np = o1.sls_5
    c3d_np = o1.rad_c3d_5
    fp, z = 170, 114
    points = nx+1
    print(f'o2a slope intercept derivation sample: lon {o1.lon2d[z, fp]}, lat {o1.lat2d[z, fp]}')
    w = 1./sls_np[z,fp,:] 
    x = c3d_np[z,fp,:]/(toa[:]*np.cos(mu))*np.pi
    x_len = len(x)
    mask = np.argsort(x)[x_len-points:]
    res = np.polyfit(x[mask], sl_np[z,fp,:][mask], 1, w=w[mask], cov=True) # now get covariance as well!
    slope, intercept = res[0]
    slopes = np.sqrt(res[1][0][0])
    intercepts = np.sqrt(res[1][1][1])

    ax2.errorbar(x[mask], sl_np[z,fp,:][mask]*100, yerr=sls_np[z,fp,:]*100, color='k',
                ecolor='k', elinewidth=1, capsize=5, linewidth=0, marker='o', ms=5)
    yy=intercept+slope*x
    y1=intercept+intercepts+(slope+slopes)*x
    y2=intercept-intercepts+(slope-slopes)*x

    ax2.tick_params(axis='both', labelsize=tick_size)
    ymin, ymax = ax2.get_ylim()
    xmin, xmax = ax2.get_xlim()
    ax2.set_xlim(0, xmax*1.1)

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
    print('slope: ', a)
    print('intercept: ', b)
    ax_index_label(ax1, '(a)', label_size+2)
    ax_index_label(ax2, '(b)', label_size+2)
    save_figure(f, f'wavelength_select_and_o2a_slope_inter_derive', img_dir)

def ax_index_label(ax, label, label_size):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text(xmin+0.0*(xmax-xmin), ymin+1.035*(ymax-ymin), label, fontsize=label_size, color='k')

if __name__ == "__main__":
    now = time.time()
    main()
    print(f'{(time.time()-now)/60:.3f} min')