import sys
sys.path.append('/Users/yuch8913/miniforge3/envs/er3t_env/lib/python3.8/site-packages')
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import numpy as np
import copy
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
import matplotlib.image as mpl_img
from haversine import Unit, haversine_vector
from matplotlib import cm, colors
import uncertainties.unumpy as unp
import uncertainties as unc

from util.oco_cfg import grab_cfg, output_h5_info, nan_array, ax_lon_lat_label

font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()



def main(result_csv='20181018_central_asia_zpt_test_fitting_result.txt'):
    # '20181018_central_asia_2_test4.csv'
    # '20150622_amazon.csv'
    # '20181018_central_asia_2_test6.csv'
    # 20190621_australia_2.csv


    df = pd.read_csv(result_csv)
        
    channel_list = ['o2a', 'wco2', 'sco2']
        
    print(df.columns)
    # print(df.columns.shape)
    # print(df.shape)
    # print(df.head())
    # df_select = df[(np.logical_and(np.logical_and(df['cth']==5, df['alb']==0.3), 
    #                                np.logical_and(df['cot']==5, df['cer']==25)),
    #                 df['aod']==0)]
    # var = 'sza'
    # sza
    # df_select = df[np.logical_and(np.logical_and(np.logical_and(df['cth']==5, df['alb']==0.3), np.logical_and(df['cot']==5, df['cer']==25)), df['aod']==0)]
    # var = 'sza'
    # aod
    # df_select = df[np.logical_and(np.logical_and(np.logical_and(df['cth']==5, df['alb']==0.3), np.logical_and(df['cot']==5, df['cer']==25)), df['sza']==45)]
    # var = 'aod'
    # alb
    df_select = df[np.logical_and(np.logical_and(np.logical_and(df['cth']==5, df['aod']==0.0), np.logical_and(df['cot']==5, df['cer']==25)), df['sza']==45)]
    var = 'alb'
    # cth_high cloud
    # df_select = df[np.logical_and(np.logical_and(np.logical_and(df['alb']==0.3, df['aod']==0.0), np.logical_and(df['cot']==5, df['cer']==25)), df['sza']==45)]
    # var = 'cth'
    # cth_low cloud
    # df_select = df[np.logical_and(np.logical_and(np.logical_and(df['alb']==0.3, df['aod']==0.0), np.logical_and(df['cot']==1, df['cer']==12)), df['sza']==45)]
    # var = 'cth'
    # cot_high cloud
    # df_select = df[np.logical_and(np.logical_and(np.logical_and(df['alb']==0.3, df['aod']==0.0), np.logical_and(df['cth']==5, df['cer']==25)), df['sza']==45)]
    # var = 'cot'
    # cot_low cloud
    # df_select = df[np.logical_and(np.logical_and(np.logical_and(df['alb']==0.3, df['aod']==0.0), np.logical_and(df['cth']==3, df['cer']==12)), df['sza']==45)]
    # var = 'cot'

    print(df_select.shape)
    df_plot(df_select, var)


def df_plot(df, var):
    fig, ((ax11, ax12, ax13),
          (ax21, ax22, ax23)) = plt.subplots(2, 3, figsize=(18, 8), sharex=False)
    ax_list = [(ax11, ax12, ax13),
               (ax21, ax22, ax23)]
    channel_list = ['o2a', 'wco2', 'sco2']
    cld_list = np.linspace(0, 50, 21)
    for i in range(3):
        for j in range(df.shape[0]):
            print(j, df.iloc[j, :][f'slope_{channel_list[i]}_amp'])
            slope_line = exp_decay_func(cld_list, df.iloc[j, :][f'slope_{channel_list[i]}_amp'], df.iloc[j, :][f'slope_{channel_list[i]}_dec'])
            ax_list[0][i].plot(cld_list, slope_line, label=df.iloc[j, :][var])
            inter_line = exp_decay_func(cld_list, df.iloc[j, :][f'inter_{channel_list[i]}_amp'], df.iloc[j, :][f'inter_{channel_list[i]}_dec'])
            ax_list[1][i].plot(cld_list, inter_line, label=df.iloc[j, :][var])
        
        ax_xy_label(ax_list[0][i], 'Weighted average cloud distance (km)', f'{channel_list[i]} slope', label_size=14, tick_size=12)
        ax_xy_label(ax_list[0][i], 'Weighted average cloud distance (km)', f'{channel_list[i]} slope', label_size=14, tick_size=12)
        for j in range(2):
            ax_list[j][i].legend()
            ax_list[j][i].tick_params(axis='both', which='major', labelsize=12)
            ax_list[j][i].set_xlim(0, 50)
            ax_list[j][i].set_ylim(0, 0.5)
            ax_list[j][i].set_xticks(np.arange(0, 51, 10))
            ax_list[j][i].set_yticks(np.arange(0, 0.51, 0.1))
            ax_list[j][i].hlines(0, xmin=0, xmax=50, color='k', linewidth=1, linestyle='--')
    # ax.hlines(lat_dom, xmin=lon_dom[0], xmax=lon_dom[1], color='k', linewidth=1)
    # mask = np.isnan(getattr(o1, rad_c3d_compare)[:,:,-1])
    # print(mask.sum())
    # c = ax.scatter(o1.lon2d, o1.lat2d, 
    #                c=getattr(o1, rad_c3d_compare)[:,:,-1], s=5, cmap='Reds')
    # ax.scatter(o1.lon2d[mask], o1.lat2d[mask], 
    #                c='grey', s=5, cmap='Reds')
    # cbar = f.colorbar(c, ax=ax, extend='both')
    # cbar.set_label('$\mathrm{O_2-A}$ continuum (mW m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)', fontsize=label_size)
    # ax_lon_lat_label(ax, label_size=14, tick_size=12)
    fig.tight_layout()
    plt.show()
    # fig.savefig(f'{img_dir}/o2a_conti_{rad_c3d_compare}.png', dpi=300)


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
    f.tight_layout()
    f.savefig(f'{img_dir}/continuum_fp_compare.png', dpi=300)


def cld_position(cfg_name):
    cldfile = f'../simulation/data/{cfg_name}_{cfg_name[:8]}/pre-data.h5'
    with h5py.File(cldfile, 'r') as f:
        lon_cld = f['lon'][...]
        lat_cld = f['lat'][...]
        cth = f[f'mod/cld/cth_ipa'][...]
        cld_list = cth>0
    return lon_cld, lat_cld, cld_list


def cld_dist_calc(cfg_name, o1, slope_compare):
    cldfile = f'../simulation/data/{cfg_name}_{cfg_name[:8]}/pre-data.h5'
    with h5py.File(cldfile, 'r') as f:
        cth = f[f'mod/cld/cth_ipa'][...]

    cld_list = cth>0
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
                cld_x, cld_y = cld_position[min_ind]
                if cld_x==cloud_dist.shape[0] or cld_y==cloud_dist.shape[1]:
                    print(cld_x, cld_y)
                cloud_dist[i, j] = geopy.distance.distance((o1.lat2d[cld_x, cld_y], o1.lon2d[cld_x, cld_y]), (o1.lat2d[i, j], o1.lon2d[i, j])).km
    
    output = np.array([o1.lon2d, o1.lat2d, cloud_dist, ])
    cld_slope_inter = pd.DataFrame(output.reshape(output.shape[0], output.shape[1]*output.shape[2]).T,
                                columns=['lon', 'lat', 'cld_dis', ])
    cld_slope_inter.to_pickle(f'{cfg_name}_cld_distance.pkl')

def weighted_cld_dist_calc(cfg_name, o1, slope_compare):
    cldfile = f'../simulation/data/{cfg_name}_{cfg_name[:8]}/pre-data.h5'
    with h5py.File(cldfile, 'r') as f:
        lon_cld = f['lon'][...]
        lat_cld = f['lat'][...]
        cth = f[f'mod/cld/cth_ipa'][...]

    cld_list = cth>0
    cld_X, cld_Y = np.where(cld_list==1)[0], np.where(cld_list==1)[1]
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
                distances = haversine_vector(point, cld_latlon, unit=Unit.KILOMETERS, comb=True)
                weights = 1 / distances**2  # Calculate the inverse distance weights
                # Calculate the weighted average distance
                cloud_dist[i, j] = np.sum(distances * weights) / np.sum(weights)
    
    output = np.array([o1.lon2d, o1.lat2d, cloud_dist, ])
    cld_slope_inter = pd.DataFrame(output.reshape(output.shape[0], output.shape[1]*output.shape[2]).T,
                                   columns=['lon', 'lat', 'cld_dis', ])
    cld_slope_inter.to_pickle(f'{cfg_name}_weighted_cld_distance.pkl')   

def weighted_cld_dist_vert_calc(cfg_name, o1, slope_compare):
    cldfile = f'../simulation/data/{cfg_name}_{cfg_name[:8]}/pre-data.h5'
    with h5py.File(cldfile, 'r') as f:
        lon_cld = f['lon'][...]
        lat_cld = f['lat'][...]
        cth = f[f'mod/cld/cth_ipa'][...]

    cld_list = cth>0
    cld_X, cld_Y = np.where(cld_list==1)[0], np.where(cld_list==1)[1]
    cld_position = []
    cld_latlon = []
    cld_top_height = []
    for i in range(len(cld_X)):
        cld_position.append(np.array([cld_X[i], cld_Y[i]]))
        cld_latlon.append([lat_cld[cld_X[i], cld_Y[i]], lon_cld[cld_X[i], cld_Y[i]]])
        cld_top_height.append(cth[cld_X[i], cld_Y[i]])
    cld_position = np.array(cld_position)
    cld_latlon = np.array(cld_latlon)

    cloud_dist = np.zeros_like(getattr(o1, slope_compare)[:,:,0])
    for j in range(cloud_dist.shape[1]):
        for i in range(cloud_dist.shape[0]):
            if cld_list[i, j] == 1:
                cloud_dist[i, j] = 0
            else:
                point = np.array([o1.lat2d[i, j], o1.lon2d[i, j]])
                horizontal_distances = haversine_vector(point, cld_latlon, unit=Unit.KILOMETERS, comb=True)
                distances = np.sqrt(horizontal_distances**2 + np.array(cld_top_height).reshape(-1, 1)**2)
                weights = 1 / distances**1  # Calculate the inverse distance weights
                # Calculate the weighted average distance
                cloud_dist[i, j] = np.sum(distances * weights) / np.sum(weights)
    
    output = np.array([o1.lon2d, o1.lat2d, cloud_dist, ])
    cld_slope_inter = pd.DataFrame(output.reshape(output.shape[0], output.shape[1]*output.shape[2]).T,
                                   columns=['lon', 'lat', 'cld_dis', ])
    cld_slope_inter.to_pickle(f'{cfg_name}_weighted_cld_distance_vertical.pkl')   


def heatmap_xy_3(x, y, ax):
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x, y = x[mask], y[mask]
    # cloud distance
    interval = 1/2
    start = 1
    
    ax.scatter(x[x>=start], y[x>=start], s=1, color='k')
    sns.kdeplot(x=x, y=y, cmap='hot_r', n_levels=20, fill=True, ax=ax, alpha=0.65)
    
    cld_levels = np.arange(start, 50, interval)
    value_avg, value_std = np.zeros(len(cld_levels)-1), np.zeros(len(cld_levels)-1)
    for i in range(len(cld_levels)-1):
        select = np.logical_and(x>=cld_levels[i], x < cld_levels[i+1])
        if select.sum()>5:
            value_avg[i] = np.percentile(y[select], 50)
            value_std[i] = np.percentile(y[select], 75)-np.percentile(y[select], 25)
        else:
            value_avg[i] = np.nan
            value_std[i] = np.nan
    cld_list = (cld_levels[:-1] + cld_levels[1:])/2
    
    ax.errorbar(cld_list, value_avg, yerr=value_std, 
                marker='s', markersize=3,
                color='r', linewidth=2, linestyle='', ecolor='skyblue')
    
    val_mask = ~(np.isnan(value_avg) | np.isnan(value_std) | np.isinf(value_avg) | np.isinf(value_std))
    temp_r2 = 0
    cld_val = cld_list[val_mask]
    cld_min_list = [1+0.25*i for i in range(10)] if cld_val.min()<=2 else [cld_val.min().round(0)+0.25*i for i in range(10)] 
    cld_max_start = 10 if cld_val.min()<=2 else  20
    for cld_min in cld_min_list:
        for cld_max in np.arange(cld_max_start, 50, 1):
            mask = np.logical_and(np.logical_and(cld_val>=cld_min, cld_val<=cld_max), value_std[val_mask]>0)
            xx = cld_val[mask]
            yy = value_avg[val_mask][mask]
            if len(yy) > 0:
                popt, pcov = curve_fit(exp_decay_func, xx, yy, bounds=([-5, 1e-3], [15, 15,]),
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
    plot_xx = np.arange(0, cld_list.max()+0.75, 0.5)
    ax.plot(plot_xx, exp_decay_func(plot_xx, *popt), '--', color='limegreen', 
            label='fit: amplitude     = {:.3f} $\pm$ {:.3f}\n     e-folding dis = {:.2f} $\pm$ {:.2f}'.format(popt[0], perr[0], e_fold_dist, e_fold_dist_err), linewidth=3.5)
    ax.legend(fontsize=13)
    return popt, perr

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

        (slope_a, slope_b), (slope_a_unc, slope_b_unc) = heatmap_xy_3(cloud_dist[mask], slope[mask], ax11)
        (inter_a, inter_b), (inter_a_unc, inter_b_unc) = heatmap_xy_3(cloud_dist[mask], inter[mask], ax12)

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


def fitting_3bands(cloud_dist, o1, o2, o3, rad_3d_compare, rad_clr_compare, 
                   slope_compare, inter_compare, region_mask,
                   img_dir='.'):

    return_list = []
    fig, ((ax11, ax12), 
          (ax21, ax22),
          (ax31, ax32)) = plt.subplots(3, 2, figsize=(12, 12), sharex=False)
    fig.tight_layout(pad=5.0)
    label_size = 16
    tick_size = 13

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
        (slope_a, slope_b), (slope_a_unc, slope_b_unc) = heatmap_xy_3(cloud_dist[mask], slope[mask], ax1)
        (inter_a, inter_b), (inter_a_unc, inter_b_unc) = heatmap_xy_3(cloud_dist[mask], inter[mask], ax2)
        return_list.append((slope_a, slope_b, inter_a, inter_b))

    cld_low, cld_max = 0, 45
    limit_1 = 0.2
    limit_2 = 0.15
    for ax_l, ax_r in zip([ax11, ax21, ax31], [ax12, ax22, ax32]):
        ax_l.set_xlim(cld_low, cld_max)
        ax_l.set_ylim(-limit_1, limit_1)
        ax_r.set_xlim(cld_low, cld_max)
        ax_r.set_ylim(-limit_2, limit_2)

    ax11.set_ylim(-0.1, 0.3)
    ax12.set_ylim(-0.05, 0.2)
    ax21.set_ylim(-0.1, 0.2)
    ax22.set_ylim(-0.05, 0.2)
    ax31.set_ylim(-0.1, 0.2)
    ax32.set_ylim(-0.05, 0.2)

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
    fig.savefig(f'{img_dir}/all_band_{slope_compare.split("_")[-1]}.png', dpi=150, bbox_inches='tight')

    return return_list

def fitting_3bands_with_weighted_dis(cloud_dist, o1, o2, o3, 
                                     rad_3d_compare, rad_clr_compare, slope_compare, inter_compare, region_mask,
                                     img_dir='.'):
    return_list, return_list_err = [], []
    fig, ((ax11, ax12), 
            (ax21, ax22),
            (ax31, ax32)) = plt.subplots(3, 2, figsize=(12, 12), sharex=False)
    fig.tight_layout(pad=5.0)
    label_size = 16
    tick_size = 13

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
        (slope_a, slope_b), (slope_a_unc, slope_b_unc) = heatmap_xy_3(cloud_dist[mask], slope[mask], ax1)
        (inter_a, inter_b), (inter_a_unc, inter_b_unc) = heatmap_xy_3(cloud_dist[mask], inter[mask], ax2)
        return_list.append((slope_a, slope_b, inter_a, inter_b))
        return_list_err.append((slope_a_unc, slope_b_unc, inter_a_unc, inter_b_unc))

    cld_low, cld_max = 0, 45
    limit_1 = 0.2
    limit_2 = 0.15
    for ax_l, ax_r in zip([ax11, ax21, ax31], [ax12, ax22, ax32]):
        ax_l.set_xlim(cld_low, cld_max)
        ax_l.set_ylim(-limit_1, limit_1)
        ax_r.set_xlim(cld_low, cld_max)
        ax_r.set_ylim(-limit_2, limit_2)

    ax11.set_ylim(-0.3, 0.5)
    ax12.set_ylim(-0.05, 0.35)
    ax21.set_ylim(-0.3, 0.35)
    ax22.set_ylim(-0.05, 0.35)
    ax31.set_ylim(-0.3, 0.35)
    ax32.set_ylim(-0.05, 0.3)

    label_list = ['a', 'b', 'c', 'd', 'e', 'f']
    ax_list = [ax11, ax12, ax21, ax31, ax22, ax32]
    for i in range(6):
        ax = ax_list[i]
        label_text = f'({label_list[i]})'
        ax.set_xlabel('Weighted Average Cloud Distance (km)', fontsize=label_size)
        ax.tick_params(axis='both', labelsize=tick_size)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.hlines(0, 0, xmax, linestyle='--', color='white')
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.05*(ymax-ymin), label_text, fontsize=label_size, color='k')
    
    for ax_l, ax_r, band_tag in zip([ax11, ax21, ax31], [ax12, ax22, ax32], ['O_2-A', 'WCO_2', 'SCO_2']):
        ax_l.set_ylabel('$\mathrm{%s}$ slope' %(band_tag), fontsize=label_size)
        ax_r.set_ylabel('$\mathrm{%s}$ intercept' %(band_tag), fontsize=label_size)
    fig.savefig(f'{img_dir}/all_band_weighted_dis_{slope_compare.split("_")[-1]}.png', dpi=150, bbox_inches='tight')

    return return_list, return_list_err

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
    sls_np = o1.sls_5#/np.sqrt(3)
    c3d_np = o1.rad_c3d_5
    fp, z = 170, 114
    points = nx+1

    w = 1./sls_np[z,fp,:] 
    x = c3d_np[z,fp,:]/(toa[:]*np.cos(mu))*np.pi
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
    print('slope: ', a)
    print('intercept: ', b)
    ax_index_label(ax1, '(a)', label_size+2)
    ax_index_label(ax2, '(b)', label_size+2)
    
    f.savefig(f'{img_dir}/wavelength_select_and_o2a_slope_inter_derive.png', dpi=300)

def ax_index_label(ax, label, label_size):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text(xmin+0.0*(xmax-xmin), ymin+1.035*(ymax-ymin), label, fontsize=label_size, color='k')

def ax_xy_label(ax, xlabel, ylabel, label_size=14, tick_size=12):
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.tick_params(axis='both', labelsize=tick_size)

if __name__ == "__main__":
    now = time.time()
    
    main()

    print(f'{(time.time()-now)/60:.3f} min')