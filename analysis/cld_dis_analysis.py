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

def read_cfg_cld_dis(cfg_csv='20181018_central_asia_2_test4.csv'):
    cfg_dir = '../simulation/cfg'
    cfg_info = grab_cfg(f'{cfg_dir}/{cfg_csv}')
    cfg_name = cfg_info['cfg_name']
    extent_png = [float(loc) + offset for loc, offset in zip(cfg_info['subdomain'], [-0.15, 0.15, -0.15, 0.15])]
    extent_analysis = [float(loc) for loc in cfg_info['subdomain']]
    weighted_cld_data = pd.read_pickle(f'../simulation/data/{cfg_name}_modis/weighted_cld_distance.pkl')
    weighted_cld_dist = weighted_cld_data['cld_dis']
    weighted_cld_lon = weighted_cld_data['lon']
    weighted_cld_lat = weighted_cld_data['lat']
    extent = [float(loc) for loc in cfg_info['subdomain']]
    mask = np.logical_and(np.logical_and(weighted_cld_lon >= extent[0], weighted_cld_lon <= extent[1]),
                          np.logical_and(weighted_cld_lat >= extent[2], weighted_cld_lat <= extent[3]))
    
    # hist = np.histogram(weighted_cld_dist, bins=np.linspace(0, 50, 51), density=True)
    cldfile = f'../simulation/data/{cfg_name}_modis/pre-data.h5'
    print(cldfile)
    with h5py.File(cldfile, 'r') as f:
        lon_cld = f['lon'][...]
        lat_cld = f['lat'][...]
        cld_position = f['mod/cld/cot_3d_650'][...]>0
    cld_mask = np.logical_and(np.logical_and(lon_cld >= extent[0], lon_cld <= extent[1]),
                              np.logical_and(lat_cld >= extent[2], lat_cld <= extent[3]))


    return weighted_cld_dist[mask], cld_position[cld_mask]

def read_cfg_img(cfg_csv='20181018_central_asia_2_test4.csv'):
    cfg_dir = '../simulation/cfg'
    cfg_info = grab_cfg(f'{cfg_dir}/{cfg_csv}')
    cfg_name = cfg_info['cfg_name']
    extent_png = [float(loc) + offset for loc, offset in zip(cfg_info['subdomain'], [-0.15, 0.15, -0.15, 0.15])]
    extent_analysis = [float(loc) for loc in cfg_info['subdomain']]
    png = cfg_info['png']
    return f'../simulation/data/{cfg_name}_modis/{png}'



def main(cfg_csv='20181018_central_asia_2_test4.csv'):
    # '20181018_central_asia_2_test4.csv'
    # '20150622_amazon.csv'
    # '20181018_central_asia_2_test6.csv'

    cfg_dir = '../simulation/cfg'

    cfg_info = grab_cfg(f'{cfg_dir}/{cfg_csv}')

    
    date   = datetime.datetime(int(cfg_info['date'][:4]),    # year
                               int(cfg_info['date'][4:6]),   # month
                               int(cfg_info['date'][6:]))    # day
                              
    cfg_name = cfg_info['cfg_name']
    case_name_tag = '%s_%s' % (cfg_info['cfg_name'], date.strftime('%Y%m%d'))
    extent_png = [float(loc) + offset for loc, offset in zip(cfg_info['subdomain'], [-0.15, 0.15, -0.15, 0.15])]
    extent_analysis = [float(loc) for loc in cfg_info['subdomain']]


    
    with h5py.File(f'../simulation/data/{case_name_tag}/pre-data.h5', 'r') as f:
        lon_2d = f['lon'][...]
        lat_2d = f['lat'][...]
        sfh_2d = f['mod/geo/sfh'][...]
        cth0 = f['mod/cld/cth_ipa'][...]
    
    extent = [float(loc) for loc in cfg_info['subdomain']]
    mask = np.logical_and(np.logical_and(lon_2d >= extent[0], lon_2d <= extent[1]),
                          np.logical_and(lat_2d >= extent[2], lat_2d <= extent[3]))

    
    lon_dom = extent_analysis[:2]
    lat_dom = extent_analysis[2:]


    # retrieval_aod_parameterization = h5py.File('full-unperturbed20181018_central_asia_2_test4_para_ideal2_20240402.h5', 'r')
    # mask = retrieval_aod_parameterization['xco2_retrieved'][...]!=-2
    # key_list = ['aod', 'cpu_minutes', 'lat', 'lon', 'psur_MT_file', 'psur_retrieved',
    #             'rfl1', 'rfl2', 'rfl3', 'snd', 'xco2_L2_file', 'xco2_retrieved', 'xco2_weighted_column']
    # df = pd.DataFrame({key:retrieval_aod_parameterization[key][...] for key in key_list})
    # df['o2a_inter'] = retrieval_aod_parameterization['pert_o2'][...][:, 0]
    # df['o2a_slope'] = retrieval_aod_parameterization['pert_o2'][...][:, 1]
    # df['wco2_inter'] = retrieval_aod_parameterization['pert_wco2'][...][:, 0]
    # df['wco2_slope'] = retrieval_aod_parameterization['pert_wco2'][...][:, 1]
    # df['sco2_inter'] = retrieval_aod_parameterization['pert_sco2'][...][:, 0]
    # df['sco2_slope'] = retrieval_aod_parameterization['pert_sco2'][...][:, 1]
    # df.replace(-2, np.nan, inplace=True)
    # df.loc[df['xco2_L2_file']<1, 'xco2_L2_file'] = df.loc[df['xco2_L2_file']<1, 'xco2_L2_file']*1e6
    # #df.drop_duplicates('snd', inplace=True)
    # df['diff_xco2'] = df['xco2_retrieved']-df['xco2_L2_file']
    # cld_dist = np.array([i for i in np.arange(0, 56, 1)]*3+[0, 0])
    # df['weighted_cld_dist'] = cld_dist

    retrieval_aod_parameterization2 = h5py.File('full-unperturbed20181018_central_asia_2_test4_para_ideal2_20240402.h5', 'r')
    mask = retrieval_aod_parameterization2['xco2_retrieved'][...]!=-2
    key_list = ['aod', 'cpu_minutes', 'lat', 'lon', 'psur_MT_file', 'psur_retrieved',
                'rfl1', 'rfl2', 'rfl3', 'snd', 'xco2_L2_file', 'xco2_retrieved', 'xco2_weighted_column']
    df2 = pd.DataFrame({key:retrieval_aod_parameterization2[key][...] for key in key_list})
    df2['o2a_inter'] = retrieval_aod_parameterization2['pert_o2'][...][:, 0]
    df2['o2a_slope'] = retrieval_aod_parameterization2['pert_o2'][...][:, 1]
    df2['wco2_inter'] = retrieval_aod_parameterization2['pert_wco2'][...][:, 0]
    df2['wco2_slope'] = retrieval_aod_parameterization2['pert_wco2'][...][:, 1]
    df2['sco2_inter'] = retrieval_aod_parameterization2['pert_sco2'][...][:, 0]
    df2['sco2_slope'] = retrieval_aod_parameterization2['pert_sco2'][...][:, 1]
    df2.replace(-2, np.nan, inplace=True)
    df2.loc[df2['xco2_L2_file']<1, 'xco2_L2_file'] = df2.loc[df2['xco2_L2_file']<1, 'xco2_L2_file']*1e6
    #df.drop_duplicates('snd', inplace=True)
    df2['diff_xco2'] = df2['xco2_L2_file']-df2['xco2_retrieved']
    cld_dist = np.array([i for i in np.arange(0, 56, 1)]*3+[0, 0])[::-1]
    df2['weighted_cld_dist'] = cld_dist
    df2[df2['diff_xco2']>0.5] = np.nan
    print(df2[df2['diff_xco2']>0])


    # df_all = pd.concat([df, df2], ignore_index=True)
    df_all = df2
    df_all.loc[df_all['weighted_cld_dist']==0, 'diff_xco2'] = np.nan
    df_gp = df_all.groupby('weighted_cld_dist')
    print(df_gp[['weighted_cld_dist', 'diff_xco2']])
    print(df_gp.mean()['diff_xco2'])
    print(df_gp.std()['diff_xco2'])

    fig = plt.figure(figsize=(18, 8.5))
    
    dx = 0.15 
    x_interval = 0.245
    y_top = 0.5
    y_bottom = 0.245
    ax11 = fig.add_axes([0.05+x_interval*0, 0.445, dx, y_top])
    ax12 = fig.add_axes([0.05+x_interval*1, 0.445, dx, y_top])
    ax13 = fig.add_axes([0.05+x_interval*2, 0.445, dx, y_top])
    ax14 = fig.add_axes([0.05+x_interval*3, 0.445, dx, y_top])
    ax21 = fig.add_axes([0.05+x_interval*0, 0.075, dx, y_bottom])
    ax22 = fig.add_axes([0.05+x_interval*1, 0.075, dx, y_bottom])
    ax23 = fig.add_axes([0.05+x_interval*2, 0.075, dx, y_bottom])
    ax24 = fig.add_axes([0.05+x_interval*3, 0.075, dx, y_bottom])

    for ax, cfg in zip([ax11, ax12, ax13, ax14], 
                       [#'20150219_central_asia_modis.csv',
                        '20210414_central_asia_modis.csv',
                        '20141130_central_asia_modis.csv',
                        '20160903_central_asia_modis.csv',
                        '20170802_central_asia_modis.csv',
                        ]):
        img_file = read_cfg_img(cfg)
        img = mpimg.imread(img_file)
        ax.imshow(img, extent=extent_png)
        ax.set_xlim(np.min(lon_dom), np.max(lon_dom))
        ax.set_ylim(np.min(lat_dom), np.max(lat_dom))
        ax.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 181.0, 0.1)))
        ax.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 91.0, 0.1)))

        ax_lon_lat_label(ax, label_size=16, tick_size=14)
    
    for ax, cfg in zip([ax21, ax22, ax23, ax24], 
                       [#'20150219_central_asia_modis.csv',
                        '20210414_central_asia_modis.csv',
                        '20141130_central_asia_modis.csv',
                        '20160903_central_asia_modis.csv',
                        '20170802_central_asia_modis.csv',
                        ]):
        weighted_cld_dist, cld_position = read_cfg_cld_dis(cfg)
        hist = ax.hist(weighted_cld_dist[weighted_cld_dist>0], 
                        bins=np.linspace(-0.5, 95.5, 97), density=True, 
                        color='skyblue', alpha=0.5,
                        edgecolor='k', linewidth=1.0)
        ax.set_xlabel('$\mathrm{D_e}$ (km)', fontsize=16)
        ax.set_ylabel('Probability density', fontsize=16)
        ax_twin = ax.twinx()
        ax_twin.errorbar(sorted(set(cld_dist)), df_gp.mean()['diff_xco2'], 
                         yerr=df_gp.std()['diff_xco2'], color='k', 
                         marker='o', markersize=3, ls='none')
        ax_twin.set_ylabel('$\Delta\mathrm{X_{CO2}}$ (ppm)', fontsize=16)
        ax.set_xlim(-0.5, 45.5)
        # print('hist:', hist)
        # print('hist[0]:', hist[0])
        # print('hist[0] shape:', hist[0].shape)
        # print("df_gp.mean()['diff_xco2'] shape:", df_gp.mean()['diff_xco2'].shape)
        avg_xco2_diff = np.sum(df_gp.mean()['diff_xco2']*hist[0][:56])/np.sum(hist[0])
        avg_xco2_diff_std = np.sum((df_gp.std()['diff_xco2']*hist[0][:56])**2)/np.sum(hist[0]**2)
        xmin, xmax = ax_twin.get_xlim()
        ymin, ymax = ax_twin.get_ylim()
        ax_twin.text(xmin+0.195*(xmax-xmin), ymin+1.055*(ymax-ymin), 
                     '  Regional $\mathbf{\Delta X_{CO2}}$:\n %.3f $\mathbf{\pm}$ %.3f ppm' %(avg_xco2_diff*-1, np.sqrt(avg_xco2_diff_std)),
                     fontsize=14, color='k', weight='bold')
        # ax_twin.text(xmin+0.185*(xmax-xmin), ymin+1.5*(ymax-ymin), 
        #              'Mean weighted average cloud distance:\n%.2f km' %(np.sum(np.linspace(0, 95, 96)*hist[0])/np.sum(hist[0])),
        #              fontsize=12, color='k',)
        # ax_twin.text(xmin+0.185*(xmax-xmin), ymin+1.25*(ymax-ymin), 
        #              'Domain cloud fraction: %.3f' %(np.sum([weighted_cld_dist==0])/weighted_cld_dist.count()),
        #              fontsize=12, color='k',)
        print(f'avg_xco2_diff: {avg_xco2_diff:.3f} +/- {np.sqrt(avg_xco2_diff_std):.3f} ppm')
        print(f'avg cloud distance: {np.sum(np.linspace(0, 95, 96)*hist[0])/np.sum(hist[0]):.3f} km')
        print(f'avg cloud distance: {np.mean(weighted_cld_dist[weighted_cld_dist>0]):.3f} km')
        print(f'cloud fraction: {np.sum([weighted_cld_dist==0]) /weighted_cld_dist.count():.3f}')
        print(f'non-cloud fraction: {np.sum([weighted_cld_dist>0])/weighted_cld_dist.count():.3f}')
        print(f'cloud fraction: {np.sum(cld_position)/len(cld_position):.3f}')


    label_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ax_list = [ax11, ax12, ax13, ax14, ax21, ax22, ax23, ax24]
    for i in range(8):
        ax = ax_list[i]
        label_text = f'({label_list[i]})'
        ax.tick_params(axis='both', labelsize=14)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.05*(ymax-ymin), label_text, 
                fontsize=22, color='k')


    fig.tight_layout()
    fig.savefig(f'cld_dist.png', dpi=300)



def ax_index_label(ax, label, label_size):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text(xmin+0.0*(xmax-xmin), ymin+1.035*(ymax-ymin), label, fontsize=label_size, color='k')


if __name__ == "__main__":
    now = time.time()
    
    main()

    print(f'{(time.time()-now)/60:.3f} min')