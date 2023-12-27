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



def main(result_csv='20181018_central_asia_zpt_test2_fitting_result.txt'):
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
    # alb high cloud
    df_select = df[np.logical_and(np.logical_and(np.logical_and(df['cth']==5, df['aod']==0.0), np.logical_and(df['cot']==5, df['cer']==25)), df['sza']==45)]
    var = 'alb'
    # alb low cloud
    # df_select = df[np.logical_and(np.logical_and(np.logical_and(df['cth']==3, df['aod']==0.0), np.logical_and(df['cot']==1, df['cer']==12)), df['sza']==45)]
    # var = 'alb'
    # cth_high cloud
    # df_select = df[np.logical_and(np.logical_and(np.logical_and(df['alb']==0.3, df['aod']==0), np.logical_and(df['cot']==5, df['cer']==25)), df['sza']==45)]
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
    df_plot_o2a(df_select, var)


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

def df_plot_o2a(df, var):
    fig, (ax11, ax12) = plt.subplots(1, 2, figsize=(16, 6), sharex=False)
    fig.tight_layout(pad=5.0)
    label_size = 20
    tick_size = 16

    cld_list = np.linspace(0, 50, 101)
    channel_list = 'o2a'
    for j in range(df.shape[0]):
        print(j, df.iloc[j, :][f'slope_{channel_list}_amp'])
        line_arg = {'linewidth': 2, 'label': df.iloc[j, :][var], }
        slope_line = exp_decay_func(cld_list, df.iloc[j, :][f'slope_{channel_list}_amp'], df.iloc[j, :][f'slope_{channel_list}_dec'])
        slope_line_low = exp_decay_func(cld_list, df.iloc[j, :][f'slope_{channel_list}_amp']-df.iloc[j, :][f'slope_{channel_list}_amp_unc'], df.iloc[j, :][f'slope_{channel_list}_dec'])#+df.iloc[j, :][f'slope_{channel_list}_dec_unc'])
        slope_line_high = exp_decay_func(cld_list, df.iloc[j, :][f'slope_{channel_list}_amp']+df.iloc[j, :][f'slope_{channel_list}_amp_unc'], df.iloc[j, :][f'slope_{channel_list}_dec'])#-df.iloc[j, :][f'slope_{channel_list}_dec_unc'])
        ax11.fill_between(cld_list, slope_line_low, slope_line_high, alpha=0.15)
        ax11.plot(cld_list, slope_line, **line_arg)
        
        inter_line = exp_decay_func(cld_list, df.iloc[j, :][f'inter_{channel_list}_amp'], df.iloc[j, :][f'inter_{channel_list}_dec'])
        inter_line_low = exp_decay_func(cld_list, df.iloc[j, :][f'inter_{channel_list}_amp']-df.iloc[j, :][f'inter_{channel_list}_amp_unc'], df.iloc[j, :][f'inter_{channel_list}_dec']+df.iloc[j, :][f'inter_{channel_list}_dec_unc'])
        inter_line_high = exp_decay_func(cld_list, df.iloc[j, :][f'inter_{channel_list}_amp']+df.iloc[j, :][f'inter_{channel_list}_amp_unc'], df.iloc[j, :][f'inter_{channel_list}_dec']-df.iloc[j, :][f'inter_{channel_list}_dec_unc'])
        ax12.fill_between(cld_list, inter_line_low, inter_line_high, alpha=0.15)
        
        ax12.plot(cld_list, inter_line, **line_arg)
        
        ax_xy_label(ax11, 'Weighted average cloud distance (km)', '$\mathrm{O_2-A}$ slope', label_size=label_size, tick_size=tick_size)
        ax_xy_label(ax12, 'Weighted average cloud distance (km)', '$\mathrm{O_2-A}$ intercept', label_size=label_size, tick_size=tick_size)
        
        ax11.set_ylim(0, 0.75)
        ax12.set_ylim(0, 0.75)
        for ax in [ax11, ax12]:
            ax.legend(fontsize=label_size)
            ax.tick_params(axis='both', which='major', labelsize=tick_size)
            ax.set_xlim(0, 50)
            
            ax.set_xticks(np.arange(0, 41, 10))
            ax.set_yticks(np.arange(0, 0.91, 0.1))
            ax.hlines(0, xmin=0, xmax=50, color='k', linewidth=1, linestyle='--')
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
    fig.savefig(f'zpt2_o2a_{var}_compare.png', dpi=300)


def fitting_plot():
    slope_amp_list = [0.454, 0.141, 0.288]
    slope_amp_unc_list = [0.126, 0.042, 0.070]
    slope_dec_list = [4.75, 5.98, 5.34]
    slope_dec_unc_list = [0.76, 1.12, 0.78]

    inter_amp_list = [0.800, 0.721, 0.869]
    inter_amp_unc_list = [0.327, 0.275, 0.449]
    inter_dec_list = [2.64, 2.81, 2.34]
    inter_dec_unc_list = [0.32, 0.32, 0.35]

    fig, (ax11, ax12) = plt.subplots(1, 2, figsize=(16, 6), sharex=False)
    fig.tight_layout(pad=5.0)
    label_size = 20
    tick_size = 16

    wcld = np.linspace(0, 50, 101)
    
    for ax in [ax11, ax12]: 
        ax.set_xlabel('Weighted Average Cloud distance (km)', fontsize=label_size)
        ax.tick_params(axis='both', labelsize=tick_size)
        
        
    ax11.set_ylabel('slope', fontsize=label_size)
    ax12.set_ylabel('intercept', fontsize=label_size)
    
    bands = ['O_2-A', 'WCO_2', 'SCO_2']
    bands_color = ['red', 'green', 'blue']
    for i in range(3):
        line_arg = {'linewidth': 2, 'label': '$\mathrm{%s}$' %bands[i], 'color': bands_color[i]}
        slope_line_low = exp_decay_func_dec(wcld, slope_amp_list[i]-slope_amp_unc_list[i], slope_dec_list[i]-slope_dec_unc_list[i])
        slope_line_high = exp_decay_func_dec(wcld, slope_amp_list[i]+slope_amp_unc_list[i], slope_dec_list[i]+slope_dec_unc_list[i])
        ax11.fill_between(wcld, slope_line_low, slope_line_high, color=bands_color[i], alpha=0.25)

        inter_line_low = exp_decay_func_dec(wcld, inter_amp_list[i]-inter_amp_unc_list[i], inter_dec_list[i]-inter_dec_unc_list[i])
        inter_line_high = exp_decay_func_dec(wcld, inter_amp_list[i]+inter_amp_unc_list[i], inter_dec_list[i]+inter_dec_unc_list[i])
        ax12.fill_between(wcld, inter_line_low, inter_line_high, color=bands_color[i], alpha=0.25)
        
        ax11.plot(wcld, exp_decay_func_dec(wcld, slope_amp_list[i], slope_dec_list[i]), **line_arg)
        ax12.plot(wcld, exp_decay_func_dec(wcld, inter_amp_list[i], inter_dec_list[i]), **line_arg)
    
    cld_low, cld_max = 0, 45
    limit_1 = 0.5
    limit_2 = 0.9
    for ax in [ax11, ax12]:
        ax.set_xlim(cld_low, cld_max)
        ax.legend(fontsize=label_size)
        
        _, xmax = ax.get_xlim()
        ax.hlines(0, 0, xmax, linestyle='--', color='k')
    ax11.set_ylim(-0.05, limit_1)
    ax12.set_ylim(-0.05, limit_2)
    plt.show()
    fig.savefig(f'central_asia_test2_three_band.png', dpi=150, bbox_inches='tight')


def exp_decay_func(x, a, b):
     return a * np.exp(-b * x)

def exp_decay_func_dec(x, a, b):
     return a * np.exp(x/(-1*b))


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
    # fitting_plot()

    print(f'{(time.time()-now)/60:.3f} min')