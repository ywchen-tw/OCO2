import sys
sys.path.append('/Users/yuch8913/miniforge3/envs/er3t_env/lib/python3.8/site-packages')
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm, colors
from matplotlib import font_manager
import seaborn as sns
from tool_code import *
import os, pickle 
import uncertainties.unumpy as unp
import uncertainties as unc
import cartopy.crs as ccrs
import cartopy.feature as cf  
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.img_tiles as cimgt
from urllib.request import urlopen, Request
import io
from PIL import Image
from util.oco_cfg import grab_cfg, output_h5_info

font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

def main(cfg_name='20151201_ocean_1_cal_para.csv'):
    # get files in the directory
    file_list = sorted(os.listdir('20151201_ocean_1_cal_para_ontop'))
    xco2_list = []
    snd_list = []
    for file_name in file_list:
        if file_name.endswith('.h5'):
            print(f'Processing {file_name}')
            with h5py.File(f'20151201_ocean_1_cal_para_ontop/{file_name}', 'r') as f:
                # get the data
                snd = f['RetrievalHeader/sounding_id_reference'][...]
                xco2 = f['RetrievalResults/xco2'][...]
                snd_list.append('SND'+str(snd[0]))
                xco2_list.append(xco2[0]*1e6)
    df = pd.DataFrame({'snd': snd_list, 'xco2': xco2_list})
    df.to_csv('20151201_ocean_1_cal_para_ontop/20151201_ocean_1_cal_para_retrieve.csv', index=False)

    df = pd.read_csv('20151201_ocean_1_cal_para_ontop/20151201_ocean_1_cal_para_retrieve_2.csv')

    img_dir='20151201_ocean_1_cal_para_ontop'
    label_size=16
    tick_size=14

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), sharex=False)
    fig.tight_layout(pad=5.0)
    light_jet = cm.jet#cmap_map(lambda x: x/3 + 0.66, cm.jet)

    ax1.scatter(df['de'], df['xco2'], color='k', s=15)
    ax1.tick_params(axis='both', labelsize=tick_size)
    ymin1, ymax1 = ax1.get_ylim()
    xmin1, xmax1 = ax1.get_xlim()
    ax1.set_ylabel('Mitigated $\mathrm{X_{CO2}}$ (ppm)', fontsize=label_size)

    ax2.scatter(df['de'], df['xco2']-df['L2'], color='k', s=15)
    ax2.tick_params(axis='both', labelsize=tick_size)
    ymin2, ymax2 = ax2.get_ylim()
    xmin2, xmax2 = ax2.get_xlim()
    ax2.set_ylabel('$\Delta \mathrm{X_{CO2}}$ (ppm)', fontsize=label_size)
    ax2.hlines(0, xmin2, xmax2, linestyles='--', colors='grey')

    ymin = min(ymin1, ymin2)
    ymax = max(ymax1, ymax2)
    xmin = min(xmin1, xmin2)
    xmax = max(xmax1, xmax2)

    for ax, label in zip([ax1, ax2], ['(a)', '(b)']):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlabel('$\mathrm{D_e}$ (km)', fontsize=label_size)
        
        
        ax.text((xmin+(xmax-xmin)*-0.05), (ymin+(ymax-ymin)*1.05), label, fontsize=20)
        # ax.set_ylim(ymin, ymax)
    # ax.set_title('title', fontsize=title_size)
    fig.tight_layout(pad=2.0)
    fig.savefig(f'{img_dir}/test.png', dpi=300)


def XCO2_before_after_parameterization(img, wesn, lon_dom, lat_dom, df,
                                       img_dir='.', label_size=16, tick_size=14):
    f, (ax1, ax2, ax3) =plt.subplots(1, 3, figsize=(17, 6))

    for ax in [ax1, ax2, ax3]:
        ax.imshow(img, extent=wesn)
        ax.set_xlim(np.min(lon_dom), np.max(lon_dom))
        ax.set_ylim(np.min(lat_dom), np.max(lat_dom))
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.set_xlabel('Longitude ($^\circ$E)', fontsize=label_size)
        ax.set_ylabel('Latitude ($^\circ$N)', fontsize=label_size)
        
    mask = df['xco2_retrieved'][...]!=-2
    c1 = ax1.scatter(df['lon'], df['lat'], 
                    c=df['xco2_L2_file'], s=30,
                    cmap='RdBu_r', vmin=394, vmax=400)
    cbar1 = f.colorbar(c1, ax=ax1, extend='both')
    cbar1.set_label('Level2 $\mathrm{X_{CO2}}$ (ppm)', fontsize=label_size)
    cbar1.ax.tick_params(labelsize=tick_size)

    c2 = ax2.scatter(df['lon'], df['lat'], 
                c=df['xco2_retrieved'], s=30,
                cmap='RdBu_r', vmin=394, vmax=400)
    cbar2 = f.colorbar(c2, ax=ax2, extend='both')
    cbar2.set_label('Modified $\mathrm{X_{CO2}}$ (ppm)', fontsize=label_size)
    cbar2.ax.tick_params(labelsize=tick_size)

    c3 = ax3.scatter(df['lon'], df['lat'],
                c=df['diff_xco2'], s=30,
                cmap='RdBu_r', vmin=-6, vmax=6)
    cbar3 = f.colorbar(c3, ax=ax3, extend='both')
    cbar3.set_label('$\Delta \mathrm{X_{CO2}}$ (ppm)', fontsize=label_size)
    cbar3.ax.tick_params(labelsize=tick_size)

    for ax, label in zip([ax1, ax2, ax3], ['(a)', '(b)', '(c)']):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.025*(ymax-ymin), label, fontsize=label_size+4, color='k')

    f.tight_layout(pad=0.2)
    f.savefig(f'{img_dir}/MODIS_XCO2_retrieval_before_after_parameterization.png', dpi=300)

def XCO2_before_after_pixel(img, wesn, lon_dom, lat_dom, df,
                                       img_dir='.', label_size=16, tick_size=14):
    f, (ax1, ax2, ax3) =plt.subplots(1, 3, figsize=(17, 6))

    for ax in [ax1, ax2, ax3]:
        ax.imshow(img, extent=wesn)
        ax.set_xlim(np.min(lon_dom), np.max(lon_dom))
        ax.set_ylim(np.min(lat_dom), np.max(lat_dom))
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.set_xlabel('Longitude ($^\circ$E)', fontsize=label_size)
        ax.set_ylabel('Latitude ($^\circ$N)', fontsize=label_size)
        
    mask = df['xco2_retrieved'][...]!=-2
    c1 = ax1.scatter(df['lon'], df['lat'], 
                    c=df['xco2_L2_file'], s=30,
                    cmap='RdBu_r', vmin=394, vmax=400)
    cbar1 = f.colorbar(c1, ax=ax1, extend='both')
    cbar1.set_label('Level2 $\mathrm{X_{CO2}}$ (ppm)', fontsize=label_size)
    cbar1.ax.tick_params(labelsize=tick_size)

    c2 = ax2.scatter(df['lon'], df['lat'], 
                c=df['xco2_retrieved'], s=30,
                cmap='RdBu_r', vmin=394, vmax=400)
    cbar2 = f.colorbar(c2, ax=ax2, extend='both')
    cbar2.set_label('Modified $\mathrm{X_{CO2}}$ (ppm)', fontsize=label_size)
    cbar2.ax.tick_params(labelsize=tick_size)

    c3 = ax3.scatter(df['lon'], df['lat'],
                c=df['diff_xco2'], s=30,
                cmap='RdBu_r', vmin=-6, vmax=6)
    cbar3 = f.colorbar(c3, ax=ax3, extend='both')
    cbar3.set_label('$\Delta \mathrm{X_{CO2}}$ (ppm)', fontsize=label_size)
    cbar3.ax.tick_params(labelsize=tick_size)

    for ax, label in zip([ax1, ax2, ax3], ['(a)', '(b)', '(c)']):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 181.0, 0.1)))
        ax.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 91.0, 0.1)))
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.025*(ymax-ymin), label, fontsize=label_size+4, color='k')

    f.tight_layout(pad=0.2)
    f.savefig(f'{img_dir}/MODIS_XCO2_retrieval_before_after_pixel.png', dpi=300)

def p_sfc_before_after_parameterization(img, wesn, lon_dom, lat_dom, df,
                                       img_dir='.', label_size=16, tick_size=14):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 6))

    for ax in [ax1, ax2, ax3]:
        ax.imshow(img, extent=wesn)
        ax.set_xlim(np.min(lon_dom), np.max(lon_dom))
        ax.set_ylim(np.min(lat_dom), np.max(lat_dom))
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.set_xlabel('Longitude ($^\circ$E)', fontsize=label_size)
        ax.set_ylabel('Latitude ($^\circ$N)', fontsize=label_size)
        
    mask = df['xco2_retrieved'][...]!=-2
    c1 = ax1.scatter(df['lon'], df['lat'], 
                    c=df.psur_MT_file*10, s=30,
                    cmap='OrRd', vmin=990, vmax=1010)
    cbar1 = f.colorbar(c1, ax=ax1, extend='both')
    cbar1.set_label('Level 2 $\mathrm{P_{sfc}}$ (hPa)', fontsize=label_size)
    cbar1.ax.tick_params(labelsize=tick_size)

    c2 = ax2.scatter(df['lon'], df['lat'], 
                c=df.psur_retrieved/100, s=30,
                cmap='OrRd', vmin=990, vmax=1010)
    cbar2 = f.colorbar(c2, ax=ax2, extend='both')
    cbar2.set_label('Modified $\mathrm{P_{sfc}}$ (hPa)', fontsize=label_size)
    cbar2.ax.tick_params(labelsize=tick_size)

    c3 = ax3.scatter(df['lon'], df['lat'],
                c=df.psur_retrieved/100-df.psur_MT_file*10, s=30,
                cmap='RdBu_r', vmin=-10, vmax=10)
    cbar3 = f.colorbar(c3, ax=ax3, extend='both')
    cbar3.set_label('$\Delta$ $\mathrm{P_{sfc}}$ (hPa)', fontsize=label_size)
    cbar3.ax.tick_params(labelsize=tick_size)

    for ax, label in zip([ax1, ax2, ax3], ['(a)', '(b)', '(c)']):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.025*(ymax-ymin), label, fontsize=label_size+4, color='k')

    f.tight_layout(pad=0.5)
    f.savefig(f'{img_dir}/P_sfc_retrieval_before_after.png', dpi=300)

def aod_before_after_parameterization(img, wesn, lon_dom, lat_dom, df, l2_aod,
                                       img_dir='.', label_size=16, tick_size=14):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 6))

    for ax in [ax1, ax2, ax3]:
        ax.imshow(img, extent=wesn)
        ax.set_xlim(np.min(lon_dom), np.max(lon_dom))
        ax.set_ylim(np.min(lat_dom), np.max(lat_dom))
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.set_xlabel('Longitude ($^\circ$E)', fontsize=label_size)
        ax.set_ylabel('Latitude ($^\circ$N)', fontsize=label_size)
        
    mask = df['xco2_retrieved'][...]!=-2
    c1 = ax1.scatter(df['lon'], df['lat'], 
                    c=l2_aod, s=30,
                    cmap='OrRd', vmin=0.06, vmax=0.12)
    cbar1 = f.colorbar(c1, ax=ax1, extend='both')
    cbar1.set_label('Level 2 AOD', fontsize=label_size)
    cbar1.ax.tick_params(labelsize=tick_size)

    c2 = ax2.scatter(df['lon'], df['lat'], 
                c=df.aod, s=30,
                cmap='OrRd', vmin=0.06, vmax=0.12)
    cbar2 = f.colorbar(c2, ax=ax2, extend='both')
    cbar2.set_label('Modified AOD', fontsize=label_size)
    cbar2.ax.tick_params(labelsize=tick_size)

    c3 = ax3.scatter(df['lon'], df['lat'],
                c=df.aod-l2_aod, s=30,
                cmap='RdBu_r', vmin=-0.04, vmax=0.04)
    cbar3 = f.colorbar(c3, ax=ax3, extend='both')
    cbar3.set_label('$\Delta$ AOD', fontsize=label_size)
    cbar3.ax.tick_params(labelsize=tick_size)

    for ax, label in zip([ax1, ax2, ax3], ['(a)', '(b)', '(c)']):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.025*(ymax-ymin), label, fontsize=label_size+4, color='k')

    f.tight_layout(pad=0.5)
    f.savefig(f'{img_dir}/AOD_retrieval_before_after.png', dpi=300)

def delta_XCO2_3_bands_parameterization(df,
                                       img_dir='.', label_size=14, tick_size=12):
    fig = plt.figure(figsize=(12, 14))
    ax11 = fig.add_axes([0.05, 0.7, 0.4, 0.2])
    ax12 = fig.add_axes([0.55, 0.7, 0.4, 0.2])
    ax21 = fig.add_axes([0.05, 0.425, 0.4, 0.2])
    ax22 = fig.add_axes([0.55, 0.425, 0.4, 0.2])
    ax31 = fig.add_axes([0.05, 0.15, 0.4, 0.2])
    ax32 = fig.add_axes([0.55, 0.15, 0.4, 0.2])

    ax_cbar = fig.add_axes([0.25, 0.075, 0.5, 0.015])
    plot_setting = dict(s=50, 
                    c=df['xco2_L2_file'].values,
                    alpha=0.85,
                    cmap='OrRd', vmin=400, vmax=415, edgecolor='k')

    c = ax11.scatter(df.o2a_slope.values, df['diff_xco2'], **plot_setting) 
    cbar = fig.colorbar(c, cax=ax_cbar, extend='both', orientation='horizontal')
    cbar.set_label('Level 2 $\mathrm{XCO_2}$ (ppm)', fontsize=label_size)
    cbar.ax.tick_params(labelsize=tick_size)

    ax12.scatter(df.o2a_inter.values, df['diff_xco2'], **plot_setting)
    ax21.scatter(df.wco2_slope.values, df['diff_xco2'], **plot_setting)
    ax22.scatter(df.wco2_inter.values, df['diff_xco2'], **plot_setting)
    ax31.scatter(df.sco2_slope.values, df['diff_xco2'], **plot_setting)
    ax32.scatter(df.sco2_inter.values, df['diff_xco2'], **plot_setting)

    ax_list = [ax11, ax12, ax21, ax22, ax31, ax32]
    label_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for ax, label in zip(ax_list, label_list):
        ax.set_ylabel('$\Delta \mathrm{XCO_2}$ (ppm)', fontsize=label_size)
        ax.tick_params(axis='both', labelsize=tick_size)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.05*(ymax-ymin), label, fontsize=label_size+2, color='k')

    ax11.set_xlabel('$\mathrm{O_2-A}$ slope', fontsize=label_size)
    ax12.set_xlabel('$\mathrm{O_2-A}$ intercept', fontsize=label_size)
    ax21.set_xlabel('$\mathrm{WCO_2}$ slope', fontsize=label_size)
    ax22.set_xlabel('$\mathrm{WCO_2}$ intercept', fontsize=label_size)
    ax31.set_xlabel('$\mathrm{SCO_2}$ slope', fontsize=label_size)
    ax32.set_xlabel('$\mathrm{SCO_2}$ intercept', fontsize=label_size)
    fig.tight_layout(pad=1.0)
    fig.savefig(f'{img_dir}/Delta_XCO2_3d_para_3_bands_20181018.png', dpi=300)

def o2a_spectra_modified_fig(df, fp, o2a_rad_convert, lam,
                             img_dir='.', label_size=14, tick_size=12):
    o2a_slope = df['o2a_slope'][fp]
    o2a_inter = df['o2a_inter'][fp]
    o2a_ref = df['rfl1'][fp]

    o2a_rad_unperturbed = o2a_rad_convert/(1+o2a_inter+o2a_slope*o2a_ref)
    f, ax=plt.subplots(figsize=(8, 4))

    rad_max = o2a_rad_convert.max()
    ax.plot(lam[1, :]*1e3, o2a_rad_convert/rad_max, 'k', linewidth=1, label='original')
    ax.plot(lam[1, :]*1e3, o2a_rad_unperturbed/rad_max, 'r', linewidth=1, label='adjusted', alpha=0.75)


    ax.tick_params(axis='both', labelsize=tick_size)

    ax.set_xlim(lam[1, :].min()*1e3, lam[1, :].max()*1e3)
    ax.legend(fontsize=14, facecolor='white')
    ax.set_xlabel('wavelength (nm)', fontsize=label_size)
    ax.set_ylabel('Normalized radiance (a.u.)', fontsize=label_size)
    f.tight_layout()
    f.savefig(f'{img_dir}/O2A_spectrum_before_after_unperturbation.png', dpi=300)

def scene_google_map(img, wesn, lon_dom, lat_dom, o1,
                    img_dir='.', label_size=14, tick_size=12):
    cimgt.GoogleWTS.get_image = new_get_image
    osm_img = cimgt.GoogleTiles(style='satellite')
    osm_img_street = cimgt.GoogleTiles(style='street')
    fig = plt.figure(figsize=(22, 8)) # open matplotlib figure

    ax2 = fig.add_axes([0.05, 0.1, 0.35, 0.8])
    ax1 = fig.add_axes([0.40, 0.1, 0.3, 0.8], projection=osm_img.crs)
    ax12 = fig.add_axes([0.70, 0.1, 0.3, 0.8], projection=osm_img_street.crs)

    #ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
    
    center_pt = [np.mean(lat_dom), 
                np.mean(lon_dom)] # lat/lon of One World Trade Center in NYC
    zoom = 0.0425 # for zooming out of center point
    extent = [center_pt[1]-(zoom*2.0),center_pt[1]+(zoom*2.0),center_pt[0]-zoom,center_pt[0]+zoom] # adjust to zoom
    extent = lon_dom+lat_dom#[-105.273, -105.257, 40.002, 40.01]
    ax1.set_extent(extent) # set extents
    ax1.set_xticks(np.linspace(55.15, 55.45, 4),crs=ccrs.PlateCarree()) # set longitude indicators
    ax1.set_yticks(np.linspace(33.9, 34.3,6),crs=ccrs.PlateCarree()) # set latitude indicatorslon_formatter = LongitudeFormatter(number_format='0.3f',degree_symbol='',dateline_direction_label=True) # format lons
    lon_formatter = LongitudeFormatter(number_format='0.2f', degree_symbol='$^\circ$', dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.1f', degree_symbol='$^\circ$') # format lats
    ax1.xaxis.set_major_formatter(lon_formatter) # set lons
    ax1.yaxis.set_major_formatter(lat_formatter) # set lats
    ax1.tick_params(axis='both', labelsize=tick_size)
    scale = np.ceil(-np.sqrt(2)*np.log(np.divide(zoom,350.0))) # empirical solve for scale based on zoom
    scale = (scale<20) and scale or 19 # scale cannot be larger than 19
    ax1.add_image(osm_img, int(scale), alpha=0.75) # add OSM with zoom specification

    ax12.set_extent(extent) # set extents
    ax12.set_xticks(np.linspace(55.15, 55.45, 4),crs=ccrs.PlateCarree()) # set longitude indicators
    ax12.set_yticks(np.linspace(33.9, 34.3,6),crs=ccrs.PlateCarree()) # set latitude indicatorslon_formatter = LongitudeFormatter(number_format='0.3f',degree_symbol='',dateline_direction_label=True) # format lons
    lon_formatter = LongitudeFormatter(number_format='0.2f', degree_symbol='$^\circ$', dateline_direction_label=True) # format lons
    lat_formatter = LatitudeFormatter(number_format='0.1f', degree_symbol='$^\circ$') # format lats
    ax12.xaxis.set_major_formatter(lon_formatter) # set lons
    ax12.yaxis.set_major_formatter(lat_formatter) # set lats
    ax12.tick_params(axis='both', labelsize=tick_size)
    scale = np.ceil(-np.sqrt(2)*np.log(np.divide(zoom,350.0))) # empirical solve for scale based on zoom
    scale = (scale<20) and scale or 19 # scale cannot be larger than 19
    ax12.add_image(osm_img_street, int(scale), alpha=0.75) # add OSM with zoom specification


    ax2.imshow(img, extent=wesn)
    ax2.set_xlim(np.min(lon_dom), np.max(lon_dom))
    ax2.set_ylim(np.min(lat_dom), np.max(lat_dom))
    ax2.vlines(lon_dom, ymin=wesn[2]+0.15, ymax=wesn[3]-0.15, color='k', linewidth=1)
    ax2.hlines(lat_dom, xmin=wesn[0]+0.15, xmax=wesn[1]-0.15, color='k', linewidth=1)
    mask = np.isnan(getattr(o1, 'rad_c3d')[:,:,-1])
    print(mask.sum())
    c = ax2.scatter(o1.lon, o1.lat, 
                c=getattr(o1, 'co2')*1e6-404, s=45,
                cmap='RdBu_r', vmin=-5, vmax=5)

    cbar = fig.colorbar(c, ax=ax2, extend='both')
    cbar.set_label('$\mathrm{XCO_2}$ - 403 (ppmv)', fontsize=label_size)
    ax2.tick_params(axis='both', labelsize=tick_size)
    ax2.set_xlabel('Longitude ($^\circ$E)', fontsize=label_size)
    ax2.set_ylabel('Latitude ($^\circ$N)', fontsize=label_size)

    for ax, label in zip([ax2, ax1, ax12], ['(a)', '(b)', '(c)']):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.025*(ymax-ymin), label, fontsize=label_size+4, color='k')
    fig.tight_layout(pad=3.0)
    fig.savefig(f'{img_dir}/MODIS_20181018_vs_google_maps.png', dpi=300)

def func(x, a, b):
    """The fitting function"""
    return a*(np.array(x))+b

def fp_perturbation_ref_fitting(o1, trnsx, refl, oco_tx,
                                fp=160, z=130,
                                img_dir='.', label_size=16, tick_size=14):
    f, (ax1, ax) =plt.subplots(1, 2, figsize=(14, 5))
    f.tight_layout(pad=3.0)


    # first fig
    x = np.arange(1016)
    sx = np.argsort(trnsx)
    y = trnsx[sx]*refl
    ax1.scatter(x, y, color='k', s=3)

    ax1.tick_params(axis='both', labelsize=tick_size)

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    ax1.set_xlabel('Wavelength index', fontsize=label_size)
    ax1.set_ylabel('Transmittance', fontsize=label_size)

    # plot setting
    norm = colors.Normalize(vmin=0.0, vmax=255.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.rainbow)
    for i in range(11):
        wli0 = np.where(sx==np.argmin(np.abs(y-oco_tx[i])))[0][0]
        ax1.plot([0,1016],[oco_tx[i],oco_tx[i]],color='orange',linestyle='dotted')
        cl = 30*(i+1)
        ax1.plot([sx[wli0], sx[wli0]], [0,oco_tx[i]], linestyle='dashed', color=mapper.to_rgba(cl), linewidth=2)


    # second fig
    toa = o1.toa
    mu = np.cos(o1.sza_avg/180*np.pi)
    print(mu)
    sl_np = o1.sl_5
    sls_np = o1.sls_5#/np.sqrt(3)
    c3d_np = o1.rad_c3d_5
    clr_np = o1.rad_clr_5
    points = 11

    w = 1./sls_np[z,fp,:] 

    x = c3d_np[z,fp,:]/(toa[:]*mu)*np.pi


    x_len = len(x)
    mask = np.argsort(x)[x_len-points:]
    res=np.polyfit(x[mask], sl_np[z,fp,:][mask], 1, w=w[mask], cov=True) # now get covariance as well!
    slope,intercept=res[0]
    slopes=np.sqrt(res[1][0][0])
    intercepts=np.sqrt(res[1][1][1])

    ax.errorbar(x[mask], sl_np[z,fp,:][mask]*100, yerr=sls_np[z,fp,:]*100, color='k',
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


    #ax.set_xticks(range(0, 160, 20))
    ax.tick_params(axis='both', labelsize=tick_size)

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    #ymin, ymax = -1., 1.
    xmin, xmax = 0., xmax*1.1
    #ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, xmax)
    # ax.text((xmin+(xmax-xmin)*0.05), (ymin+(ymax-ymin)*0.9), '(a)', fontsize=18)

    # ax.legend(loc='center left', bbox_to_anchor=(0.65, 0.15), fontsize=legend_size)
    ax.set_xlabel('Reflectance', fontsize=label_size)
    ax.set_ylabel('Perturbation (%)', fontsize=label_size)


    popt, pcov = res[0], res[1]
    # calculate parameter confidence interval
    a, b = unc.correlated_values(popt, pcov)
    px = np.linspace(0, xmax, num=50, endpoint=True) 
    py = a*px+b
    nom = unp.nominal_values(py)*100
    std = unp.std_devs(py)*100
    # plot the regression line and uncertainty band (95% confidence)
    ax.plot(px, nom, c='r')
    ax.fill_between(px, nom - 1.96 * std, nom + 1.96 * std, color='orange', alpha=0.2)

    for ax, label in zip([ax1, ax], ['(a)', '(b)']):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.025*(ymax-ymin), label, fontsize=label_size+4, color='k')
    f.tight_layout(pad=1.0)
    f.savefig(f'{img_dir}/wavelength_select_and_slope_inter_derive.png', dpi=300)

def delta_XCO2_comparison_para_pixel(cld_dist, df, df_pixel,
                                     img_dir='.', label_size=16, tick_size=14):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), sharex=False)
    fig.tight_layout(pad=5.0)
    light_jet = cm.jet#cmap_map(lambda x: x/3 + 0.66, cm.jet)

    ax1.scatter(cld_dist, df['diff_xco2'], color='k', s=15)
    ax1.tick_params(axis='both', labelsize=tick_size)
    ymin1, ymax1 = ax1.get_ylim()
    xmin1, xmax1 = ax1.get_xlim()

    ax2.scatter(cld_dist, df_pixel['diff_xco2'], color='k', s=15)
    ax2.tick_params(axis='both', labelsize=tick_size)
    ymin2, ymax2 = ax2.get_ylim()
    xmin2, xmax2 = ax2.get_xlim()

    ymin = min(ymin1, ymin2)
    ymax = max(ymax1, ymax2)
    xmin = min(xmin1, xmin2)
    xmax = max(xmax1, xmax2)

    for ax, label in zip([ax1, ax2], ['(a)', '(b)']):
        # xmin, xmax = ax.get_xlim()
        # ymin, ymax = ax.get_ylim()
        ax.set_xlabel('$\mathrm{D_e}$ (km)', fontsize=label_size)
        ax.set_ylabel('$\Delta \mathrm{X_{CO2}}$ (ppm)', fontsize=label_size)
        ax.hlines(0, xmin, xmax, linestyles='--', colors='grey')
        ax.text((xmin+(xmax-xmin)*-0.05), (ymin+(ymax-ymin)*1.05), label, fontsize=20)
        ax.set_ylim(ymin, ymax)
    # ax.set_title('title', fontsize=title_size)
    fig.tight_layout(pad=2.0)
    fig.savefig(f'{img_dir}/Delta_XCO2_comparison_cld_para_pixel-by-pixel.png', dpi=300)

def co2_prfl_diff(co2_prf, l2_co2_profile, l2_alt_level, l2_avg_kernel,
                  img_dir='.', label_size=16, tick_size=14, legend_size=14):
    x = (co2_prf-l2_co2_profile).mean(axis=0)
    x_std = (co2_prf-l2_co2_profile).std(axis=0)
    y = l2_alt_level.mean(axis=0)/1000

    x2 = l2_avg_kernel.mean(axis=0)
    x2_std = l2_avg_kernel.std(axis=0)


    f, ax1=plt.subplots(figsize=(6, 8))

    # rad_max = o2a_rad_convert.max()
    ln1 = ax1.plot(x, y, 'k', linewidth=1, label='$\Delta \mathrm{XCO_2}$')
    ax1.fill_betweenx(y, x-x_std, x+x_std, color='lightgrey')

    ax1.vlines(0, y.min(), 20, linestyle='--', colors='grey')

    ax2 = ax1.twiny()
    ln2 = ax2.plot(x2, y, 'blue', linewidth=1, label='averaging kernel')
    ax2.fill_betweenx(y, x2-x2_std, x2+x2_std, color='skyblue')

    # added these three lines
    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, fontsize=legend_size)

    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.set_ylim(y.min(), 20)
        #ax.legend(fontsize=14, facecolor='white')

    ax1.set_xlabel('$\Delta \mathrm{XCO_2}$ (ppm)', fontsize=label_size)    
    ax1.set_ylabel('Altitude (m)', fontsize=label_size)
    ax2.set_xlabel('$\mathrm{XCO_2}$ averaging kernel (a.u.)', fontsize=label_size)

    f.tight_layout()
    f.savefig(f'{img_dir}/co2_retrieval_profile_avg_kernel')


def xco2_spread(cld_dist, xco2_l2, xco2_unpert,
                img_dir='.', label_size=16, tick_size=14, legend_size=14):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), sharex=False, )
    
    cld_dist_threshold = 15
    true_mean = np.mean(xco2_l2[cld_dist>cld_dist_threshold])
    x = cld_dist
    y = xco2_l2-true_mean

    print(np.mean(xco2_l2[cld_dist>cld_dist_threshold]))
    print(np.mean(xco2_l2[cld_dist>(cld_dist_threshold*2)]))
    print(np.mean(xco2_unpert[cld_dist>cld_dist_threshold]))
    ax.hlines(0, 0, 40, linestyle='--', color='orange')
    ax.fill_between([0, 40], -1, 1, color='orange', alpha=0.3)

    ax.fill_betweenx([-10, 10], 0, cld_dist_threshold, color='skyblue', alpha=0.25)

    ax.scatter(x, y, color='k', label='OCO-2 product (v10r)')

    #ax.set_xticks(range(0, 160, 20))
    ax.tick_params(axis='both', labelsize=tick_size)


    x = cld_dist
    y = xco2_unpert-true_mean
 
    print(f'true mean: {true_mean}')
    
    ax.scatter(x, y, color='r', s=25, alpha=0.75, label='Mitigated')

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = -10., 10.
    xmin, xmax = -0.5, 40
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.text((xmin+(xmax-xmin)*0.405), (ymin+(ymax-ymin)*0.72), 
            'regional uncertainty requirement:\n                      1 ppm', color='orange', fontsize=14)

    
    cld_mask = cld_dist<= cld_dist_threshold
    ax_histy = ax.inset_axes([1.1, 0, 0.25, 1], sharey=ax)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.hist((xco2_l2-true_mean)[cld_mask], bins=31, orientation='horizontal', density=True, color='black', alpha=0.75)

    ax_histy.hist(y[cld_mask], bins=31, orientation='horizontal', density=True, color='r', alpha=0.75)

    black_kde = sns.kdeplot(y=(xco2_l2-true_mean)[cld_mask], color='k', bw_adjust=1.5, linewidth=3, 
                            ax=ax_histy, alpha=0.95)
    print(f'l2 cloud dist <= {cld_dist_threshold} anomaly average: {np.nanmean((xco2_l2-true_mean)[cld_mask]):.3f} +/- {np.nanstd((xco2_l2-true_mean)[cld_mask]):.3f}')

    kde_curve = black_kde.lines[0]
    black_kde_x = kde_curve.get_xdata()
    black_kde_y = kde_curve.get_ydata()
    halfmax = black_kde_x.max() / 2
    maxpos = black_kde_x.argmax()
    leftpos = (np.abs(black_kde_x[:maxpos] - halfmax)).argmin()
    rightpos = (np.abs(black_kde_x[maxpos:] - halfmax)).argmin() + maxpos
    fullwidthathalfmax = black_kde_y[rightpos] - black_kde_y[leftpos]

    ax_histy.vlines(halfmax, black_kde_y[leftpos], black_kde_y[rightpos], color='k', ls=':', linewidth=3)
    ax_histy.text(0.2, 2.75,
                #halfmax*2,
                f'{fullwidthathalfmax:.2f}\n',
                color='k', ha='center', va='center', fontsize=legend_size)

    ax_histy.text(0.2, 4.75,
                #halfmax*2,
                'FWHM',
                color='k', ha='center', va='center', fontsize=legend_size)


    print(fullwidthathalfmax)


    red_kde = sns.kdeplot(y=(xco2_unpert-true_mean)[cld_mask], color='r', bw_adjust=1.5, linewidth=3, 
                        ax=ax_histy, alpha=0.95)

    print(f'unperturbed cloud dist <= {cld_dist_threshold} anomaly average: {np.nanmean((xco2_unpert-true_mean)[cld_mask]):.3f} +/- {np.nanstd((xco2_unpert-true_mean)[cld_mask]):.3f}')

    red_kde_curve = red_kde.lines[1]
    red_kde_x = red_kde_curve.get_xdata()
    red_kde_y = red_kde_curve.get_ydata()
    halfmax = red_kde_x.max() / 2
    maxpos = red_kde_x.argmax()
    leftpos = (np.abs(red_kde_x[:maxpos] - halfmax)).argmin()
    rightpos = (np.abs(red_kde_x[maxpos:] - halfmax)).argmin() + maxpos
    fullwidthathalfmax = red_kde_y[rightpos] - red_kde_y[leftpos]

    ax_histy.vlines(halfmax, red_kde_y[leftpos], red_kde_y[rightpos], color='r', ls=':', linewidth=3)
    print(red_kde_y[maxpos])
    ax_histy.text(0.2, -4.05,
                #halfmax*2,
                f'{fullwidthathalfmax:.2f}\n',
                color='r', ha='center', va='center', fontsize=legend_size)

    ax_histy.text(0.195, 8.75,
                #halfmax*2,
                'cloud distance\n$\mathrm{\leq}$ %s km' %cld_dist_threshold,
                color='k', ha='center', va='center', fontsize=legend_size-4)


    hist_xmin, hist_xmax = ax_histy.get_xlim()
    ax_histy.set_xlim(hist_xmin, hist_xmax)
    ax_histy.tick_params(axis='both', labelsize=tick_size)
    ax_histy.set_xlabel('Density', fontsize=label_size)

    ax_histy.hlines(0, hist_xmin, hist_xmax, linestyle='--', color='orange')
    ax_histy.hlines([-1, 1], hist_xmin, hist_xmax, linestyle='-', color='orange')
    # ax_histy.fill_between([hist_xmin, hist_xmax], -1, 1, color='orange', alpha=0.3)
    ax_histy.fill_betweenx([-10, 10], hist_xmin, hist_xmax, color='skyblue', alpha=0.25)

    ax.legend(loc='center left', bbox_to_anchor=(0.39, 0.11), fontsize=legend_size)
    ax.set_xlabel('$\mathrm{D_e}$ (km)', fontsize=label_size)
    ax.set_ylabel('$\mathrm{X_{CO2}}$ anomaly (ppm)', fontsize=label_size)
    
    for ax, label in zip([ax, ax_histy], ['(a)', '(b)']):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text((xmin+(xmax-xmin)*-0.05), (ymin+(ymax-ymin)*1.05), label, fontsize=20)
    
    fig.tight_layout(pad=2.0)
    fig.savefig(f'{img_dir}/Delta_XCO2_cloud_distance_cld_parameterization.png', dpi=300)

def fp_pol_ang_compare(l1b_polar_ang, l1b_lon, l1b_lat,
                        img, wesn, lon_dom, lat_dom, 
                        tick_size=12, label_size=14, img_dir='.'):
    f, (ax1, ax2, ax3)=plt.subplots(1, 3, figsize=(24, 7.5))
    for ax in [ax1, ax2, ax3]:
        ax.imshow(img, extent=wesn)
        ax.set_xlim(np.min(lon_dom), np.max(lon_dom))
        ax.set_ylim(np.min(lat_dom), np.max(lat_dom))
        ax_lon_lat_label(ax, label_size=14, tick_size=12)

    # ax1
    vmin, vmax = 0, 360
    lev = np.arange(vmin, vmax+1e-7, 5)
    scatter_arg = {'s': 60, 'cmap': 'jet', 'marker': 'o', 'edgecolors': 'k'}
    print(l1b_lon.shape)
    print(l1b_polar_ang.shape)
    for i, ax in zip(range(3), [ax1, ax2, ax3]):
        
        cc = ax.scatter(l1b_lon[:, 0, i], l1b_lat[:, 0, i],
                    c=l1b_polar_ang[:, 0, i],
                    # vmin=vmin, vmax=vmax, 
                    **scatter_arg)
        cbar = f.colorbar(cc, ax=ax)
        cbar.set_label('Polarization angle ($\deg$)', fontsize=16)

    for ax, label in zip([ax1, ax2, ax3], ['(a)', '(b)', '(c)']):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(xmin+0.0*(xmax-xmin), ymin+1.025*(ymax-ymin), label, fontsize=label_size+4, color='k')
    
    f.tight_layout(pad=2.0)
    f.savefig(f'{img_dir}/pol_ang_distribution.png', dpi=300)

def ax_lon_lat_label(ax, label_size=14, tick_size=12):
    ax.set_xlabel('Longitude ($^\circ$E)', fontsize=label_size)
    ax.set_ylabel('Latitude ($^\circ$N)', fontsize=label_size)
    ax.tick_params(axis='both', labelsize=tick_size)

if __name__ == "__main__":
    now = time.time()
    
    main()

    print(f'{(time.time()-now)/60:.3f} min')