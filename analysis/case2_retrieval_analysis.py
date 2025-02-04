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

def main(cfg_name='20191005_central_asia.csv'):
    # 20181018_central_asia_2
    cfg_dir = '../simulation/cfg'
    cfg_info = grab_cfg(f'{cfg_dir}/{cfg_name}')
    if 'o2' in cfg_info.keys():
        id_num = output_h5_info(f'{cfg_dir}/{cfg_name}', 'o2')[-12:-3]
    else:
        sys.exit('Error: no output file in cfg_info[o2]')


    cfg_name = cfg_info['cfg_name']
    date   = datetime.datetime(int(cfg_info['date'][:4]),    # year
                               int(cfg_info['date'][4:6]),   # month
                               int(cfg_info['date'][6:])     # day
                              )
    case_name_tag = '%s_%s' % (cfg_info['cfg_name'], date.strftime('%Y%m%d'))
    extent_analysis = [float(loc) - offset for loc, offset in zip(cfg_info['subdomain'], [-0.15, 0.15, -0.15, 0.15])]
    # extent_png = [float(loc) for loc in cfg_info['subdomain']]
    extent_analysis = [float(loc) for loc in cfg_info['subdomain']]
    extent_png = [float(loc) + offset for loc, offset in zip(cfg_info['subdomain'], [-0.15, 0.15, -0.15, 0.15])]

    img_file = f'../simulation/data/{case_name_tag}/{cfg_info["png"]}'
    wesn = extent_png
    img = mpimg.imread(img_file)
    lon_dom = extent_analysis[:2]
    lat_dom = extent_analysis[2:]

    png       = [img_file, wesn]
    title_size = 16
    label_size = 14
    legend_size = 14
    tick_size = 12

    cld_data = np.load(f"../simulation/data/{case_name_tag}/dist_data_modis_cth_oco_cld_distance_weighted_{cfg_info['cfg_name']}.npy")
    cld_dist = cld_data#['cld_dis']

    # with h5py.File(f'../simulation/data/{case_name_tag}/pre-data.h5', 'r') as predata:
    #     modis_aod = predata['mod/aod/AOD_550_land'][...]
    #     modis_lon, modis_lat = predata['lon'][...], predata['lat'][...]
    # modis_aod[modis_aod<0] = np.nan
    img_dir = f'output/{case_name_tag}'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # aod_550_plot(img, wesn, lon_dom, lat_dom, modis_lon, modis_lat, modis_aod,
    #             img_dir=img_dir)
    # retrieval_aod_parameterization = h5py.File('full-unperturbed20181018_central_asia_2_test4_para_2.h5', 'r')
    retrieval_aod_parameterization = h5py.File('full-unperturbed20191005_central_asia.h5', 'r')
    mask = retrieval_aod_parameterization['xco2_retrieved'][...]!=-2
    key_list = ['aod', 'cpu_minutes', 'lat', 'lon', 'psur_MT_file', 'psur_retrieved',
                'rfl1', 'rfl2', 'rfl3', 'snd', 'xco2_L2_file', 'xco2_retrieved', 'xco2_weighted_column']
    df = pd.DataFrame({key:retrieval_aod_parameterization[key][...] for key in key_list})
    df['o2a_inter'] = retrieval_aod_parameterization['pert_o2'][...][:, 0]
    df['o2a_slope'] = retrieval_aod_parameterization['pert_o2'][...][:, 1]
    df['wco2_inter'] = retrieval_aod_parameterization['pert_wco2'][...][:, 0]
    df['wco2_slope'] = retrieval_aod_parameterization['pert_wco2'][...][:, 1]
    df['sco2_inter'] = retrieval_aod_parameterization['pert_sco2'][...][:, 0]
    df['sco2_slope'] = retrieval_aod_parameterization['pert_sco2'][...][:, 1]
    df.replace(-2, np.nan, inplace=True)
    df.loc[df['xco2_L2_file']<1, 'xco2_L2_file'] = df.loc[df['xco2_L2_file']<1, 'xco2_L2_file']*1e6
    df.drop_duplicates('snd', inplace=True)
    df['diff_xco2'] = df['xco2_retrieved']-df['xco2_L2_file']
    lonlat_mask = (df['lon']>=lon_dom[0]) & (df['lon']<=lon_dom[1]) & (df['lat']>=lat_dom[0]) & (df['lat']<=lat_dom[1])
    df = df[lonlat_mask]
    
    XCO2_l2_plot(img, wesn, lon_dom, lat_dom, df, img_dir=img_dir)
    XCO2_before_after_parameterization(img, wesn, lon_dom, lat_dom, df,
                                       img_dir=img_dir)
    print(df['psur_MT_file'].min(), df['psur_MT_file'].max())
    print(df['psur_retrieved'].min(), df['psur_retrieved'].max())
    p_sfc_before_after_parameterization(img, wesn, lon_dom, lat_dom, df,
                                       img_dir=img_dir)
    
    delta_XCO2_3_bands_parameterization(df, img_dir=img_dir)

    oco_l1b = h5py.File(f"{cfg_info['path_sat_data']}/oco/{cfg_info['l1b']}", 'r')
    oco_l1b_id = oco_l1b['SoundingGeometry/sounding_id'][...]

    dis = oco_l1b["InstrumentHeader/dispersion_coef_samp"][...]
    lam = np.zeros([8,1016]) # Those are the wavelengths in the radiance file
    wli = np.arange(1,1017,dtype=float)
    for i in range(8): 
        for k in range(5):
            lam[i,:]=lam[i,:] + dis[0,i,k]*wli**k

    o2a_rad = oco_l1b['SoundingMeasurements/radiance_o2'][...][6195, :]
    o2a_rad_convert = convert_photon_unit(o2a_rad[0], lam[1, :]*1e3)
    oco_l1b_pol_ang = oco_l1b['FootprintGeometry/footprint_polarization_angle'][...]
    oco_l1b_lon = oco_l1b['FootprintGeometry/footprint_longitude'][...]
    oco_l1b_lat = oco_l1b['FootprintGeometry/footprint_latitude'][...]
    oco_l1b_o2a = oco_l1b['SoundingMeasurements/radiance_o2'][...]
    oco_l1b_wco2 = oco_l1b['SoundingMeasurements/radiance_weak_co2'][...]
    oco_l1b_sco2 = oco_l1b['SoundingMeasurements/radiance_strong_co2'][...]
    oco_l1b_stokes = oco_l1b['FootprintGeometry/footprint_stokes_coefficients'][...]
    oco_l1b_bidir_angle = oco_l1b['FootprintGeometry/footprint_solar_surface_bidirectional_angle'][...]
    


    # o2a_spectra_modified_fig(df, 90, o2a_rad_convert, lam, img_dir=img_dir)
    
    oco_l2 = h5py.File(f"{cfg_info['path_sat_data']}/oco/{cfg_info['l2']}", 'r')
    oco_l2_snd = oco_l2['RetrievalHeader/sounding_id'][...]
    oco_l2_xco2 = oco_l2['RetrievalResults/xco2'][...]
    oco_l2_co2_prf = oco_l2['RetrievalResults/co2_profile'][...]
    oco_l2_p_level = oco_l2['RetrievalResults/vector_pressure_levels'][...]
    oco_l2_p_sfc = oco_l2['RetrievalResults/surface_pressure_fph'][...]
    oco_l2_alt_level = oco_l2['RetrievalResults/vector_altitude_levels'][...]
    oco_l2_avg_kernel = oco_l2['RetrievalResults/xco2_avg_kernel'][...]
    oco_l2_o2a_ref = oco_l2['BRDFResults/brdf_reflectance_o2'][...]
    oco_l2_wco2_ref = oco_l2['BRDFResults/brdf_reflectance_weak_co2'][...]
    oco_l2_sco2_ref = oco_l2['BRDFResults/brdf_reflectance_strong_co2'][...]
    oco_l2_aod = oco_l2['AerosolResults/aerosol_total_aod'][...]

    snd_co2 = df['snd']
    # co2_prf = df['co2_profile'][...]*1e6
    # xco2_l2 = df['xco2_L2_file'][...]
    # xco2_unpert = df['xco2_retrieved'][...]
    # aod_unpert = df['aod'][...]
    print("oco_l2_snd:", oco_l2_snd)

    l2_co2_profile = []
    l2_xco2 = []
    l2_p_level = []
    l2_p_sfc = []
    l2_alt_level = []
    l2_avg_kernel = []
    l2_o2a_ref = []
    l2_wco2_ref = []
    l2_sco2_ref = []
    l2_aod = []
    l1b_polar_ang = []
    l1b_lon, l1b_lat = [], []
    l1b_o2a, l1b_wco2, l1b_sco2 = [], [], []
    l1b_stokes, l1b_bidir_ang = [], []
    for snd in snd_co2:
        print(snd)
        print(np.where(oco_l2_snd==snd))
        if np.where(oco_l2_snd==snd)[0].size==0:
            l2_co2_profile.append(np.nan)
            l2_xco2.append(np.nan)
            l2_p_level.append(np.nan)
            l2_p_sfc.append(np.nan)
            l2_alt_level.append(np.nan)
            l2_avg_kernel.append(np.nan)
            l2_o2a_ref.append(np.nan)
            l2_wco2_ref.append(np.nan)
            l2_sco2_ref.append(np.nan)
            l2_aod.append(np.nan)
        else:
            index = np.where(oco_l2_snd==snd)[0]
            l2_co2_profile.append(oco_l2_co2_prf[index, :][0]*1e6)
            l2_xco2.append(oco_l2_xco2[index][0]*1e6)
            l2_p_level.append(oco_l2_p_level[index, :][0])
            l2_p_sfc.append(oco_l2_p_sfc[index][0])
            l2_alt_level.append(oco_l2_alt_level[index, :][0])
            l2_avg_kernel.append(oco_l2_avg_kernel[index, :][0])
            l2_o2a_ref.append(oco_l2_o2a_ref[index][0])
            l2_wco2_ref.append(oco_l2_wco2_ref[index][0])
            l2_sco2_ref.append(oco_l2_sco2_ref[index][0])
            l2_aod.append(oco_l2_aod[index][0])
            # l1b
            index_l1b = np.where(oco_l1b_id==snd)
            l1b_polar_ang.append(oco_l1b_pol_ang[index_l1b])
            l1b_lon.append(oco_l1b_lon[index_l1b])
            l1b_lat.append(oco_l1b_lat[index_l1b])
            l1b_o2a.append(oco_l1b_o2a[index_l1b])
            l1b_wco2.append(oco_l1b_wco2[index_l1b])
            l1b_sco2.append(oco_l1b_sco2[index_l1b])
            l1b_stokes.append(oco_l1b_stokes[index_l1b])
            l1b_bidir_ang.append(oco_l1b_bidir_angle[index_l1b])
    

    l2_co2_profile = np.array(l2_co2_profile)
    l2_xco2 = np.array(l2_xco2)
    l2_p_level = np.array(l2_p_level)
    l2_p_sfc = np.array(l2_p_level)
    l2_alt_level = np.array(l2_alt_level)
    l2_avg_kernel = np.array(l2_avg_kernel)
    l2_o2a_ref = np.array(l2_o2a_ref)
    l2_wco2_ref = np.array(l2_wco2_ref)
    l2_sco2_ref = np.array(l2_sco2_ref)
    l2_aod = np.array(l2_aod)

    l1b_polar_ang = np.array(l1b_polar_ang)
    l1b_lon = np.array(l1b_lon)
    l1b_lat = np.array(l1b_lat)
    l1b_o2a = np.array(l1b_o2a)
    l1b_wco2 = np.array(l1b_wco2)
    l1b_sco2 = np.array(l1b_sco2)
    l1b_stokes = np.array(l1b_stokes)
    l1b_bidir_ang = np.array(l1b_bidir_ang)
    l1b_stokes_test = np.abs(l1b_stokes[:, 0, :, 0]) + np.abs(l1b_stokes[:, 0, :, 1])

    print(f'l1b_stokes shape: {l1b_stokes.shape}')
    print(f'l1b_stokes_test shape: {l1b_stokes_test.shape}')

    print(f'l1b_o2a shape: {l1b_o2a.shape}')
    print(f'l1b_polar_ang shape: {l1b_polar_ang.shape}')
    print(f'l2_xco2 shape: {l2_xco2.shape}')

    aod_before_after_parameterization(img, wesn, lon_dom, lat_dom, df, l2_aod, img_dir=img_dir)

    """
    plt.close('all')
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(lam[1, :]*1e3, l1b_o2a[0, 0, :], 'k')
    min, max = np.nanmin(l1b_o2a[0, 0, :]), np.nanmax(l1b_o2a[0, 0, :])
    ax.vlines(lam[1, 7]*1e3, min, max, 'r')
    ax.vlines(lam[1, 151]*1e3, min, max, 'g')
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('Radiance')
    fig.tight_layout()
    plt.show()
    plt.close()

    print(f'min arg: {np.argmin(l1b_o2a[0, 0, :])}')
    print(f'max: {lam[1, 7]*1e3}')
    print(f'min: {lam[1, 151]*1e3}')

    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.scatter(l1b_o2a[:, 0, 7], l1b_polar_ang[:, 0, 0], s=30,
                c=l2_xco2, vmin=394, vmax=400, cmap='RdBu_r')
    cb = fig.colorbar(c, ax=ax)
    ax.set_xlabel('I(757.7822 nm)')
    ax.set_ylabel('Polarization angle ($^o$)')
    cb.set_label('XCO2 (ppmv)')
    fig.tight_layout()
    plt.show()
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.scatter(l1b_o2a[:, 0, 151], l1b_polar_ang[:, 0, 0], s=30,
                c=l2_xco2, vmin=394, vmax=400, cmap='RdBu_r')
    cb = fig.colorbar(c, ax=ax)
    ax.set_xlabel('I(760.2377 nm)')
    ax.set_ylabel('Polarization angle ($^o$)')
    cb.set_label('XCO2 (ppmv)')
    fig.tight_layout()
    plt.show()
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.scatter(l1b_o2a[:, 0, 151]/l1b_o2a[:, 0, 7], l1b_polar_ang[:, 0, 0], s=30, 
                c=l2_xco2, vmin=394, vmax=400, cmap='RdBu_r')
    cb = fig.colorbar(c, ax=ax)
    ax.set_xlabel('I(757.7822 nm)/I(760.2377 nm)')
    ax.set_ylabel('Polarization angle ($^o$)')
    cb.set_label('XCO2 (ppmv)')
    fig.tight_layout()
    plt.show()
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.scatter(l1b_o2a[:, 0, 7], l1b_polar_ang[:, 0, 0], s=30,
                c=l1b_stokes_test[:, 0])
    cb = fig.colorbar(c, ax=ax)
    ax.set_xlabel('I(757.7822 nm)')
    ax.set_ylabel('Polarization angle ($^o$)')
    cb.set_label('|m_I| + |m_Q|')
    fig.tight_layout()
    plt.show()
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.scatter(l1b_o2a[:, 0, 151], l1b_polar_ang[:, 0, 0], s=30,
                c=l1b_stokes_test[:, 0])
    cb = fig.colorbar(c, ax=ax)
    ax.set_xlabel('I(760.2377 nm)')
    ax.set_ylabel('Polarization angle ($^o$)')
    cb.set_label('|m_I| + |m_Q|')
    fig.tight_layout()
    plt.show()
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.scatter(l1b_o2a[:, 0, 7], l1b_polar_ang[:, 0, 0], s=30,
                c=l1b_bidir_ang[:, 0, 0])
    cb = fig.colorbar(c, ax=ax)
    ax.set_xlabel('I(757.7822 nm)')
    ax.set_ylabel('Polarization angle ($^o$)')
    cb.set_label('Bidirection angle')
    fig.tight_layout()
    plt.show()
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.scatter(l1b_o2a[:, 0, 151], l1b_polar_ang[:, 0, 0], s=30,
                c=l1b_bidir_ang[:, 0, 0])
    cb = fig.colorbar(c, ax=ax)
    ax.set_xlabel('I(760.2377 nm)')
    ax.set_ylabel('Polarization angle ($^o$)')
    cb.set_label('Bidirection angle')
    fig.tight_layout()
    plt.show()
    plt.close()
    """
    # sys.exit()

    cld_xco2 = pd.read_csv(f'{cfg_name}_footprint_cld_distance.csv')
    xco2 = cld_xco2['L2XCO2[ppm]']
    cld_dist = cld_xco2['weighted_cld_distance']
    xco2_snd = cld_xco2['SND'].apply(lambda x: x[3:]).astype(int)

    snd_co2 = df['snd'][...]
    xco2_l2 = df['xco2_L2_file'][...]
    xco2_unpert = df['xco2_retrieved'][...]
    xco2_unpert[xco2_unpert<0] = np.nan

    # scene_google_map(img, wesn, lon_dom, lat_dom, o1,
    #                 img_dir=img_dir)
    
    # with h5py.File(f'../simulation/data/{case_name_tag}/atm_abs_o2a_11.h5', 'r') as file:
    #     wvl = file['wl_oco'][...]
    #     trnsx = file['trns_oco'][...]
    #     oco_lam = file['lamx'][...]
    #     oco_tx = file['tx'][...]

    retrieval_pixel = h5py.File('full-unperturbed20191005_central_asia_pbp.h5', 'r')
    mask = retrieval_pixel['xco2_retrieved'][...]!=-2
    key_list = ['aod', 'cpu_minutes', 'lat', 'lon', 'psur_MT_file', 'psur_retrieved',
                'rfl1', 'rfl2', 'rfl3', 'snd', 'xco2_L2_file', 'xco2_retrieved', 'xco2_weighted_column']
    df_pbp = pd.DataFrame({key:retrieval_pixel[key][...] for key in key_list})
    df_pbp.replace(-2, np.nan, inplace=True)
    df_pbp.loc[df_pbp['xco2_L2_file']<1, 'xco2_L2_file'] = df_pbp.loc[df_pbp['xco2_L2_file']<1, 'xco2_L2_file']*1e6
    df_pbp.drop_duplicates('snd', inplace=True)
    df_pbp['diff_xco2'] = df_pbp['xco2_retrieved']-df_pbp['xco2_L2_file']
    
    mask = cld_dist>5
    # print(f"mean df_pixel['diff_xco2'] for cld_dist > 5: {np.mean(df_pixel['diff_xco2'][mask]):.3f} +/- {np.std(df_pixel['diff_xco2'][mask]):.3f}")
    print(f"mean df['diff_xco2'] for cld_dist > 5: {np.mean(df['diff_xco2'][mask]):.3f} +/- {np.std(df['diff_xco2'][mask]):.3f}")
    delta_XCO2_comparison_para(cld_dist, df, img_dir=img_dir)
    
    delta_XCO2_comparison_para_pbp(cld_dist, df, df_pbp, img_dir=img_dir)
        
    xco2_spread(cld_dist, xco2_l2, xco2_unpert,
                img_dir=img_dir)
    # co2_prfl_diff(co2_prf, l2_co2_profile, l2_alt_level, l2_avg_kernel,
    #               img_dir=img_dir)
    fp_pol_ang_compare(l1b_polar_ang, l1b_lon, l1b_lat,
                        img, wesn, lon_dom, lat_dom, 
                        tick_size=12, label_size=14, img_dir=img_dir)


def convert_photon_unit(data_photon, wavelength, scale_factor=2.0):
    # original: 
    # Ph sec^{-1} m^{-2} sr^{-1} um^{-1}
    
    c = 299792458.0
    h = 6.62607015e-34
    wavelength = wavelength * 1e-9
    data = data_photon/1000.0*c*h/wavelength*scale_factor

    return data

def new_get_image(self, tile):
    url = self._image_url(tile) 
    req = Request(url)
    req.add_header('User-agent', 'your bot 0.1')
    fh = urlopen(req)
    im_data = io.BytesIO(fh.read())
    fh.close()
    img = Image.open(im_data)
    img = img.convert(self.desired_tile_form)
    return img, self.tileextent(tile), 'lower'

def aod_550_plot(img, wesn, lon_dom, lat_dom, modis_lon, modis_lat, modis_aod,
                img_dir='.', label_size=14, tick_size=12):
    f, ax=plt.subplots(figsize=(8, 7))
    ax.imshow(img, extent=wesn)
    ax.vlines(lon_dom, ymin=wesn[2]+0.15, ymax=wesn[3]-0.15, color='k', linewidth=1)
    ax.hlines(lat_dom, xmin=wesn[0]+0.15, xmax=wesn[1]-0.15, color='k', linewidth=1)
    c = ax.scatter(modis_lon, modis_lat, c=modis_aod, 
                   s=5, cmap='OrRd')
    cbar = f.colorbar(c, ax=ax, extend='max')
    cbar.set_label('550 nm AOD', fontsize=label_size)
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.set_xlabel('Longitude ($^\circ$E)', fontsize=label_size)
    ax.set_ylabel('Latitude ($^\circ$N)', fontsize=label_size)
    f.tight_layout()
    f.savefig(f'{img_dir}/MODIS_550AOD.png', dpi=300)

def XCO2_l2_plot(img, wesn, lon_dom, lat_dom, df,
                 img_dir='.', label_size=16, tick_size=14):

    f, ax =plt.subplots(1, 1, figsize=(6, 6))

    ax.imshow(img, extent=wesn)
    ax.set_xlim(np.min(lon_dom), np.max(lon_dom))
    ax.set_ylim(np.min(lat_dom), np.max(lat_dom))
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.set_xlabel('Longitude ($^\circ$E)', fontsize=label_size)
    ax.set_ylabel('Latitude ($^\circ$N)', fontsize=label_size)
    ax.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 181.0, 0.1)))
    ax.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 91.0, 0.1)))
 
    mask = df['xco2_retrieved'][...]!=-2
    c = ax.scatter(df['lon'], df['lat'], 
                    c=df['xco2_L2_file'], s=30,
                    cmap='RdBu_r', vmin=394, vmax=412)
    cbar = f.colorbar(c, ax=ax, extend='both')
    cbar.set_label('$\mathrm{X_{CO2}}$ (ppm)', fontsize=label_size)
    cbar.ax.tick_params(labelsize=tick_size)
    f.tight_layout(pad=0.2)
    f.savefig(f'{img_dir}/MODIS_XCO2_l2.png', dpi=300)


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
                    cmap='RdBu_r', vmin=394, vmax=412)
    cbar1 = f.colorbar(c1, ax=ax1, extend='both')
    cbar1.set_label('Level2 $\mathrm{X_{CO2}}$ (ppm)', fontsize=label_size)
    cbar1.ax.tick_params(labelsize=tick_size)

    c2 = ax2.scatter(df['lon'], df['lat'], 
                c=df['xco2_retrieved'], s=30,
                cmap='RdBu_r', vmin=394, vmax=412)
    cbar2 = f.colorbar(c2, ax=ax2, extend='both')
    cbar2.set_label('Modified $\mathrm{X_{CO2}}$ (ppm)', fontsize=label_size)
    cbar2.ax.tick_params(labelsize=tick_size)

    c3 = ax3.scatter(df['lon'], df['lat'],
                c=df['diff_xco2'], s=30,
                cmap='RdBu_r', vmin=-2, vmax=2)
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
                    cmap='RdBu_r', vmin=394, vmax=412)
    cbar1 = f.colorbar(c1, ax=ax1, extend='both')
    cbar1.set_label('Level2 $\mathrm{X_{CO2}}$ (ppm)', fontsize=label_size)
    cbar1.ax.tick_params(labelsize=tick_size)

    c2 = ax2.scatter(df['lon'], df['lat'], 
                c=df['xco2_retrieved'], s=30,
                cmap='RdBu_r', vmin=394, vmax=412)
    cbar2 = f.colorbar(c2, ax=ax2, extend='both')
    cbar2.set_label('Modified $\mathrm{X_{CO2}}$ (ppm)', fontsize=label_size)
    cbar2.ax.tick_params(labelsize=tick_size)

    c3 = ax3.scatter(df['lon'], df['lat'],
                c=df['diff_xco2'], s=30,
                cmap='RdBu_r', vmin=-3, vmax=3)
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
                    cmap='OrRd', vmin=910, vmax=950)
    cbar1 = f.colorbar(c1, ax=ax1, extend='both')
    cbar1.set_label('Level 2 $\mathrm{P_{sfc}}$ (hPa)', fontsize=label_size)
    cbar1.ax.tick_params(labelsize=tick_size)

    c2 = ax2.scatter(df['lon'], df['lat'], 
                c=df.psur_retrieved/100, s=30,
                cmap='OrRd', vmin=910, vmax=950)
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
                    cmap='OrRd', vmin=0.1, vmax=0.4)
    cbar1 = f.colorbar(c1, ax=ax1, extend='both')
    cbar1.set_label('Level 2 AOD', fontsize=label_size)
    cbar1.ax.tick_params(labelsize=tick_size)

    c2 = ax2.scatter(df['lon'], df['lat'], 
                c=df.aod, s=30,
                cmap='OrRd', vmin=0.1, vmax=0.4)
    cbar2 = f.colorbar(c2, ax=ax2, extend='both')
    cbar2.set_label('Modified AOD', fontsize=label_size)
    cbar2.ax.tick_params(labelsize=tick_size)

    c3 = ax3.scatter(df['lon'], df['lat'],
                c=df.aod-l2_aod, s=30,
                cmap='RdBu_r', vmin=-0.2, vmax=0.2)
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

def delta_XCO2_comparison_para_pbp(cld_dist, df, df_pixel,
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
    fig.savefig(f'{img_dir}/Delta_XCO2_comparison_cld_para_test_pbp.png', dpi=300)


def delta_XCO2_comparison_para(cld_dist, df, img_dir='.', label_size=16, tick_size=14):
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4.8), sharex=False)
    fig.tight_layout(pad=5.0)
    light_jet = cm.jet#cmap_map(lambda x: x/3 + 0.66, cm.jet)

    ax1.scatter(cld_dist, df['diff_xco2'], color='k', s=15)
    ax1.tick_params(axis='both', labelsize=tick_size)
    ymin, ymax = ax1.get_ylim()
    xmin, xmax = ax1.get_xlim()

    for ax, label in zip([ax1], ['(a)']):
        # xmin, xmax = ax.get_xlim()
        # ymin, ymax = ax.get_ylim()
        ax.set_xlabel('$\mathrm{D_e}$ (km)', fontsize=label_size)
        ax.set_ylabel('$\Delta \mathrm{X_{CO2}}$ (ppm)', fontsize=label_size)
        ax.hlines(0, xmin, xmax, linestyles='--', colors='grey')
        ax.text((xmin+(xmax-xmin)*-0.05), (ymin+(ymax-ymin)*1.05), label, fontsize=20)
        ax.set_ylim(ymin, ymax)
    # ax.set_title('title', fontsize=title_size)
    fig.tight_layout(pad=2.0)
    fig.savefig(f'{img_dir}/Delta_XCO2_comparison_cld_para.png', dpi=300)

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
    ax.text((xmin+(xmax-xmin)*0.405), (ymin+(ymax-ymin)*0.90), 
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
    
    print(f'cloud dist <= {cld_dist_threshold} anomaly change: {np.nanmean((xco2_unpert-xco2_l2)[cld_mask]):.3f} +/- {np.nanstd((xco2_unpert- xco2_l2)[cld_mask]):.3f}')

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

    ax_histy.text(0.155, 8.75,
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