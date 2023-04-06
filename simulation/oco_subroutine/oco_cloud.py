

from genericpath import isfile
import os
import sys
import h5py
import numpy as np
import datetime

from scipy import interpolate

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from matplotlib import rcParams
import er3t
from er3t.pre.atm import atm_atmmod
from er3t.pre.abs import abs_16g
from er3t.pre.cld import cld_sat
from er3t.pre.sfc import sfc_sat
from er3t.pre.pha import pha_mie_wc as pha_mie # newly added for phase function
from er3t.util import cal_r_twostream, grid_by_extent, grid_by_lonlat, cal_ext

from er3t.rtm.mca import mca_atm_1d, mca_atm_3d, mca_sfc_2d
from er3t.rtm.mca import mcarats_ng
from er3t.rtm.mca import mca_out_ng
from er3t.rtm.mca import mca_sca # newly added for phase function


from mpl_toolkits.axes_grid1 import make_axes_locatable


class func_cot_vs_rad:
    def __init__(self,
                sat,
                modl1b,
                fdir,
                wavelength,
                sfc_albedo=0,
                cth=3, 
                ctt=250, 
                cot=np.concatenate((np.arange(0.0, 1.0, 0.1),
                                    np.arange(1.0, 10.0, 1.0),
                                    np.arange(10.0, 20.0, 2.0),
                                    np.arange(20.0, 50.0, 5.0),
                                    np.arange(50.0, 100.0, 10.0),
                                    np.arange(100.0, 200.0, 20.0),
                                    np.arange(200.0, 401.0, 50.0))),
                run=False,
                ):

        if not os.path.exists(fdir):
            os.makedirs(fdir)

        self.fdir       = fdir
        self.wavelength = wavelength
        self.cot        = cot
        self.rad        = np.array([])

        if run:
            self.run_all(sat, modl1b, sfc_albedo, cth, ctt)

        for i in range(self.cot.size):
            cot0 = self.cot[i]
            fname = '%s/mca-out-rad-3d_cot-%.2f.h5' % (self.fdir, cot0)
            out0  = mca_out_ng(fname=fname, mode='all', squeeze=True)
            self.rad = np.append(self.rad, out0.data['rad']['data'].mean())

    def run_all(self, sat, modl1b, sfc_albedo, cth, ctt):
        f = h5py.File(sat.fnames['oco_l1b'][0], 'r')
        lon_oco_l1b = f['SoundingGeometry/sounding_longitude'][...]
        lat_oco_l1b = f['SoundingGeometry/sounding_latitude'][...]
        logic = (lon_oco_l1b>=sat.extent[0]) & (lon_oco_l1b<=sat.extent[1]) & (lat_oco_l1b>=sat.extent[2]) & (lat_oco_l1b<=sat.extent[3])
        sza = f['SoundingGeometry/sounding_solar_zenith'][...][logic].mean()
        saa = f['SoundingGeometry/sounding_solar_azimuth'][...][logic].mean()
        vza = f['SoundingGeometry/sounding_zenith'][...][logic].mean()
        vaa = f['SoundingGeometry/sounding_azimuth'][...][logic].mean()
        f.close()

        for cot0 in self.cot:
            print(cot0)
            self.run_mca_one(modl1b, cot0, sza, saa, vza, vaa, sfc_albedo, cth, ctt)

    def run_mca_one(self, modl1b, cot, sza, saa, vza, vaa, sfc_albedo, cth, ctt):

        

        """
        levels    = np.linspace(0.0, 20.0, 21)
        fname_atm = '%s/atm.pk' % self.fdir
        atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=False)

        fname_abs = '%s/abs.pk' % self.fdir
        abs0      = abs_16g(wavelength=self.wavelength, fname=fname_abs, atm_obj=atm0, overwrite=False)
        """

        # atm object
        # =================================================================================
        levels    = np.array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. , 5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. , 11. , 12. , 13. , 14. , 20. , 25. , 30. , 35. , 40. ])
        fname_atm = '%s/atm.pk' % self.fdir
        atm0      = atm_atmmod(levels=levels, fname=fname_atm, overwrite=True)
        # =================================================================================

        # abs object, in the future, we will implement OCO2 MET file for this
        # =================================================================================
        fname_abs = '%s/abs.pk' % self.fdir
        abs0      = abs_16g(wavelength=650, fname=fname_abs, atm_obj=atm0, overwrite=True)
        # =================================================================================

        cot_2d    = np.zeros((2, 2), dtype=np.float64); cot_2d[...] = cot
        cer_2d    = np.zeros((2, 2), dtype=np.float64); cer_2d[...] = 12.0
        ext_3d    = np.zeros((2, 2, 2), dtype=np.float64)

        fname_cld  = '%s/cld.pk' % self.fdir
        #cld0          = cld_les(fname_nc=fname_les, fname=fname_les_pk, coarsen=[1, 1, 1, 1], overwrite=False)
        cld0      = cld_sat(sat_obj=modl1b, fname=fname_cld, cth=modl1b.data['cth_2d']['data'], cgt=0.5, dz=np.unique(atm0.lay['thickness']['data'])[0], overwrite=False)

        cld0.lev['altitude']['data']    = cld0.lay['altitude']['data'][2:5]

        cld0.lay['x']['data']           = np.array([0, 1])
        cld0.lay['y']['data']           = np.array([0, 1])
        cld0.lay['nx']['data']          = 2
        cld0.lay['ny']['data']          = 2
        cth_hist, cth_edges = np.histogram(cth, density=False, bins=25)
        cth_index = np.argmax(cth_hist)
        cld0.lay['altitude']['data']    = cth_edges[cth_index:cth_index+2]
        #cld0.lay['pressure']['data']    = cld0.lay['pressure']['data'][2:4]
        ctt_hist, ctt_edges = np.histogram(ctt, density=False, bins=25)
        ctt_index = np.argmax(ctt_hist)
        cld0.lay['temperature']['data'] = np.zeros((2, 2, 2))
        cld0.lay['temperature']['data'][:,:, 0] = ctt_edges[ctt_index]
        cld0.lay['temperature']['data'][:,:, 1] = ctt_edges[ctt_index+1]
        cld0.lay['cot']['data']         = cot_2d
        cld0.lay['thickness']['data']   = cld0.lay['thickness']['data'][2:4]

        ext_3d[:, :, 0]  = cal_ext(cot_2d, cer_2d)/(cld0.lay['thickness']['data'].sum()*1000.0)
        ext_3d[:, :, 1]  = cal_ext(cot_2d, cer_2d)/(cld0.lay['thickness']['data'].sum()*1000.0)
        cld0.lay['extinction']['data']  = ext_3d

        atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)

        fname_atm3d = '%s/mca_atm_3d.bin' % self.fdir
        atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, fname='%s/mca_atm_3d.bin' % self.fdir, overwrite=True)

        atm_1ds   = [atm1d0]
        atm_3ds   = [atm3d0]

        mca0 = mcarats_ng(
                date=datetime.datetime(2016, 8, 29),
                atm_1ds=atm_1ds,
                atm_3ds=atm_3ds,
                Ng=abs0.Ng,
                target='radiance',
                surface_albedo=sfc_albedo,
                solar_zenith_angle=sza,
                solar_azimuth_angle=-saa,
                sensor_zenith_angle=vza,
                sensor_azimuth_angle=vaa,
                fdir='%s/%.2f/les_rad_3d' % (self.fdir, cot),
                Nrun=1,
                photons=1e6,
                solver='3D',
                Ncpu=24,
                mp_mode='py',
                overwrite=True)

        out0 = mca_out_ng(fname='%s/mca-out-rad-3d_cot-%.2f.h5' % (self.fdir, cot), mca_obj=mca0, abs_obj=abs0, mode='all', squeeze=True, verbose=True)

    def interp_from_rad(self, rad, method='cubic'):

        f = interp1d(self.rad, self.cot, kind=method, bounds_error=False)

        return f(rad)

    def interp_from_cot(self, cot, method='cubic'):

        f = interp1d(self.cot, self.rad, kind=method, bounds_error=False)

        return f(cot)


def para_corr(lon0, lat0, vza, vaa, cld_h, sfc_h, R_earth=6378000.0, verbose=True):

    """
    Parallax correction for the cloud positions

    lon0: input longitude
    lat0: input latitude
    vza : sensor zenith angle [degree]
    vaa : sensor azimuth angle [degree]
    cld_h: cloud height [meter]
    sfc_h: surface height [meter]
    R_earth: earth radius [meter]
    """

    if verbose:
        print('Message [para_corr]: Please make sure the units of \'cld_h\' and \'sfc_h\' are in \'meter\'.')

    dist = (cld_h-sfc_h)*np.tan(np.deg2rad(vza))

    delta_lon = dist*np.sin(np.deg2rad(vaa)) / (np.pi*R_earth) * 180.0
    delta_lat = dist*np.cos(np.deg2rad(vaa)) / (np.pi*R_earth) * 180.0

    lon = lon0 + delta_lon
    lat = lat0 + delta_lat

    return lon, lat

def wind_corr(lon0, lat0, u, v, dt, R_earth=6378000.0, verbose=True):

    """
    Wind correction for the cloud positions

    lon0: input longitude
    lat0: input latitude
    u   : meridional wind [meter/second], positive when eastward
    v   : zonal wind [meter/second], positive when northward
    dt  : delta time [second]
    """

    if verbose:
        print('Message [wind_corr]: Please make sure the units of \'u\' and \'v\' are in \'meter/second\' and \'dt\' in \'second\'.')
        print('Message [wind_corr]: U: %.4f m/s; V: %.4f m/s; Time offset: %.2f s' % (np.median(u), np.median(v), dt))

    delta_lon = (u*dt) / (np.pi*R_earth) * 180.0
    delta_lat = (v*dt) / (np.pi*R_earth) * 180.0

    lon = lon0 + delta_lon
    lat = lat0 + delta_lat

    return lon, lat

def create_sfc_alb_2d(x_ref, y_ref, data_ref, x_bkg_2d, y_bkg_2d, data_bkg_2d, scale=True, replace=True):

    def func(x, a):
        return a*x

    points = np.transpose(np.vstack((x_bkg_2d.ravel(), y_bkg_2d.ravel())))
    data_bkg = interpolate.griddata(points, data_bkg_2d.ravel(), (x_ref, y_ref), method='nearest')

    if scale:
        popt, pcov = curve_fit(func, data_bkg, data_ref)
        slope = popt[0]
    else:
        slope = 1.0

    data_2d = data_bkg_2d*slope

    dx = x_bkg_2d[1, 0] - x_bkg_2d[0, 0]
    dy = y_bkg_2d[0, 1] - y_bkg_2d[0, 0]

    if replace:
        indices_x = np.int_(np.round((x_ref-x_bkg_2d[0, 0])/dx, decimals=0))
        indices_y = np.int_(np.round((y_ref-y_bkg_2d[0, 0])/dy, decimals=0))
        data_2d[indices_x, indices_y] = data_ref

    return data_2d

def cloud_mask_rgb(
        rgb,
        extent,
        lon_2d,
        lat_2d,
        ref_470_2d,
        alb_470,
        ref_threshold,
        frac=0.5,
        a_r=1.06,
        a_g=1.06,
        a_b=1.06,

        logic_good=None
        ):

    # Find cloudy pixels based on MODIS RGB imagery and upscale/downscale to 250m resolution
    #/----------------------------------------------------------------------------\#
    lon_rgb0 = np.linspace(extent[0], extent[1], rgb.shape[1]+1)
    lat_rgb0 = np.linspace(extent[2], extent[3], rgb.shape[0]+1)
    lon_rgb = (lon_rgb0[1:]+lon_rgb0[:-1])/2.0
    lat_rgb = (lat_rgb0[1:]+lat_rgb0[:-1])/2.0

    _r = rgb[:, :, 0]
    _g = rgb[:, :, 1]
    _b = rgb[:, :, 2]

    logic_rgb_nan0 = (_r<=(np.quantile(_r, frac)*a_r)) |\
                     (_g<=(np.quantile(_g, frac)*a_g)) |\
                     (_b<=(np.quantile(_b, frac)*a_b))
    logic_rgb_nan = np.flipud(logic_rgb_nan0).T

    if logic_good is not None:
        logic_rgb_nan[logic_good] = False

    x0_rgb = lon_rgb[0]
    y0_rgb = lat_rgb[0]
    dx_rgb = lon_rgb[1] - x0_rgb
    dy_rgb = lat_rgb[1] - y0_rgb

    indices_x = np.int_(np.round((lon_2d-x0_rgb)/dx_rgb, decimals=0))
    indices_y = np.int_(np.round((lat_2d-y0_rgb)/dy_rgb, decimals=0))

    print('indices_x:', indices_x)

    """
    print('indices_x shape:', indices_x.shape)

    logic_ref_nan = (ref_470_2d-alb_470) < ref_threshold
    indices    = np.where(logic_ref_nan!=1)
    print(logic_ref_nan.shape)
    print('indices[0] shape:', indices[0].shape)
    indices_x  = np.unique(np.concatenate((indices_x, indices[0])))
    indices_y  = np.unique(np.concatenate((indices_y, indices[1])))"""
    logic_ref_nan0 = (ref_470_2d-alb_470) < ref_threshold
    logic_ref_nan = np.logical_and(logic_rgb_nan[indices_x, indices_y], logic_ref_nan0)


    indices    = np.where(logic_ref_nan!=1)
    #\----------------------------------------------------------------------------/#

    return indices[0], indices[1]


def rgb2hsv(rgb):
    """ convert RGB to HSV color space

    :param rgb: np.ndarray
    :return: np.ndarray
    """

    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv

    return hsv


def cdata_cld_ipa(oco_band, sat0, fdir_data, fdir_cot, zpt_file, ref_threshold, photons=1e6, plot=True):

    # process wavelength
    #/----------------------------------------------------------------------------\#
    if oco_band.lower() == 'o2a':
        wvl = 650
        index_wvl = 0      # select MODIS 650 nm band radiance/reflectance for IPA cloud retrieval
        wvl_sfc = 860
        index_wvl_sfc = 1  # select MODIS 860 nm band surface albedo for scaling
    elif oco_band.lower() == 'wco2':
        wvl = 1640
        index_wvl = 5      # select MODIS 650 nm band radiance/reflectance for IPA cloud retrieval
        wvl_sfc = 1640
        index_wvl_sfc = 5  # select MODIS 860 nm band surface albedo for scaling
    elif oco_band.lower() == 'sco2':
        wvl = 2130
        index_wvl = 6      # select MODIS 650 nm band radiance/reflectance for IPA cloud retrieval
        wvl_sfc = 2130
        index_wvl_sfc = 6  # select MODIS 860 nm band surface albedo for scaling
    else:
        msg = '\nError [cdata_sat_raw]: Currently, only <oco_band=\'o2a\'> is supported.>'
        sys.exit(msg)
    #\----------------------------------------------------------------------------/#

    # read in data
    #/----------------------------------------------------------------------------\#
    f0 = h5py.File(f'{sat0.fdir_out}/pre-data.h5', 'r')
    extent = f0['extent'][...]
    ref_2d = f0['mod/rad/ref_650'][...]
    rad_2d = f0['mod/rad/rad_650'][...]
    ref_470_2d = f0['mod/rad/ref_470'][...]
    ref_550_2d = f0['mod/rad/ref_555'][...]
    rgb    = f0['mod/rgb'][...]
    cot_l2 = f0['mod/cld/cot_l2'][...]
    cer_l2 = f0['mod/cld/cer_l2'][...]
    lon_2d = f0['lon'][...]
    lat_2d = f0['lat'][...]
    cth = f0['mod/cld/cth_l2'][...]
    sfh = f0['mod/geo/sfh'][...]
    sza = f0['mod/geo/sza'][...]
    saa = f0['mod/geo/saa'][...]
    vza = f0['mod/geo/vza'][...]
    vaa = f0['mod/geo/vaa'][...]
    alb = f0['mod/sfc/alb_43_%d' % wvl_sfc][...]
    alb_470 = f0['mod/sfc/alb_43_470'][...]
    alb_oco = f0['oco/sfc/alb_%s_2d' % oco_band.lower()][...]
    u_10m = f0['oco/met/u_10m'][...]
    v_10m = f0['oco/met/v_10m'][...]
    delta_t = f0['oco/met/delta_t'][...]
    f0.close()
    #\----------------------------------------------------------------------------/#


    # cloud mask method based on rgb image and l2 data
    #/----------------------------------------------------------------------------\#
    # primary selection (over-selection of cloudy pixels is expected)
    #/--------------------------------------------------------------\#
    cld_frac0 = (np.logical_not(np.isnan(cot_l2)) & (cot_l2>0.0)).sum() / cot_l2.size
    frac0     = 1.0 - cld_frac0
    scale_factor = 1.08
    indices_x0, indices_y0 = cloud_mask_rgb(rgb, extent, lon_2d, lat_2d, ref_470_2d, alb_470,
        ref_threshold, frac=frac0, a_r=scale_factor, a_g=scale_factor, a_b=scale_factor)

    lon_cld0 = lon_2d[indices_x0, indices_y0]
    lat_cld0 = lat_2d[indices_x0, indices_y0]
    #\--------------------------------------------------------------/#

    """
    # secondary filter (remove incorrect cloudy pixels)
    #/--------------------------------------------------------------\#
    ref_cld0    = ref_2d[indices_x0, indices_y0]

    logic_nan_cth = np.isnan(cth[indices_x0, indices_y0])
    logic_nan_cot = np.isnan(cot_l2[indices_x0, indices_y0])
    logic_nan_cer = np.isnan(cer_l2[indices_x0, indices_y0])

    logic_bad = (ref_cld0<np.median(ref_cld0)) & \
                (logic_nan_cth & \
                 logic_nan_cot & \
                 logic_nan_cer)
    
    logic = np.logical_not(logic_bad)
    lon_cld = lon_cld0[logic]
    lat_cld = lat_cld0[logic]
    

    Nx, Ny = ref_2d.shape
    indices_x = indices_x0[logic]
    indices_y = indices_y0[logic]
    #"""
    lon_cld = lon_cld0
    lat_cld = lat_cld0
    

    Nx, Ny = ref_2d.shape
    indices_x = indices_x0
    indices_y = indices_y0

    #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


    # ipa retrievals
    #/----------------------------------------------------------------------------\#
    # cth_ipa0
    # get cth for new cloud field obtained from radiance thresholding
    # [indices_x[logic], indices_y[logic]] from cth from MODIS L2 cloud product
    # this is counter-intuitive but we need to account for the parallax
    # correction (approximately) that has been applied to the MODIS L2 cloud
    # product before assigning CTH to cloudy pixels we selected from reflectance
    # field, where the clouds have not yet been parallax corrected
    #/--------------------------------------------------------------\#
    data0 = np.zeros(ref_2d.shape, dtype=np.int32)
    data0[indices_x, indices_y] = 1

    data = np.zeros(cth.shape, dtype=np.int32)
    data[cth>0.0] = 1

    offset_dx, offset_dy = er3t.util.move_correlate(data0, data)
    dlon = (lon_2d[1, 0]-lon_2d[0, 0]) * offset_dx
    dlat = (lat_2d[0, 1]-lat_2d[0, 0]) * offset_dy

    lon_2d_ = lon_2d + dlon
    lat_2d_ = lat_2d + dlat
    extent_ = [extent[0]+dlon, extent[1]+dlon, extent[2]+dlat, extent[3]+dlat]

    cth_ = cth.copy()
    cth_[cth_==0.0] = np.nan

    cth_ipa0 = np.zeros_like(ref_2d)
    cth_ipa0[indices_x, indices_y] = er3t.util.find_nearest(lon_cld, lat_cld, cth_, lon_2d_, lat_2d_)
    cth_ipa0[np.isnan(cth_ipa0)] = 0.0
    #\--------------------------------------------------------------/#

    # cer_ipa0
    #/--------------------------------------------------------------\#
    cer_ipa0 = np.zeros_like(ref_2d)
    cer_ipa0[indices_x, indices_y] = er3t.util.find_nearest(lon_cld, lat_cld, cer_l2, lon_2d_, lat_2d_)
    #\--------------------------------------------------------------/#

    # cot_ipa0
    # two relationships: one for geometrically thick clouds, one for geometrically thin clouds
    # ipa relationship of reflectance vs cloud optical thickness
    #/--------------------------------------------------------------\#
    dx = np.pi*6378.1*(lon_2d[1, 0]-lon_2d[0, 0])/180.0
    dy = np.pi*6378.1*(lat_2d[0, 1]-lat_2d[0, 0])/180.0

    fdir  = '%s/ipa-%06.1fnm_thick' % (fdir_cot, 650)

    cot_ipa = np.concatenate((       \
               np.arange(0.0, 2.0, 0.5),     \
               np.arange(2.0, 30.0, 2.0),    \
               np.arange(30.0, 60.0, 5.0),   \
               np.arange(60.0, 100.0, 10.0), \
               np.arange(100.0, 201.0, 50.0) \
               ))
    print('cot_ipa shape:', cot_ipa.shape)

    f_mca_thick = er3t.rtm.mca.func_ref_vs_cot(
                    cot_ipa,
                    cer0=25.0,
                    dx=dx,
                    dy=dy,
                    fdir=fdir,
                    date=sat0.date,
                    wavelength=650,
                    surface_albedo=alb.mean(),
                    solar_zenith_angle=sza.mean(),
                    solar_azimuth_angle=saa.mean(),
                    sensor_zenith_angle=vza.mean(),
                    sensor_azimuth_angle=vaa.mean(),
                    cloud_top_height=10.0,
                    cloud_geometrical_thickness=7.0,
                    photon_number=photons,
                    solver='3d',
                    overwrite=False
                    )

    fdir  = '%s/ipa-%06.1fnm_thin' % (fdir_cot, 650)
    f_mca_thin= er3t.rtm.mca.func_ref_vs_cot(
                    cot_ipa,
                    cer0=10.0,
                    dx=dx,
                    dy=dy,
                    fdir=fdir,
                    date=sat0.date,
                    wavelength=650,
                    surface_albedo=alb.mean(),
                    solar_zenith_angle=sza.mean(),
                    solar_azimuth_angle=saa.mean(),
                    sensor_zenith_angle=vza.mean(),
                    sensor_azimuth_angle=vaa.mean(),
                    cloud_top_height=3.0,
                    cloud_geometrical_thickness=1.0,
                    photon_number=photons,
                    solver='3d',
                    overwrite=False
                    )

    ref_cld_norm = ref_2d[indices_x, indices_y]/np.cos(np.deg2rad(sza.mean()))

    logic_thick = (cth_ipa0[indices_x, indices_y] > 4.0)
    logic_thin  = (cth_ipa0[indices_x, indices_y] < 4.0)

    cot_ipa0 = np.zeros_like(ref_2d)

    cot_ipa0[indices_x[logic_thick], indices_y[logic_thick]] = f_mca_thick.get_cot_from_ref(ref_cld_norm[logic_thick])
    cot_ipa0[indices_x[logic_thin] , indices_y[logic_thin]]  = f_mca_thin.get_cot_from_ref(ref_cld_norm[logic_thin])

    logic_out = (cot_ipa0<cot_ipa[0]) | (cot_ipa0>cot_ipa[-1])
    logic_low = (logic_out) & (ref_2d<np.median(ref_2d[indices_x, indices_y]))
    logic_high = logic_out & np.logical_not(logic_low)
    cot_ipa0[logic_low]  = cot_ipa[0]
    cot_ipa0[logic_high] = cot_ipa[-1]
    #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


    # for IPA calculation (only wind correction)
    #/----------------------------------------------------------------------------\#
    # wind correction
    # calculate new lon_corr, lat_corr based on wind speed
    #/--------------------------------------------------------------\#
    print(np.nanmedian(u_10m), np.nanmedian(v_10m))
    print(np.nanmean(u_10m), np.nanmean(v_10m))
    u_wd_correct = np.zeros_like(lon_cld)
    v_wd_correct = np.zeros_like(lon_cld)
    oco_zpt = h5py.File(zpt_file, 'r')
    zpt_h_lay = oco_zpt['h_lay'][...]
    zpt_u_lay = oco_zpt['u_lay'][...]
    zpt_v_lay = oco_zpt['v_lay'][...]
    for i in range(indices_x.size):
        ix = indices_x[i]
        iy = indices_y[i]
        cth_tmp = np.max([cth_ipa0[ix, iy], zpt_h_lay[2]])
        f_h_u = interp1d(zpt_h_lay, zpt_u_lay)
        f_h_v = interp1d(zpt_h_lay, zpt_v_lay)
        u_wd_correct[i] = f_h_u(cth_tmp)
        v_wd_correct[i] = f_h_v(cth_tmp)
    lon_corr, lat_corr  = wind_corr(lon_cld, lat_cld, u_wd_correct, v_wd_correct, delta_t)
    #\--------------------------------------------------------------/#

    # perform parallax correction on cot_ipa0, cer_ipa0, and cot_ipa0
    #/--------------------------------------------------------------\#
    Nx, Ny = ref_2d.shape
    cot_ipa_ = np.zeros_like(ref_2d)
    cer_ipa_ = np.zeros_like(ref_2d)
    cth_ipa_ = np.zeros_like(ref_2d)
    cld_msk_  = np.zeros(ref_2d.shape, dtype=np.int32)
    for i in range(indices_x.size):
        ix = indices_x[i]
        iy = indices_y[i]

        lon_corr0 = lon_corr[i]
        lat_corr0 = lat_corr[i]
        ix_corr = int((lon_corr0-lon_2d[0, 0])//(lon_2d[1, 0]-lon_2d[0, 0]))
        iy_corr = int((lat_corr0-lat_2d[0, 0])//(lat_2d[0, 1]-lat_2d[0, 0]))
        if (ix_corr>=0) and (ix_corr<Nx) and (iy_corr>=0) and (iy_corr<Ny):
            cot_ipa_[ix_corr, iy_corr] = cot_ipa0[ix, iy]
            cer_ipa_[ix_corr, iy_corr] = cer_ipa0[ix, iy]
            cth_ipa_[ix_corr, iy_corr] = cth_ipa0[ix, iy]
            cld_msk_[ix_corr, iy_corr] = 1
    #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


    # for 3D calculation (parallax correction and wind correction)
    #/----------------------------------------------------------------------------\#
    # parallax correction
    # calculate new lon_corr, lat_corr based on cloud, surface and sensor geometries
    #/--------------------------------------------------------------\#
    vza_cld = vza[indices_x, indices_y]
    vaa_cld = vaa[indices_x, indices_y]
    sfh_cld = sfh[indices_x, indices_y] * 1000.0  # convert to meter from km
    cth_cld = cth_ipa0[indices_x, indices_y] * 1000.0 # convert to meter from km
    lon_corr_p, lat_corr_p = para_corr(lon_cld, lat_cld, vza_cld, vaa_cld, cth_cld, sfh_cld)
    #\--------------------------------------------------------------/#

    # wind correction
    # calculate new lon_corr, lat_corr based on wind speed
    #/--------------------------------------------------------------\#
    lon_corr, lat_corr  = wind_corr(lon_corr_p, lat_corr_p, np.nanmedian(u_10m), np.nanmedian(v_10m), delta_t)
    #\--------------------------------------------------------------/#

    # perform parallax correction on cot_ipa0, cer_ipa0, and cot_ipa0
    #/--------------------------------------------------------------\#
    Nx, Ny = ref_2d.shape
    cot_ipa = np.zeros_like(ref_2d)
    cer_ipa = np.zeros_like(ref_2d)
    cth_ipa = np.zeros_like(ref_2d)
    cld_msk  = np.zeros(ref_2d.shape, dtype=np.int32)
    for i in range(indices_x.size):
        ix = indices_x[i]
        iy = indices_y[i]

        lon_corr0 = lon_corr[i]
        lat_corr0 = lat_corr[i]
        ix_corr = int((lon_corr0-lon_2d[0, 0])//(lon_2d[1, 0]-lon_2d[0, 0]))
        iy_corr = int((lat_corr0-lat_2d[0, 0])//(lat_2d[0, 1]-lat_2d[0, 0]))
        if (ix_corr>=0) and (ix_corr<Nx) and (iy_corr>=0) and (iy_corr<Ny):
            cot_ipa[ix_corr, iy_corr] = cot_ipa0[ix, iy]
            cer_ipa[ix_corr, iy_corr] = cer_ipa0[ix, iy]
            cth_ipa[ix_corr, iy_corr] = cth_ipa0[ix, iy]
            cld_msk[ix_corr, iy_corr] = 1
    #\--------------------------------------------------------------/#

    # fill-in the empty cracks originated from parallax and wind correction
    #/--------------------------------------------------------------\#
    Npixel = 2
    percent_a = 0.7
    percent_b = 0.7
    for i in range(indices_x.size):
        ix = indices_x[i]
        iy = indices_y[i]
        if (ix>=Npixel) and (ix<Nx-Npixel) and (iy>=Npixel) and (iy<Ny-Npixel) and \
           (cot_ipa[ix, iy] == 0.0) and (cot_ipa_[ix, iy] > 0.0):
               data_cot_ipa_ = cot_ipa_[ix-Npixel:ix+Npixel, iy-Npixel:iy+Npixel]

               data_cot_ipa  = cot_ipa[ix-Npixel:ix+Npixel, iy-Npixel:iy+Npixel]
               data_cer_ipa  = cer_ipa[ix-Npixel:ix+Npixel, iy-Npixel:iy+Npixel]
               data_cth_ipa  = cth_ipa[ix-Npixel:ix+Npixel, iy-Npixel:iy+Npixel]

               logic_cld0 = (data_cot_ipa_>0.0)
               logic_cld  = (data_cot_ipa>0.0)

               if (logic_cld0.sum() > int(percent_a * logic_cld0.size)) and \
                  (logic_cld.sum()  > int(percent_b * logic_cld.size)):
                   cot_ipa[ix, iy] = data_cot_ipa[logic_cld].mean()
                   cer_ipa[ix, iy] = data_cer_ipa[logic_cld].mean()
                   cth_ipa[ix, iy] = data_cth_ipa[logic_cld].mean()
                   cld_msk[ix, iy] = 1
    #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


    # write cot_ipa into file
    #/----------------------------------------------------------------------------\#
    f0 = h5py.File(f'{sat0.fdir_out}/pre-data.h5', 'r+')
    try:
        f0['mod/cld/cot_ipa'] = cot_ipa
        f0['mod/cld/cer_ipa'] = cer_ipa
        f0['mod/cld/cth_ipa'] = cth_ipa
        f0['mod/cld/cot_ipa0'] = cot_ipa_
        f0['mod/cld/cer_ipa0'] = cer_ipa_
        f0['mod/cld/cth_ipa0'] = cth_ipa_
        f0['mod/cld/logic_cld'] = (cld_msk==1)
        f0['mod/cld/logic_cld0'] = (cld_msk_==1)
    except:
        del(f0['mod/cld/cot_ipa'])
        del(f0['mod/cld/cer_ipa'])
        del(f0['mod/cld/cth_ipa'])
        del(f0['mod/cld/cot_ipa0'])
        del(f0['mod/cld/cer_ipa0'])
        del(f0['mod/cld/cth_ipa0'])
        del(f0['mod/cld/logic_cld'])
        del(f0['mod/cld/logic_cld0'])
        f0['mod/cld/cot_ipa'] = cot_ipa
        f0['mod/cld/cer_ipa'] = cer_ipa
        f0['mod/cld/cth_ipa'] = cth_ipa
        f0['mod/cld/cot_ipa0'] = cot_ipa0
        f0['mod/cld/cer_ipa0'] = cer_ipa0
        f0['mod/cld/cth_ipa0'] = cth_ipa0
        f0['mod/cld/logic_cld'] = (cld_msk==1)
        f0['mod/cld/logic_cld0'] = (cld_msk_==1)
    try:
        g0 = f0.create_group('cld_msk')
        g0['indices_x0'] = indices_x0
        g0['indices_y0'] = indices_y0
        g0['indices_x']  = indices_x
        g0['indices_y']  = indices_y
    except:
        del(f0['cld_msk/indices_x0'])
        del(f0['cld_msk/indices_y0'])
        del(f0['cld_msk/indices_x'])
        del(f0['cld_msk/indices_y'])
        del(f0['cld_msk'])
        g0 = f0.create_group('cld_msk')
        g0['indices_x0'] = indices_x0
        g0['indices_y0'] = indices_y0
        g0['indices_x']  = indices_x
        g0['indices_y']  = indices_y
    try:
        g0 = f0.create_group('mca_ipa_thick')
        g0['cot'] = f_mca_thick.cot
        g0['ref'] = f_mca_thick.ref
        g0['ref_std'] = f_mca_thick.ref_std
        g0 = f0.create_group('mca_ipa_thin')
        g0['cot'] = f_mca_thin.cot
        g0['ref'] = f_mca_thin.ref
        g0['ref_std'] = f_mca_thin.ref_std
    except:
        del(f0['mca_ipa_thick/cot'])
        del(f0['mca_ipa_thick/ref'])
        del(f0['mca_ipa_thick/ref_std'])
        del(f0['mca_ipa_thick'])
        del(f0['mca_ipa_thin/cot'])
        del(f0['mca_ipa_thin/ref'])
        del(f0['mca_ipa_thin/ref_std'])
        del(f0['mca_ipa_thin'])
        g0 = f0.create_group('mca_ipa_thick')
        g0['cot'] = f_mca_thick.cot
        g0['ref'] = f_mca_thick.ref
        g0['ref_std'] = f_mca_thick.ref_std
        g0 = f0.create_group('mca_ipa_thin')
        g0['cot'] = f_mca_thin.cot
        g0['ref'] = f_mca_thin.ref
        g0['ref_std'] = f_mca_thin.ref_std
    try:
        g0 = f0.create_group('cld_corr')
        g0['lon_ori'] = lon_cld
        g0['lat_ori'] = lat_cld
        g0['lon_corr_p'] = lon_corr_p
        g0['lat_corr_p'] = lat_corr_p
        g0['lon_corr'] = lon_corr
        g0['lat_corr'] = lat_corr
    except:
        del(f0['cld_corr/lon_ori'])
        del(f0['cld_corr/lat_ori'])
        del(f0['cld_corr/lon_corr_p'])
        del(f0['cld_corr/lat_corr_p'])
        del(f0['cld_corr/lon_corr'])
        del(f0['cld_corr/lat_corr'])
        del(f0['cld_corr'])
        g0 = f0.create_group('cld_corr')
        g0['lon_ori'] = lon_cld
        g0['lat_ori'] = lat_cld
        g0['lon_corr_p'] = lon_corr_p
        g0['lat_corr_p'] = lat_corr_p
        g0['lon_corr'] = lon_corr
        g0['lat_corr'] = lat_corr
    f0.close()
    #\----------------------------------------------------------------------------/#

    if plot:

        # figure
        #/----------------------------------------------------------------------------\#
        plt.close('all')
        rcParams['font.size'] = 12
        fig = plt.figure(figsize=(16, 16))

        fig.suptitle('MODIS Cloud Re-Processing')

        # RGB
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(441)
        cs = ax1.imshow(rgb, zorder=0, extent=extent)
        ax1.set_title('RGB Imagery')

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cax.axis('off')
        #\--------------------------------------------------------------/#

        # L1B reflectance
        #/----------------------------------------------------------------------------\#
        ax2 = fig.add_subplot(442)
        cs = ax2.imshow(ref_2d.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=1.0)
        ax2.set_title('L1B Reflectance (%d nm)' % wvl)

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cloud mask (primary)
        #/----------------------------------------------------------------------------\#
        ax3 = fig.add_subplot(443)
        cs = ax3.imshow(rgb, zorder=0, extent=extent)
        ax3.scatter(lon_2d[indices_x0, indices_y0], lat_2d[indices_x0, indices_y0], s=0.1, c='r', alpha=0.1)
        ax3.set_title('Primary Cloud Mask')

        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', '5%', pad='3%')
        cax.axis('off')
        #\----------------------------------------------------------------------------/#

        # cloud mask (final)
        #/----------------------------------------------------------------------------\#
        ax4 = fig.add_subplot(444)
        cs = ax4.imshow(rgb, zorder=0, extent=extent)
        ax4.scatter(lon_2d[indices_x, indices_y], lat_2d[indices_x, indices_y], s=0.1, c='r', alpha=0.1)
        ax4.set_title('Secondary Cloud Mask (Final)')

        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', '5%', pad='3%')
        cax.axis('off')
        #\----------------------------------------------------------------------------/#

        # cot l2
        #/----------------------------------------------------------------------------\#
        ax5 = fig.add_subplot(445)
        cs = ax5.imshow(cot_l2.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=50.0)
        ax5.set_title('L2 COT')

        divider = make_axes_locatable(ax5)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cer l2
        #/----------------------------------------------------------------------------\#
        ax6 = fig.add_subplot(446)
        cs = ax6.imshow(cer_l2.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=30.0)
        ax6.set_title('L2 CER [$\mu m$]')

        divider = make_axes_locatable(ax6)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cth l2
        #/----------------------------------------------------------------------------\#
        ax7 = fig.add_subplot(447)
        cs = ax7.imshow(cth.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=15.0)
        ax7.set_title('L2 CTH [km]')

        divider = make_axes_locatable(ax7)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cot ipa0
        #/----------------------------------------------------------------------------\#
        ax9 = fig.add_subplot(449)
        cs = ax9.imshow(cot_ipa0.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=50.0)
        ax9.set_title('New IPA COT')

        divider = make_axes_locatable(ax9)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cer ipa0
        #/----------------------------------------------------------------------------\#
        ax10 = fig.add_subplot(4, 4, 10)
        cs = ax10.imshow(cer_ipa0.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=30.0)
        ax10.set_title('New L2 CER [$\mu m$]')

        divider = make_axes_locatable(ax10)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cth ipa0
        #/----------------------------------------------------------------------------\#
        ax11 = fig.add_subplot(4, 4, 11)
        cs = ax11.imshow(cth_ipa0.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=15.0)
        ax11.set_title('New L2 CTH [km]')

        divider = make_axes_locatable(ax11)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cot_ipa
        #/----------------------------------------------------------------------------\#
        ax13 = fig.add_subplot(4, 4, 13)
        cs = ax13.imshow(cot_ipa.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=50.0)
        ax13.set_title('New IPA COT (Para. Corr.)')

        divider = make_axes_locatable(ax13)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cer_ipa
        #/----------------------------------------------------------------------------\#
        ax14 = fig.add_subplot(4, 4, 14)
        cs = ax14.imshow(cer_ipa.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=30.0)
        ax14.set_title('New L2 CER [$\mu m$] (Para. Corr.)')

        divider = make_axes_locatable(ax14)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # cth_ipa
        #/----------------------------------------------------------------------------\#
        ax15 = fig.add_subplot(4, 4, 15)
        cs = ax15.imshow(cth_ipa.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=15.0)
        ax15.set_title('New L2 CTH [km] (Para. Corr.)')

        divider = make_axes_locatable(ax15)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#

        # surface albedo (MYD43A3, white sky albedo)
        #/----------------------------------------------------------------------------\#
        ax16 = fig.add_subplot(4, 4, 16)
        cs = ax16.imshow(alb_oco.T, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=0.0, vmax=0.4)
        ax16.set_title('43A3 WSA (filled and scaled)')

        divider = make_axes_locatable(ax16)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        #\----------------------------------------------------------------------------/#
        ax_list = [f'ax{num}' for num in range(1, 17)]
        ax_list.remove('ax8')
        ax_list.remove('ax12')
        for num in range(len(ax_list)):
            ax = vars()[ax_list[num]]
            ax.set_xlim((extent[:2]))
            ax.set_ylim((extent[2:]))
            ax.set_xlabel('Longitude [$^\circ$]')
            ax.set_ylabel('Latitude [$^\circ$]')

        # save figure
        #/--------------------------------------------------------------\#
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s/<%s>.png' % (sat0.fdir_out, _metadata['Function']), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        #\----------------------------------------------------------------------------/#




if __name__ == '__main__':
    None


    



