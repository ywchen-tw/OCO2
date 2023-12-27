import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pickle
from er3t.util.modis import modis_l1b, modis_l2, modis_03, modis_09a1, modis_43a3, download_modis_https, get_sinusoidal_grid_tag, get_filename_tag, download_modis_rgb
from er3t.util.daac import download_oco2_https
import h5py


class satellite_download:

    def __init__(
            self,
            date=None,
            extent=None,
            fname=None,
            fdir_out='data',
            overwrite=False,
            quiet=False,
            verbose=False):

        self.date     = date
        self.extent   = extent
        self.extent_simulation = extent
        self.fdir_out = fdir_out
        self.quiet    = quiet
        self.verbose  = verbose

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif (((date is not None) and (extent is not None)) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             (((date is not None) and (extent is not None)) and (fname is not None) and (not os.path.exists(fname))):

            self.run()
            self.dump(fname)

        elif (((date is not None) and (extent is not None)) and (fname is None)):

            self.run()

        else:

            sys.exit('Error   [satellite_download]: Please check if \'%s\' exists or provide \'date\' and \'extent\' to proceed.' % fname)

    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'fnames') and hasattr(obj, 'extent') and hasattr(obj, 'fdir_out') and hasattr(obj, 'date'):
                if self.verbose:
                    print('Message [satellite_download]: Loading %s ...' % fname)
                self.date     = obj.date
                self.extent   = obj.extent
                self.fnames   = obj.fnames
                self.fdir_out = obj.fdir_out
            else:
                sys.exit('Error   [satellite_download]: File \'%s\' is not the correct pickle file to load.' % fname)

    def run(self, run=True):

        lon = np.array(self.extent[:2])
        lat = np.array(self.extent[2:])

        self.fnames = {}

        self.fnames['mod_rgb'] = [download_modis_rgb(self.date, self.extent, fdir=self.fdir_out, which='aqua', coastline=True)]

        # MODIS Level 2 Cloud Product and MODIS 03 geo file
        self.fnames['mod_l2'] = []
        self.fnames['mod_02_1km'] = []
        self.fnames['mod_02_hkm'] = []
        self.fnames['mod_02'] = []
        self.fnames['mod_03'] = []
        self.fnames['mod_04'] = []

        modis_parent_fdir = '/Users/yuch8913/programming/oco/simulation/data/modis'
        if not os.path.exists(modis_parent_fdir):
            os.makedirs(modis_parent_fdir)
        modis_fdir = f"{modis_parent_fdir}/{self.date.strftime('%Y%m%d')}"
        if not os.path.exists(modis_fdir):
            os.makedirs(modis_fdir)

        filename_tags_03 = get_filename_tag(self.date, lon, lat, satID='aqua')
        for filename_tag in filename_tags_03:
            fnames_l2 = download_modis_https(self.date, '61/MYD06_L2', filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            fnames_02_1km = download_modis_https(self.date, '61/MYD021KM', filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            fnames_02_hkm = download_modis_https(self.date, '61/MYD02HKM', filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            fnames_02 = download_modis_https(self.date, '61/MYD02QKM', filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            fnames_03 = download_modis_https(self.date, '61/MYD03'   , filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            fnames_04 = download_modis_https(self.date, '61/MYD04_L2'   , filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            self.fnames['mod_l2'] += fnames_l2
            self.fnames['mod_02_1km'] += fnames_02_1km
            self.fnames['mod_02_hkm'] += fnames_02_hkm
            self.fnames['mod_02'] += fnames_02
            self.fnames['mod_03'] += fnames_03
            self.fnames['mod_04'] += fnames_04

        # MOD09A1 surface reflectance product
        self.fnames['mod_09'] = []
        filename_tags_09 = get_sinusoidal_grid_tag(lon, lat)
        for filename_tag in filename_tags_09:
            fnames_09 = download_modis_https(self.date, '61/MOD09A1', filename_tag, day_interval=8, fdir_out=modis_fdir, run=run)
            self.fnames['mod_09'] += fnames_09
        # MOD43A3 surface reflectance product
        self.fnames['mcd_43'] = []
        filename_tags_43 = get_sinusoidal_grid_tag(lon, lat)
        for filename_tag in filename_tags_43:
            fnames_43 = download_modis_https(self.date, '61/MCD43A3', filename_tag, day_interval=16, fdir_out=modis_fdir, run=run)
            self.fnames['mcd_43'] += fnames_43

        # OCO2 std and met file
        self.fnames['oco_std'] = []
        self.fnames['oco_met'] = []
        self.fnames['oco_l1b'] = []
        self.fnames['oco_co2prior'] = []
        self.fnames['oco_imap'] = []
        oco_fdir = '/Users/yuch8913/programming/oco/simulation/data/oco'
        if not os.path.exists(oco_fdir):
            os.makedirs(oco_fdir)
        for filename_tag in filename_tags_03:
            dtime = datetime.datetime.strptime(filename_tag, 'A%Y%j.%H%M') + datetime.timedelta(minutes=7.0)
            fnames_std = download_oco2_https(dtime, 'OCO2_L2_Standard.10r', fdir_out=oco_fdir, run=run)
            fnames_met = download_oco2_https(dtime, 'OCO2_L2_Met.10r'     , fdir_out=oco_fdir, run=run)
            fnames_l1b = download_oco2_https(dtime, 'OCO2_L1B_Science.10r', fdir_out=oco_fdir, run=run)
            fnames_co2prior = download_oco2_https(dtime, 'OCO2_L2_CO2Prior.10r', fdir_out=oco_fdir, run=run)
            fnames_imap = download_oco2_https(dtime, 'OCO2_L2_IMAPDOAS.10r', fdir_out=oco_fdir, run=run)
            self.fnames['oco_std'] += fnames_std
            self.fnames['oco_met'] += fnames_met
            self.fnames['oco_l1b'] += fnames_l1b
            self.fnames['oco_co2prior'] += fnames_co2prior
            self.fnames['oco_imap'] += fnames_imap

    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [satellite_download]: Saving object into %s ...' % fname)
            pickle.dump(self, f)

def grab_cfg(path):
    """
    Read the setting information in the assigned csv file.

    path: relative or absolute path to the setting csv file.
    """
    cfg_file = pd.read_csv(path, header=None, index_col=0)
    result = {'cfg_name':path.split('/')[-1].replace('.csv', ''), 
              'cfg_path':path}
    for ind in cfg_file.index.dropna():
        contents = [str(i) for i in cfg_file.loc[ind].dropna() if str(i)[0] != '#']
        if len(contents) == 1:
            result[ind] = contents[0]
        elif len(contents) > 1:
            result[ind] = contents
    # check whether julian day exists and is correct
    date   = datetime.datetime(int(result['date'][:4]),    # year
                               int(result['date'][4:6]),   # month
                               int(result['date'][6:])     # day
                              )
    if 'juld' in result.keys():
        if not result['juld'] == date.timetuple().tm_yday:
            result['juld'] = date.timetuple().tm_yday
            save_h5_info(path, 'juld', date.timetuple().tm_yday)
    else: 
        save_h5_info(path, 'juld', date.timetuple().tm_yday)
        result['juld'] = date.timetuple().tm_yday 

    return result

def save_h5_info(cfg, index, filename):
    """
    Save the output h5 name into cfg file
    """
    cfg_file = pd.read_csv(cfg, header=None, index_col=0)
    cfg_file.loc[index, 1] = filename
    cfg_file.to_csv(cfg, header=False)
    return None


def get_sat(cfg):
    cfg_info = grab_cfg(cfg)
    # define date and region to study
    # ===============================================================
    date   = datetime.datetime(int(cfg_info['date'][:4]),    # year
                               int(cfg_info['date'][4:6]),   # month
                               int(cfg_info['date'][6:])     # day
                              )
    extent = [float(loc) for loc in cfg_info['subdomain']]
    extent[0] -= 0.15 
    extent[1] += 0.15
    extent[2] -= 0.15 
    extent[3] += 0.15
    print(extent)
    ref_threshold = float(cfg_info['ref_threshold'])

    name_tag = '%s_%s' % (cfg_info['cfg_name'], date.strftime('%Y%m%d'))
    # ===============================================================

    # create data/australia_YYYYMMDD directory if it does not exist
    # ===============================================================
    if not os.path.exists('data'):
        os.makedirs('data')
    fdir_data = os.path.abspath('data/%s' % name_tag)
    if not os.path.exists(fdir_data):
        os.makedirs(fdir_data)
    # ===============================================================

    # download satellite data based on given date and region
    # ===============================================================
    fname_sat = '%s/sat.pk' % fdir_data
    sat0 = satellite_download(date=date, fdir_out=fdir_data, extent=extent, fname=fname_sat, overwrite=False)

    return sat0