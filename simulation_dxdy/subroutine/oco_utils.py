import os
import time
from functools import wraps
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpl_img
from matplotlib import rcParams

plt.rcParams["font.family"] = "Arial"

def path_dir(path_dir):
    """
    Description:
        Create a directory if it does not exist.
    Return:
        path_dir: path of the directory
    """
    abs_path = os.path.abspath(path_dir)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
    return abs_path

class sat_tmp:

    def __init__(self, data):

        self.data = data

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r took: %.4f min (%.4f h)' % \
          (f.__name__, (te-ts)/60, (te-ts)/3600))
        return result
    return wrap

def plot_mca_simulation(sat, modl1b, out0, oco_std0,
                         solver, fdir, cth, scale_factor, wavelength):
    mod_img = mpl_img.imread(sat.fnames['mod_rgb'][0])
    mod_img_wesn = sat.extent
    fig = plt.figure(figsize=(12, 5.5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(mod_img, extent=mod_img_wesn)
    ax2.imshow(mod_img, extent=mod_img_wesn)
    ax1.pcolormesh(modl1b.data['lon_2d']['data'], modl1b.data['lat_2d']['data'], 
                   modl1b.data['rad_2d']['data'], 
                   cmap='Greys_r', vmin=0.0, vmax=0.3, zorder=0)
    scatter_arg = {'s':20, 'c':oco_std0.data['xco2']['data'], 'cmap':'jet',
                   'alpha':0.4, 'zorder':1}
    ax1.scatter(oco_std0.data['lon']['data'], oco_std0.data['lat']['data'], **scatter_arg)
    ax1.set_title('MODIS Chanel 1')
    ax2.pcolormesh(modl1b.data['lon_2d']['data'], modl1b.data['lat_2d']['data'], 
                   out0.data['rad']['data'], 
                   cmap='Greys_r', zorder=0)
    ax2.scatter(oco_std0.data['lon']['data'], oco_std0.data['lat']['data'], **scatter_arg)
    ax2.set_title('MCARaTS %s' % solver)
    for ax in [ax1, ax2]:
        ax.set_xlabel('Longitude ($^\circ$)')
        ax.set_ylabel('Latitude ($^\circ$)')
        ax.set_xlim(sat.extent[:2])
        ax.set_ylim(sat.extent[2:])
    plt.subplots_adjust(hspace=0.5)
    if cth is not None:
        plt.savefig('%s/mca-out-rad-modis-%s_cth-%.2fkm_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), cth, scale_factor, wavelength), bbox_inches='tight')
    else:
        plt.savefig('%s/mca-out-rad-modis-%s_sf-%.2f_%.4fnm.png' % (fdir, solver.lower(), scale_factor, wavelength), bbox_inches='tight')
    plt.close(fig)