import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt

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
    return result



def save_h5_info(cfg, index, filename):
    """
    Save the output h5 name into cfg file
    """
    cfg_file = pd.read_csv(cfg, header=None, index_col=0)
    cfg_file.loc[index, 1] = filename
    cfg_file.to_csv(cfg, header=False)
    return None



def check_h5_info(cfg, index):
    """
    Check whether the output h5 name is saved in cfg file
    """
    try: 
        cfg_file = grab_cfg(cfg)
    except OSError as err:
        print('{} not exists!'.format(cfg))
        return None
    if index in cfg_file.keys():
        if cfg_file[index][-2:] == 'h5':
            print('Output file {} exists.'.format(cfg_file[index]))
            return True
        return False
    else:
        return False

def output_h5_info(cfg, index):
    """
    Get the output h5 name is in cfg file
    """
    try: 
        cfg_file = grab_cfg(cfg)
    except OSError as err:
        print('{} not exists!'.format(cfg))
        return None
    if index in cfg_file.keys():
        if cfg_file[index][-2:] == 'h5':
            print('Output file {} exists.'.format(cfg_file[index]))
            return cfg_file[index]
        else:
            raise IOError(f'No h5 file for the {index}.')
    else:
        raise IOError(f'No {index} in the cfg file.')




def save_subdomain_info(cfg, subdomain):
    """
    Save the subdomain info into cfg file
    """
    cfg_file = pd.read_csv(cfg, header=None, index_col=0)
    for j in range(4):
        cfg_file.loc['subdomain', j+1] = subdomain[j]
    cfg_file.to_csv(cfg, header=False)
    return None

def nan_array(shape, dtype):
    tmp = np.zeros(shape, dtype=dtype)
    tmp.fill(np.nan)
    return tmp

def ax_lon_lat_label(ax, label_size=14, tick_size=12):
    ax.set_xlabel('Longitude ($^\circ$E)', fontsize=label_size)
    ax.set_ylabel('Latitude ($^\circ$N)', fontsize=label_size)
    ax.tick_params(axis='both', labelsize=tick_size)


if __name__ == '__main__':
    None


    



