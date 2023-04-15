import pandas as pd
import datetime

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



def save_subdomain_info(cfg, subdomain):
    """
    Save the subdomain info into cfg file
    """
    cfg_file = pd.read_csv(cfg, header=None, index_col=0)
    for j in range(4):
        cfg_file.loc['subdomain', j+1] = subdomain[j]
    cfg_file.to_csv(cfg, header=False)
    return None

if __name__ == '__main__':
    None


    



