import h5py
import numpy as np
import datetime
from pyhdf.SD import SD, SDC

def cal_sat_delta_t(sat):

    # estimate average OCO-2 passing time for the scene
    #/----------------------------------------------------------------------------\#
    f = h5py.File(sat.fnames['oco_l1b'][0], 'r')
    lon_oco_l1b = f['SoundingGeometry/sounding_longitude'][...]
    lat_oco_l1b = f['SoundingGeometry/sounding_latitude'][...]
    logic = (lon_oco_l1b>=sat.extent[0]) & (lon_oco_l1b<=sat.extent[1]) & (lat_oco_l1b>=sat.extent[2]) & (lat_oco_l1b<=sat.extent[3])
    utc_oco_byte = f['SoundingGeometry/sounding_time_string'][...][logic]
    f.close()
    utc_oco = np.zeros(utc_oco_byte.size, dtype=np.float64)
    for i, utc_oco_byte0 in enumerate(utc_oco_byte):
        utc_oco_str0 = utc_oco_byte0.decode('utf-8').split('.')[0]
        utc_oco[i] = (datetime.datetime.strptime(utc_oco_str0, '%Y-%m-%dT%H:%M:%S')-datetime.datetime(1993, 1, 1)).total_seconds()
    #\----------------------------------------------------------------------------/#

    # estimate average MODIS passing time for the scene
    #/----------------------------------------------------------------------------\#
    f = SD(sat.fnames['mod_03'][0], SDC.READ)
    lon_mod = f.select('Longitude')[:][::10, :]
    lat_mod = f.select('Latitude')[:][::10, :]
    utc_mod = f.select('SD start time')[:]
    f.end()
    logic = (lon_mod>=sat.extent[0]) & (lon_mod<=sat.extent[1]) & (lat_mod>=sat.extent[2]) & (lat_mod<=sat.extent[3])
    logic = (np.sum(logic, axis=1)>0)
    utc_mod = utc_mod[logic]
    #\----------------------------------------------------------------------------/#

    return utc_oco.mean()-utc_mod.mean()