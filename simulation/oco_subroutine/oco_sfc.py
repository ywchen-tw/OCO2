import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate

def func(x, a):
    return a*x

def cal_sfc_alb_2d(x_ref, y_ref, data_ref, x_bkg_2d, y_bkg_2d, data_bkg_2d, scale=True, replace=True):

    logic = (x_ref>=x_bkg_2d.min()) & (x_ref<=x_bkg_2d.max()) & (y_ref>=y_bkg_2d.min()) & (y_ref<=y_bkg_2d.max())
    x_ref = x_ref[logic]
    y_ref = y_ref[logic]
    data_ref = data_ref[logic]

    points = np.transpose(np.vstack((x_bkg_2d.ravel(), y_bkg_2d.ravel())))
    data_bkg = interpolate.griddata(points, data_bkg_2d.ravel(), (x_ref, y_ref), method='nearest')

    logic_valid = (data_bkg>0.0) & (data_ref>0.0)
    x_ref = x_ref[logic_valid]
    y_ref = y_ref[logic_valid]
    data_bkg = data_bkg[logic_valid]
    data_ref = data_ref[logic_valid]

    if scale:
        popt, pcov = curve_fit(func, data_bkg, data_ref)
        slope = popt[0]
    else:
        slope = 1.0

    print('Message [cal_sfc_alb_2d]: slope:', slope)
    data_2d = data_bkg_2d*slope

    dx = x_bkg_2d[1, 0] - x_bkg_2d[0, 0]
    dy = y_bkg_2d[0, 1] - y_bkg_2d[0, 0]

    if replace:
        indices_x = np.int_(np.round((x_ref-x_bkg_2d[0, 0])/dx, decimals=0))
        indices_y = np.int_(np.round((y_ref-y_bkg_2d[0, 0])/dy, decimals=0))
        data_2d[indices_x, indices_y] = data_ref

    return data_2d