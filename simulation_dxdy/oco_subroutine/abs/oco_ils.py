import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

def oco_ils(iband, sat):

    # Get real ILS from l1b sounding data
    with h5py.File(sat.fnames['oco_l1b'][0], 'r') as f:
        del_lambda = f['InstrumentHeader/ils_delta_lambda'][...]
        rel_lresponse = f['InstrumentHeader/ils_relative_response'][...]

        del_lambda_mean = del_lambda[iband, :, :, :].mean(axis=(0, 1))
        rel_lresponse_mean = rel_lresponse[iband, :, :, :].mean(axis=(0, 1))

    xx = del_lambda_mean*1000 # micron to nm
    yy = rel_lresponse_mean/(rel_lresponse_mean).max()

    print('xx shape', xx.shape, file=sys.stderr)
    return np.array(xx), np.array(yy)
