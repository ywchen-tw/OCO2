import os
import time
from functools import wraps

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