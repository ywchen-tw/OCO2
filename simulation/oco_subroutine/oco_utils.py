import os

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